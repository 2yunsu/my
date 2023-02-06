from datetime import datetime

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.models import resnet50
import torch
import torch.nn.functional as F
from torch import nn
import math
from torch.optim import SGD
# from tqdm import tqdm

from ImageReader import ImageReader
from Sampler import MPerClassSampler

import argparse
from torch.utils.tensorboard import SummaryWriter


class FeatureExtractor(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.feature = []
        for name, module in resnet50(weights='ResNet50_Weights.DEFAULT').named_children():
            if isinstance(module, nn.Linear):
                continue
            self.feature.append(module)
        self.feature = nn.Sequential(*self.feature)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Refactor Layer
        self.refactor = nn.Linear(2048, feature_dim)

    def forward(self, x):
        x = self.feature(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = F.layer_norm(x, [x.size(-1)])
        x = self.refactor(x)

        return x


class ProxyLinear(nn.Module):
    def __init__(self, in_features, out_features, scale):
        super(ProxyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.scale = scale

    def forward(self, x):
        output = F.normalize(x).matmul(F.normalize(self.weight).t())
        return output * self.scale


class SoftLabelLoss(nn.Module):
    def __init__(self, dim=-1):
        super(SoftLabelLoss, self).__init__()
        self.dim = dim

    def forward(self, pred, target):
        """
        :param pred: 신경망이 출력하는 배치 x class num의 행렬
        :param target: 정답 레이블 배치 x class num의 행렬, s.t. sum(target[0]) = 1 i.e. softlabel
        :return: uniformity_loss value
        """
        pred = pred.log_softmax(dim=self.dim)
        return torch.mean(torch.sum(-target * pred, dim=self.dim))


class Runner:
    def __init__(self, _args, _writer):

        self.writer = _writer
        self.data_path = _args.data_path
        self.data_name = _args.data_name
        self.crop_type = _args.crop_type
        self.num_epochs = _args.num_epochs
        self.inner_epochs = _args.inner_epochs
        self.soft_k = _args.soft_k
        self.label_policy = _args.label_policy
        self.batch_size = _args.batch_size
        self.num_sample = _args.batch_size
        self.feature_dim = _args.feature_dim
        self.scale_hyperparm = _args.scale_hyperparm
        self.lr = _args.lr
        self.lr_gamma = _args.lr_gamma
        self.recalls = [int(k) for k in _args.recalls.split(',')]

        self.device = torch.device(
            "cuda:" + str(_args.device) if torch.cuda.is_available() else "cpu")  # torch.device("cpu") #

        # dataset prepare
        train_dataset = ImageReader(self.data_path, self.data_name, 'train', self.crop_type)

        train_sample = MPerClassSampler(train_dataset.labels, self.batch_size, self.num_sample)
        self.train_data_loader = DataLoader(train_dataset, batch_sampler=train_sample,
                                            num_workers=4)

        if self.data_name == 'isc':
            self.test_dataset = ImageReader(self.data_path, self.data_name, 'query', self.crop_type)
            self.gallery_dataset = ImageReader(self.data_path, self.data_name, 'gallery', self.crop_type)
            self.gallery_data_loader = DataLoader(self.gallery_dataset, self.batch_size, shuffle=False, num_workers=4)
        else:
            self.test_dataset = ImageReader(self.data_path, self.data_name, 'test', self.crop_type)
        self.test_data_loader = DataLoader(self.test_dataset, self.batch_size, shuffle=False, num_workers=4)

        self.num_class = len(train_dataset.class_to_idx)
        self.feature_extractor = FeatureExtractor(self.feature_dim).to(self.device)
        self.classifier = ProxyLinear(self.feature_dim, self.num_class, self.scale_hyperparm).to(self.device)

        self.model = nn.Sequential(self.feature_extractor, self.classifier).to(self.device)
        self.loss_criterion = nn.CrossEntropyLoss()
        self.loss_ssl = SoftLabelLoss()

        self.optimizer_init = SGD(
            [{'params': self.feature_extractor.refactor.parameters()}, {'params': self.classifier.parameters()}],
            lr=self.lr, momentum=0.9, weight_decay=1e-4)
        self.optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        self.optimizer_magnet = SGD(
            [{'params': self.feature_extractor.feature.parameters()},
             {'params': self.feature_extractor.refactor.parameters()}],
            lr=self.lr, momentum=0.9, weight_decay=1e-4)
        self.lr_scheduler = StepLR(self.optimizer, step_size=15, gamma=self.lr_gamma)

    def train(self, net, _optim, _label_mat=None):
        net.train()
        total_loss = 0
        total_num = 0

        for inputs, labels in self.train_data_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            _optim.zero_grad()
            output = net.forward(inputs)
            if _label_mat is not None:
                _loss = self.loss_ssl(output, _label_mat[labels])
            else:
                _loss = self.loss_criterion(output, labels)
            _loss.backward()
            _optim.step()
            total_loss += _loss.item() * inputs.shape[0]
            total_num += inputs.shape[0]
        return total_loss / total_num

    def recall(self, feature_vectors, feature_labels, rank, gallery_vectors=None, gallery_labels=None, binary=False):
        num_features = len(feature_labels)
        feature_labels = torch.tensor(feature_labels, device=feature_vectors.device)
        gallery_vectors = feature_vectors if gallery_vectors is None else gallery_vectors

        sim_matrix = torch.mm(feature_vectors, gallery_vectors.t().contiguous())
        if binary:
            sim_matrix = sim_matrix / feature_vectors.size(-1)

        if gallery_labels is None:
            sim_matrix.fill_diagonal_(0)
            gallery_labels = feature_labels
        else:
            gallery_labels = torch.tensor(gallery_labels, device=feature_vectors.device)

        idx = sim_matrix.topk(k=rank[-1], dim=-1, largest=True)[1]
        acc_list = []
        for r in rank:
            correct = (gallery_labels[idx[:, 0:r]] == feature_labels.unsqueeze(dim=-1)).any(dim=-1).float()
            acc_list.append((torch.sum(correct) / num_features).item())
        return acc_list

    def test(self, net, recall_ids):
        net.eval()
        with torch.no_grad():
            if self.data_name == 'isc':
                test_feature = []
                for inputs, labels in self.test_data_loader:
                    features = net.forward(inputs.to(self.device))
                    test_feature.append(features)
                test_feature = torch.cat(test_feature, dim=0)
                test_feature = torch.sign(test_feature['test']['features']).cpu()

                gallery_feature = []
                for inputs, labels in self.gallery_data_loader:
                    features = net.forward(inputs.to(self.device))
                    gallery_feature.append(features)
                gallery_feature = torch.cat(gallery_feature, dim=0).cpu()

                dense_acc_list = self.recall(test_feature, self.test_dataset.labels, recall_ids,
                                             gallery_feature, self.gallery_dataset.labels)
                # gallery_feature = torch.sign(gallery_feature)
                # binary_acc_list = recall(test_feature, test_dataset.labels, recall_ids,
                # gallery_feature, gallery_dataset.labels, binary=True)

            else:
                test_feature = []
                for inputs, labels in self.test_data_loader:
                    features = net.forward(inputs.to(self.device))
                    test_feature.append(features)
                test_feature = torch.cat(test_feature, dim=0)
                test_feature = torch.sign(test_feature).cpu()
                dense_acc_list = self.recall(test_feature, self.test_dataset.labels, recall_ids)
                # binary_acc_list = recall(test_feature, test_dataset.labels, recall_ids, binary=True)

        return dense_acc_list

    def uniformity_loss(self, _points):
        # torch.Size([98, 2048])
        return torch.sum(torch.triu(1 / (torch.cdist(_points, _points, p=2) ** 2), diagonal=1))

    def run(self):
        # tq = tqdm(range(self.num_epochs), desc='Loading')
        for epoch in range(self.num_epochs):
            if epoch == 0:
                train_loss = self.train(self.model, self.optimizer_init)
                self.writer.add_scalar("Train/Loss", train_loss, epoch)

            elif 0 < epoch < 30:
                train_loss = self.train(self.model, self.optimizer)
                with torch.no_grad():
                    acc = self.test(self.feature_extractor, self.recalls)  # 0.85
                    self.writer.add_scalar("Train/Loss", train_loss, epoch)
                    for i in range(4):
                        self.writer.add_scalar("Train/Accuracy" + str(self.recalls[i]), acc[i], epoch)
                    self.writer.add_scalar("Train/Uniformity",
                                           self.uniformity_loss(
                                               F.normalize(self.classifier.weight.detach(), dim=1)).item(),
                                           epoch)
                self.lr_scheduler.step()
                # 20에폭 학습 후 유니폼니티 : 2374
                # 20에폭 학습 후 최적화 후 유니폼니티 : 2354
                # 랜덤 초기화 후 수렴 유니폼니티 : 2352
            else:
                if self.label_policy == 'norm':
                    return
                with torch.no_grad():
                    points = F.normalize(self.classifier.weight.detach(), dim=1)
                    mask = torch.ones(self.num_class, dtype=torch.bool).to(self.device)

                    for inner_epoch in range(self.inner_epochs):
                        temp_grad = torch.zeros(self.num_class, self.feature_dim).to(self.device)
                        for i in range(points.shape[0]):
                            mask[i] = False
                            nomi = points[i] - points[mask]
                            denomi = (torch.norm(points[i] - points[mask], p=2, dim=1) ** 4)
                            denomi = denomi.reshape(-1, 1).expand(self.num_class - 1, self.feature_dim)
                            sum_grad = torch.sum(nomi / denomi, dim=0)
                            temp_grad[i] = -2 * sum_grad
                            mask[i] = True

                        if torch.norm(temp_grad) > 5:
                            temp_grad = 5 * temp_grad / torch.norm(temp_grad)
                        points = F.normalize(points - self.lr * temp_grad, dim=1)

                sim_mat = torch.matmul(F.normalize(self.classifier.weight.detach(), dim=1), points.t())

                if self.label_policy == 'hard':
                    hard_label = torch.zeros_like(sim_mat).scatter_(1, torch.argmax(sim_mat, dim=1, keepdim=True), 1)
                    label_mat = hard_label
                elif self.label_policy == 'soft1':
                    soft_label = F.softmax(sim_mat, dim=1)
                    label_mat = soft_label
                elif self.label_policy == 'soft2':
                    _, idx = torch.topk(sim_mat, k=self.soft_k, dim=1)
                    soft2_label = torch.zeros_like(sim_mat).scatter_(dim=1, index=idx, value=1)
                    soft2_label /= self.soft_k
                    label_mat = soft2_label
                elif self.label_policy == 'soft3':
                    _, idx = torch.topk(sim_mat, k=self.soft_k, dim=1)
                    mask = torch.zeros_like(sim_mat).scatter_(dim=1, index=idx, value=1)
                    label_mat = F.softmax(sim_mat * mask * 10, dim=1)
                else:
                    print("problem in policy")
                    return

                label_mat = label_mat.to(self.device)
                self.classifier.weight = nn.Parameter(points).to(self.device)

                train_loss = self.train(self.model, self.optimizer_magnet, label_mat)
                with torch.no_grad():
                    acc = self.test(self.feature_extractor, self.recalls)  # 0.85
                    # tq.set_description("Train Loss: %.04f, accuracy: %.04f, uniformity: %.04f" % (
                    #     train_loss, acc,
                    #     self.uniformity_loss(F.normalize(self.classifier.weight.detach(), dim=1)).item()))
                    self.writer.add_scalar("Train/Loss", train_loss, epoch)
                    for i in range(4):
                        self.writer.add_scalar("Train/Accuracy" + str(self.recalls[i]), acc[i], epoch)
                    self.writer.add_scalar("Train/Uniformity",
                                           self.uniformity_loss(
                                               F.normalize(self.classifier.weight.detach(), dim=1)).item(),
                                           epoch)
                self.lr_scheduler.step()


from multiprocessing import Process


def work(arg_list):
    parser = argparse.ArgumentParser(description='Set hyperparameter')
    parser.add_argument('--data_path', default='/data/hbyoo/metric', type=str)
    parser.add_argument('--data_name', default='car', type=str, choices=['car', 'cub', 'sop', 'isc'])
    parser.add_argument('--crop_type', default='cropped', type=str, choices=['cropped', 'uncropped'])

    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=75, type=int)
    parser.add_argument('--num_sample', default=25, type=int)

    parser.add_argument('--feature_dim', default=512, type=int)
    parser.add_argument('--scale_hyperparm', default=20, type=int)

    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--lr_gamma', default=0.1, type=float)

    parser.add_argument('--recalls', default="1,2,4,8", type=str)

    parser.add_argument('--inner_epochs', default=5000, type=int)
    parser.add_argument('--label_policy', default='hard', type=str, choices=['norm', 'hard', 'soft1', 'soft2', 'soft3'])
    parser.add_argument('--soft_k', default=3, type=int)
    parser.add_argument('--soft_scale', default=10, type=int)

    parser.add_argument('--device', default=0, type=int)

    for _arg_in in arg_list:
        args = parser.parse_args(_arg_in)
        dt = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        writer = SummaryWriter("/data/hbyoo/tensorboard/%s/%s" % (dt, str(_arg_in[1::2])))
        Runner(args, writer).run()
        writer.flush()
        writer.close()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')


    def arg_gen(data_path, data_name, crop_type, num_epochs, batch_size, num_sample, feature_dim, scale_hyperparm,
                lr, lr_gamma, recalls, inner_epochs, label_policy, soft_k=3, soft_scale=10, device=0):
        return ['--data_path', data_path, '--data_name', data_name, '--crop_type', crop_type,
                '--num_epochs', str(num_epochs), '--batch_size', str(batch_size), '--num_sample', str(num_sample),
                '--feature_dim', str(feature_dim), '--scale_hyperparm', str(scale_hyperparm), '--lr', str(lr),
                '--lr_gamma', str(lr_gamma), '--recalls', recalls, '--inner_epochs', str(inner_epochs),
                '--label_policy', label_policy, '--soft_k', str(soft_k), '--soft_scale', str(soft_scale),
                '--device', str(device)]




    device_num = 5
    policy = 'norm'
    feature_dim = 512
    p1 = Process(target=work, args=
    ([arg_gen(data_path='/data/hbyoo/metric', data_name='car', crop_type='cropped', num_epochs=30, batch_size=75,
              num_sample=25, feature_dim=feature_dim, scale_hyperparm=20, lr=0.01, lr_gamma=0.1, recalls="1,2,4,8",
              inner_epochs=5000, label_policy=policy, device=device_num),
      arg_gen(data_path='/data/hbyoo/metric', data_name='cub', crop_type='cropped', num_epochs=30, batch_size=75,
              num_sample=25, feature_dim=feature_dim, scale_hyperparm=20, lr=0.01, lr_gamma=0.1, recalls="1,2,4,8",
              inner_epochs=5000, label_policy=policy, device=device_num),
      arg_gen(data_path='/data/hbyoo/metric', data_name='sop', crop_type='uncropped', num_epochs=30, batch_size=75,
              num_sample=5, feature_dim=feature_dim, scale_hyperparm=20, lr=0.01, lr_gamma=0.1,
              recalls="1,10,100,1000",
              inner_epochs=5000, label_policy=policy, device=device_num),
      arg_gen(data_path='/data/hbyoo/metric', data_name='isc', crop_type='uncropped', num_epochs=30, batch_size=75,
              num_sample=5, feature_dim=feature_dim, scale_hyperparm=20, lr=0.01, lr_gamma=0.1,
              recalls="1,10,20,30",
              inner_epochs=5000, label_policy=policy, device=device_num)],))

    device_num = 1
    policy = 'norm'
    feature_dim = 2048
    p2 = Process(target=work, args=
    ([arg_gen(data_path='/data/hbyoo/metric', data_name='car', crop_type='cropped', num_epochs=30, batch_size=75,
              num_sample=25, feature_dim=feature_dim, scale_hyperparm=20, lr=0.01, lr_gamma=0.1, recalls="1,2,4,8",
              inner_epochs=5000, label_policy=policy, device=device_num),
      arg_gen(data_path='/data/hbyoo/metric', data_name='cub', crop_type='cropped', num_epochs=30, batch_size=75,
              num_sample=25, feature_dim=feature_dim, scale_hyperparm=20, lr=0.01, lr_gamma=0.1, recalls="1,2,4,8",
              inner_epochs=5000, label_policy=policy, device=device_num),
      arg_gen(data_path='/data/hbyoo/metric', data_name='sop', crop_type='uncropped', num_epochs=30, batch_size=75,
              num_sample=5, feature_dim=feature_dim, scale_hyperparm=20, lr=0.01, lr_gamma=0.1,
              recalls="1,10,100,1000",
              inner_epochs=5000, label_policy=policy, device=device_num),
      arg_gen(data_path='/data/hbyoo/metric', data_name='isc', crop_type='uncropped', num_epochs=30, batch_size=75,
              num_sample=5, feature_dim=feature_dim, scale_hyperparm=20, lr=0.01, lr_gamma=0.1,
              recalls="1,10,20,30",
              inner_epochs=5000, label_policy=policy, device=device_num)],))

    p1.start()
    p2.start()
    p1.join()
    p2.join()