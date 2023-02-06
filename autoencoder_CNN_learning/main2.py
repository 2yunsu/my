#!/usr/bin/env python
# coding: utf-8
import os
import torch
from torch import nn, optim
from torchvision import transforms, datasets
from multiprocessing import freeze_support
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import random

import matplotlib.pyplot as plt
import numpy as np
import sys
from datetime import datetime
from tqdm import tqdm


# 하이퍼파라미터
EPOCH = 1
BATCH_SIZE = 64
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# MNIST 데이터셋
trainset = datasets.MNIST(
    root      = './.data',
    train     = True,
    download  = True,
    transform = transforms.ToTensor()
)
testset = datasets.MNIST(root='./.data',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

train_loader = torch.utils.data.DataLoader(
    dataset     = trainset,
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    num_workers = 2,
    drop_last = True
)

test_loader = torch.utils.data.DataLoader(
    dataset     = testset,
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    num_workers = 2,
    drop_last = True
)

#AutoEncoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
        )
        self.decoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),       # 픽셀당 0과 1 사이로 값을 출력합니다
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = Autoencoder().to(DEVICE)
optimizer_AE = torch.optim.Adam(autoencoder.parameters(), lr=0.005)
criterion = nn.MSELoss()

#CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

model     = Net().to(DEVICE)
optimizer_CNN = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            # 배치 오차를 합산
            test_loss += F.cross_entropy(output, target,
                                         reduction='sum').item()

            # 가장 높은 값을 가진 인덱스가 바로 예측값
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def pretrain(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        CNN_Loss_List.append(loss)
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        if batch_idx >= 10:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            break

if __name__=='__main__':
    freeze_support()
    # 랜덤시드 고정
    random_seed = 0
    torch.manual_seed(random_seed)  # torch
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True  # cudnn
    torch.backends.cudnn.benchmark = False  # cudnn
    np.random.seed(random_seed)  # numpy
    random.seed(random_seed)  # random

    Test_Accuracy_List = []
    ask_rate_List = []
    epoch_List = []
    AE_Loss_List = []
    CNN_Loss_List = []
    unq_CNN_Loss_List = []
    threshold_List = []
    crr_ask_rate_List = []

    #threshold
    threshold = 0.02
    cnt_over_thres = 0
    cnt_under_thres = 0
    cnt_cnn = 0
    cnt_unq_cnn = 0
    unq_CNN_loss = torch.tensor(2.5)
    CNN_ratio = 15
    Folder_number = "19"

    # sys.stdout = open('./plot/record{}_batch1.txt'.format(Folder_number), 'a')
    print(datetime.now())
    print("Start Threshold:", threshold)

    # for epoch in range(1, 10):
    #     pretrain(model, train_loader, optimizer_CNN, epoch)
    #     test_loss, test_accuracy = evaluate(model, test_loader)
    #
    #     print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(
    #         epoch, test_loss, test_accuracy))

    for epoch in tqdm(range(1, EPOCH+1)):
        crr_cnt_over_thres = 0
        crr_cnt_under_thres = 0
        step_List = []

        for step, (data, target) in enumerate(train_loader):
            model.eval()
            data, target = data.to(DEVICE), target.to(DEVICE)

            #오토인코더
            autoencoder.train()
            x = data.view(-1, 28 * 28).to(DEVICE)
            y = data.view(-1, 28 * 28).to(DEVICE)
            target = target.to(DEVICE)

            encoded, decoded = autoencoder(x)

            AE_loss = criterion(decoded, y)
            optimizer_AE.zero_grad()
            AE_loss.backward()
            optimizer_AE.step()
            AE_Loss = AE_loss.item()

            #AE_Loss 바탕으로 물어볼지, 스스로 추론할지 결정
            if AE_Loss >= threshold: #물어보기
                # 사람에게 물어볼 수 있도록 이미지 출력
                # _view_data = x.view(-1, 28 * 28)
                # _view_data = _view_data.type(torch.FloatTensor) / 255.
                # test_x = _view_data.to(DEVICE)
                # _, decoded_data = autoencoder(test_x)
                # f, a = plt.subplots(2, 1, figsize=(5, 5))
                # img = np.reshape(_view_data.data.numpy()[0], (28, 28))
                # a[0].imshow(img, cmap='gray')
                # a[0].set_xticks(());
                # a[0].set_yticks(())
                #
                # # 오토인코더가 추상화한 이미지 출력
                # img = np.reshape(decoded_data.to("cpu").data.numpy()[0], (28, 28))
                # a[1].imshow(img, cmap='gray')
                # a[1].set_xticks(());
                # a[1].set_yticks(())
                # plt.show()
                # print(AE_Loss)
                # label = int(input("What is it?:"))#질문하기
                # label = torch.tensor([label])

                #CNN 학습
                optimizer_CNN.zero_grad()
                output = model(data)
                model.train()
                CNN_loss = F.cross_entropy(output, target)#모의구동시에는 target, 실사용시에는 label
                CNN_loss.backward()
                optimizer_CNN.step()

                #물어본 횟수 증가
                cnt_over_thres += 1
                crr_cnt_over_thres += 1

            elif AE_Loss < threshold:#안 물어보고 스스로 추론하기
                # _view_data = x.view(-1, 28 * 28)
                # _view_data = _view_data.type(torch.FloatTensor) / 255.
                # test_x = _view_data.to(DEVICE)
                # _, decoded_data = autoencoder(test_x)
                # f, a = plt.subplots(2, 1, figsize=(5, 5))
                # img = np.reshape(_view_data.data.numpy()[0], (28, 28))
                # a[0].imshow(img, cmap='gray')
                # a[0].set_xticks(());
                # a[0].set_yticks(())
                #
                # # 오토인코더가 추상화한 이미지 출력
                # img = np.reshape(decoded_data.to("cpu").data.numpy()[0], (28, 28))
                # a[1].imshow(img, cmap='gray')
                # a[1].set_xticks(());
                # a[1].set_yticks(())
                # plt.show()

                optimizer_CNN.zero_grad()
                output = model(data)
                pred = output.max(1, keepdim=True)[1]
                pred = pred.reshape(-1)
                model.train()
                unq_CNN_loss = F.cross_entropy(output, pred)
                unq_CNN_loss.backward()
                optimizer_CNN.step()

                # print(AE_Loss)
                # input("it is {}".format(pred))  # 자신이 추론한 거 알려주기

                #스스로 추론한 횟수 증가
                cnt_under_thres += 1
                crr_cnt_under_thres += 1

            if unq_CNN_loss < CNN_loss and threshold > 0.01:
                threshold -= 0.0001
                cnt_unq_cnn += 1


            elif unq_CNN_loss >= CNN_loss:
                threshold += 0.0001 * CNN_ratio
                cnt_cnn += 1

            crr_ask_rate = crr_cnt_over_thres / (crr_cnt_over_thres + crr_cnt_under_thres)

            if step % 3000 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAE_Loss: {:.6f}\tCNN_Loss: {:.6f}\tAsk_rate: {:.6f}\tthreshold: {:.6f}\tunq CNN/CNN: {:.6f}'.format(
                    epoch, step * len(data), len(train_loader.dataset),
                           100. * step / len(train_loader), AE_loss.item(), CNN_loss.item(), crr_ask_rate, threshold,cnt_unq_cnn / cnt_cnn))

            AE_Loss_List.append(AE_Loss)
            CNN_Loss_List.append(CNN_loss.item())
            unq_CNN_Loss_List.append(unq_CNN_loss.item())
            threshold_List.append(threshold)
            crr_ask_rate_List.append(crr_ask_rate)
            step_List.append(step)

        # threshold = AE_Loss

        ask_rate = 100 * cnt_over_thres / (cnt_over_thres + cnt_under_thres)
        #에폭과 Loss_value 출력

        test_loss, test_accuracy = evaluate(model, test_loader)
        Test_Accuracy_List.append(test_accuracy)
        epoch_List.append(epoch)
        ask_rate_List.append(ask_rate)

        print("[Epoch {}]".format(epoch))
        print("cnt_over_thres:", cnt_over_thres)
        print("cnt_under_thres:", cnt_under_thres)
        print("Total Ask_rate: {:.2f}%" .format(ask_rate))
        print('Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss, test_accuracy))
        print('cnt_cnn:', cnt_cnn)
        print('unq_cnt_cnn: ', cnt_unq_cnn)
        print('CNN_ratio_setting: ', CNN_ratio)
        print()

        cnt_over_thres = 0
        cnt_under_thres = 0
        cnt_cnn = 0
        cnt_unq_cnn = 0

    #CNN Loss ratio
    fig1 = plt.subplot(3, 1, 1)
    plt.plot(unq_CNN_Loss_List, label="unq_CNN_Loss", color='red', linestyle="-")
    plt.plot(CNN_Loss_List, label="CNN_Loss", color='blue', linestyle="-")
    plt.title('CNN Loss ratio(per step)')
    plt.legend()

    fig2 = plt.subplot(3, 1, 2)
    plt.plot(AE_Loss_List, label="AE_Loss", color='red', linestyle="-")
    plt.plot(threshold_List, label="Threshold", color='blue', linestyle="-")
    plt.title('threshold & AE_Loss(per step)')
    plt.legend()

    # crr_ask_rate(per step)
    fig3 = plt.subplot(3, 1, 3)
    plt.plot(crr_ask_rate_List, label="ask_rate", color='green', linestyle="-")
    plt.title('crr_ask_rate(per step)')
    plt.legend()

    plt.tight_layout()
    plt.show()
    # plt.savefig('./plot/{}_batch1.png'.format(Folder_number))
    # plt.close()
    #
    # # Ask_rate & test_Accuracy per epoch
    # fig1 = plt.subplot(3, 1, 1)
    # plt.plot(epoch_List, ask_rate_List, label="ask_rate", color='red', linestyle="-")
    # plt.plot(epoch_List, Test_Accuracy_List, label="test_Accuracy", color='blue', linestyle="-")
    # plt.title('Ask_rate & test_Accuracy(per epoch)')
    # plt.xlabel('epoch')
    # plt.ylabel('%')
    # plt.legend()
    #
    # plt.show()
    # plt.tight_layout()
    # plt.savefig('./plot/{}/graph.png'.format(Folder_number))
    # plt.close()


