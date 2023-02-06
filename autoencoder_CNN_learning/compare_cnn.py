#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib as plt
from tqdm import tqdm
from multiprocessing import freeze_support

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

# ## 학습하기
def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        if batch_idx >= 467:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            break


# ## 테스트하기

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

# AECNN_csv = pd.read_csv('./plot/02.csv', names=['seed', 'Ask_rate', 'Accuracy', 'cnt_over_thres'])
# AECNN_csv = AECNN_csv.drop([0], axis = 0)
# cnt_over_thres_csv = AECNN_csv['cnt_over_thres']
# cnt_over_thres = cnt_over_thres_csv.values
# cnt_over_thres_list = cnt_over_thres.tolist()
# cnt_over_thres = 0
# total_Ask_List = []
# total_Accuracy_List = []

if __name__=='__main__':
    freeze_support()

    random_seed = 23
    torch.manual_seed(random_seed)  # torch
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True  # cudnn
    torch.backends.cudnn.benchmark = False  # cudnn
    np.random.seed(random_seed)  # numpy
    random.seed(random_seed)  # random

    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

    EPOCHS = 1
    BATCH_SIZE = 64
    save_file = False

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./.data',
                       train=True,
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./.data',
                       train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)

    # ## 하이퍼파라미터
    model = Net().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(1, EPOCHS + 1):
        train(model, train_loader, optimizer, epoch)
        test_loss, test_accuracy = evaluate(model, test_loader)

        # total_Accuracy_List.append(test_accuracy)
        # total_Ask_List.append(cnt_over_thres_list[23])
        print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(
              epoch, test_loss, test_accuracy))

    # if save_csv:
    #     df = pd.DataFrame(total_Accuracy_List, columns=['Accuracy'])
    #     df['Ask_List'] = total_Ask_List
    #     df.to_csv('./plot/cnn_mnist{}.csv'.format("02"))

     #plot
            # fig1 = plt.subplot(3, 1, 1)
            # plt.plot(unq_CNN_Loss_List, label="unq_CNN_Loss", color='red', linestyle="-")
            # plt.plot(CNN_Loss_List, label="CNN_Loss", color='blue', linestyle="-")
            # plt.title('CNN Loss ratio(per step)')
            # plt.legend()
            #
            #
            # if save_file:
            #     plt.savefig('./plot/rs{}.png'.format(random_seed))
            #
            # elif save_file == False and save_csv==False:
            #     plt.show()
            #
            # plt.close()
