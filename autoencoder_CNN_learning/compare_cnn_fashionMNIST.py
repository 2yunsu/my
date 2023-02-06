#!/usr/bin/env python
# coding: utf-8

# # CNN으로 패션 아이템 구분하기
# Convolutional Neural Network (CNN) 을 이용하여 패션아이템 구분 성능을 높여보겠습니다.

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import random
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

        if batch_idx >= 516:
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


# ## 데이터셋 불러오기
if __name__ == '__main__':
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

    trainset = datasets.FashionMNIST(
        root='./.data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    testset = datasets.FashionMNIST(root='./.data',
                                    train=False,
                                    transform=transforms.ToTensor(),
                                    download=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        drop_last=True
    )

    # ## 하이퍼파라미터
    model = Net().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    cnt_over_thres = 0
    for epoch in range(1, EPOCHS + 1):
        train(model, train_loader, optimizer, epoch)
        test_loss, test_accuracy = evaluate(model, test_loader)

        print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(
              epoch, test_loss, test_accuracy))