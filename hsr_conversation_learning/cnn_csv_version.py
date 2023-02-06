# -*- coding: utf8 -*-
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
import cv2



class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=['.', 'label'])
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv2.imread(img_path) # image == color_image
        image = Image.fromarray(image)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label


# nn.Module을 상속 받는다.

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(2704, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	# 연산 순서 정의
	def forward(self, x):
		# 64 x 64
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 13 * 13)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)

		return x

if __name__ == '__main__':
	#load_data
	transform = tr.Compose([tr.Resize([64,64]), tr.ToTensor(), tr.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
	customset = CustomImageDataset(annotations_file='labels.csv',
								  img_dir='images',
								  transform=transform)
	trainset, testset = train_test_split(customset, test_size=0.2)
	trainloader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=2)
	testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

	# 클래스에 대한 인스턴스 생성
	net = Net()
	print(net)

	# optimizer 사용 정의
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	for epoch in tqdm(range(5)):
		running_loss = 0.0
		# traindata 불러오기(배치 형태로 들어옴)
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data
			# optimizer 초기화
			optimizer.zero_grad()
			# net에 input 이미지 넣어서 output 나오기
			outputs = net(inputs)
			# output로 loss값 계산
			loss = criterion(outputs, labels)
			# loss를 기준으로 미분자동계산
			loss.backward()
			# optimizer 계산
			optimizer.step()
			# loss값 누적
			running_loss += loss.item()
			if i % 2000 == 1999:
				print('[%d, %5d] loss: %.3f' %
					  (epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0

	print('Finished Training')

	# 학습한 모델 저장
	PATH = "hsr_net.pth"
	torch.save(net.state_dict(), PATH)

	# 저장한 모델 불러오기
	net = Net()
	net.load_state_dict(torch.load(PATH))

	# 테스트 데이터로 예측하기
	correct = 0
	total = 0
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			outputs = net(images)
			_, predicted = torch.max(outputs.data, 1)  # argmax랑 비슷
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print("Accuracy of the network on th 10000 test images : %d %%" % (100 * correct / total))