# -*- coding: utf8 -*-
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd



"""
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=['.', 'label'])
        self.img_dir = "./images"
        self.transform = tr.Compose([tr.Resize(64), tr.ToTensor(), tr.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
"""

"""
class MyDataset(Dataset):
	def __init__(self, x_data, y_data, transform=None):
		# 텐서 변환 안함
		self.x_data = x_data
		self.y_data = y_data
		self.transform = transform
		self.len = len(y_data)

	def __getitem__(self, index):
		# 튜플 형태로 내보내기 전에 전처리 작업 실행
		sample = self.x_data[index], self.y_data[index]
		if self.transform:
			sample = self.trasform(sample)
		return sample  # 넘파이로 출력됨

	def __len__(self):
		return self.len

class ToTensor:
	def __call__(self, sample):
		inputs, labels = sample
		inputs = torch.FloatTensor(inputs)
		inputs = inputs.permute(2, 0, 1)
		return inputs, torch.LongTensor(labels)

		transf = tr.Compose
# 들어온 데이터를 연산
class LinearTensor:
	def __init__(self, slope=1, bias=0):
		self.slope = slope
		self.bias = bias

	def __call__(self, sample):
		inputs, labels = sample
		inputs = self.slope * inputs + self.bias
		return inputs, labels

trans = tr.Compose([ToTensor(),LinearTensor(2,5)])
ds1 = MyDataset(train_images, train_labels, transform=trans)
train_loader1 = DataLoader(ds1, batch_size=10, shuffle=True)
first_data = ds1[0]
features, labels = first_data
print(type(features), type(labels))
"""

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
	trainset = torchvision.datasets.ImageFolder(root="./images_class", transform=transform)
	trainloader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=2)
	testset = torchvision.datasets.ImageFolder(root="./images_class", transform=transform)
	testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

	# 클래스에 대한 인스턴스 생성
	net = Net()
	print(net)

	# optimizer 사용 정의
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	for epoch in range(5):
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
