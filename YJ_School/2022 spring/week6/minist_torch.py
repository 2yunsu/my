import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from mnist import load_mnist
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

USE_CUDA=torch.cuda.is_available()
DEVICE=torch.device("cuda" if USE_CUDA else "cpu")

EPOCH=2
BATCH_SIZE=64

transform = transforms.Compose([
    transforms.ToTensor()
])


trainset = datasets.FashionMNIST(
    root      = './.data/',
    train     = True,
    download  = True,
    transform = transform
)
testset = datasets.FashionMNIST(
    root      = './.data/',
    train     = False,
    download  = True,
    transform = transform
)

train_loader = torch.utils.data.DataLoader(
    dataset     = trainset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
)
test_loader = torch.utils.data.DataLoader(
    dataset     = testset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1=nn.Linear(784, 256)
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,10)

    def forward(self,x):
        x=x.view(-1,784)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

model=Net().to(DEVICE)
optimizer = optim.SGD(model.parameters(),lr=0.01)

def train(model, train_loader, optimizer):
    model.train()
    for batch_dix, (data, target) in enumerate(train_loader):
        data, target=data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output=model(data)
        loss=F.cross_entropy(output,target)
        loss.backward()
        optimizer.step()

DNN_test_accuracy=[]

def evaluate(model, test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for data, target in test_loader:
            data, target=data.to(DEVICE), target.to(DEVICE)
            output=model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred=output.max(1,keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss/=len(test_loader.dataset)
    test_accuracy=100. * correct / len(test_loader.dataset)
    DNN_test_accuracy.append(test_accuracy)
    return test_loss, test_accuracy

for epoch in range(1,EPOCH+1):
    train(model,train_loader,optimizer)
    test_loss, test_accuracy = evaluate(model, test_loader)

    print('[{}] Test Loss: {:.4f}, Acccuracy: {:.2f}%.'.format(epoch, test_loss, test_accuracy))

print(DNN_test_accuracy)
plt.plot(DNN_test_accuracy,'r',label='DNN')
plt.legend()
plt.show()