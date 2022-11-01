from filecmp import cmp
from pickletools import optimize
from sklearn.model_selection import learning_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# 定义训练设备
device = torch.device("cuda:0")

data_transform = torchvision.transforms.Compose(
                    [torchvision.transforms.ToTensor(), 
                    torchvision.transforms.Normalize([0.5], [0.5])])

train_data = torchvision.datasets.MNIST("./dataset/Mnist", train=True, transform=data_transform, download=True)
test_data = torchvision.datasets.MNIST("./dataset/Mnist", train=False, transform=data_transform, download=True)
# plt.imshow(train_data.data[0].numpy(), cmap='gray')
# plt.show()
# print(train_data.data.size())

train_data_size = len(train_data)
test_data_size = len(test_data)

train_dataloader = DataLoader(train_data, 64)
test_dataloader = DataLoader(test_data, 64)


class Mnist_nn(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x


mnist_model = Mnist_nn()
mnist_model.to(device)

writer = SummaryWriter('./Mnist_logs')

loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)
learning_rate = 0.02
optimizer = torch.optim.SGD(mnist_model.parameters(), lr=learning_rate)

epoch = 20

total_train_step = 0
total_test_loss = 0.0
total_accuracy = 0.0
total_test_step = 0

for i in range(epoch):
    mnist_model.train()
    for imgs, targets in train_dataloader:
        imgs = imgs.view(imgs.size(0),-1)
        imgs = imgs.to(device) # torch.Size([64, 1, 28, 28])
        targets = targets.to(device)
        predict = mnist_model(imgs)

        loss = loss_fn(predict, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1

        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)


    total_test_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        mnist_model.eval()
        for imgs, targets in test_dataloader:
            imgs = imgs.to(device)
            imgs = imgs.view(imgs.size(0),-1)
            targets = targets.to(device)
            outputs = mnist_model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", loss.item(), total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)

    total_test_step+=1

writer.close()