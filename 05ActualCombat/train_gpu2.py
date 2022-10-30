import torch
import torchvision
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *
import time

# 定义训练设备
device = torch.device("cpu")


train_data = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

train_data_size = len(train_data) # 50000
test_data_size = len(test_data) # 10000

train_dataloader = DataLoader(train_data, 64)
test_dataloader = DataLoader(test_data, 64)


myModule = MyModule()
myModule.to(device)

# 损失函数
loss_fn = CrossEntropyLoss()
loss_fn.to(device)

learning_rate = 1e-2
optimizer = torch.optim.SGD(myModule.parameters(), lr=learning_rate)

# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 1

writer = SummaryWriter("./logs_train")
start_time = time.time()

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))
    myModule.train()

    for imgs, targets in train_dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = myModule(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time-start_time)
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
 
    myModule.eval()
    total_test_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        for imgs, targets in test_dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = myModule(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", loss.item(), total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)

    total_test_step += 1

torch.save(myModule, "./modules/myModule_cpu.pth")
