import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)
        self.sigmoid1 = torch.nn.Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output

myModule = MyModule()

writer = SummaryWriter("logs")
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = myModule(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()