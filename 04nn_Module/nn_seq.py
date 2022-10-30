from cv2 import dilate
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
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


myModule = MyModule()
# input = torch.ones((64, 3, 32, 32))
# output = myModule(input)

# writer = SummaryWriter("logs")
step = 0

for data in dataloader:
    imgs, targets = data
    # writer.add_images("input", imgs, step)
    output = myModule(imgs)
    
    # writer.add_images("output", output, step)
    step += 1
    # writer.add_graph(myModule, input)
# writer.close()