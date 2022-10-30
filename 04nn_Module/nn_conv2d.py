import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

data_set = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(data_set, 64)

class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = Conv2d(3, 6, 3, 1, (0, 0), 1)

    def forward(self, x):
        x = self.conv1(x)
        return x

myModule = MyModule()

writer = SummaryWriter("logs")

step = 0
for data in dataloader:
    imgs, targets = data
    output = myModule(imgs)
    # print(help(torch.shape))
    writer.add_images("input", imgs, step)

    # output = torch.reshape(output, (-1, 3, 30, 30))
    # writer.add_images("output", output, step)
    step+=1

writer.close()


