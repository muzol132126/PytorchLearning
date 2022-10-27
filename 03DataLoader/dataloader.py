from asyncore import write
import torchvision
from torch.utils.tensorboard import SummaryWriter


from torch.utils.data import DataLoader
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data,batch_size=64, shuffle=True,num_workers=0, drop_last=False)

writer = SummaryWriter("dataloader")
for epoch in range(3):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images(f"Epoch: {epoch}", imgs, step)
        step+=1

writer.close()
