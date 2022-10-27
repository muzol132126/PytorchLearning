import torchvision
from torch.utils.tensorboard import SummaryWriter


dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10("./dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10("./dataset", train=False, transform=dataset_transform, download=True)

# print(len(test_set))
# print(test_set.classes)

# img, target = test_set[2]
# print(f"{img}, {target}")

writer = SummaryWriter("DT_logs")


writer.close()