import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "./dataset/dog.png"
image = Image.open(image_path)
image = image.convert('RGB') # png有四个通道
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
image = image.cuda()

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

model = torch.load("./modules/myModule.pth", map_location=torch.device('cuda'))
model = model.cuda()
# print(model)

image = torch.reshape(image, (1,3,32,32))
with torch.no_grad():
    output = model(image)

print(output)

print(output.argmax(1))
