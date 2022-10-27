import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.relu(x)
        return F.relu(x)


myModule = MyModule()
x = torch.tensor([1,-1,-2,3])

output = myModule(x)

print(output)