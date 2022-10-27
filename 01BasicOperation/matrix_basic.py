import torch
import numpy as np

x = torch.empty(5, 3)

# 零矩阵
x = torch.zeros(5, 4, dtype=torch.long)
# 随机矩阵
y = torch.rand(5, 4)
torch.add(x, y )

x = torch.randn(4, 4)
# 变为一行
y = x.view(16)
# -1指的是自动计算第一个维度
z = x.view(-1, 8)

# numpy torch 相互转换
a = torch.ones(5)
b = a.numpy()
print(b)
a = np.ones(5)
b = torch.from_numpy(a)
print(b)
