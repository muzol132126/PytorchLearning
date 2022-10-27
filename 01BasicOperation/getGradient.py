import torch

x = torch.rand(1)
b = torch.rand(1, requires_grad=True)
w = torch.rand(1, requires_grad=True)

y = w * x
z = y + b

z.backward(retain_graph=True)

print(b.grad)

# print(dir(torch.cuda.is_available))
# print(help(torch.cuda.is_available))

