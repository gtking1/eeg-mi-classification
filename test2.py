import torch

x = torch.randn(2, 3, 5)
print(x)
print(x.size())
print(torch.permute(x, (2, 0, 1)).size())
print(torch.permute(x, (2, 0, 1)))