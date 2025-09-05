import torch

x = torch.rand(3, 4, 5)
print(x)
y = x[0,-1,:]
print(y)

vocab_size = y.size(0)
mask = torch.full((vocab_size,), float("-inf"), dtype=y.dtype)
mask[[4]] = 0  # Set allowed tokens to 0 (no penalty)
print(y + mask)