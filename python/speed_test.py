import torch
import torchvision
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# An instance of your model.
model = torchvision.models.resnet34()
model.to(device)

# An example input you would normally provide to your model's forward() method.
inp = torch.rand(64, 3, 224, 224).to(device)

start = time.time()
for _ in range(1000):
    out = model.forward(inp)
end = time.time()

print("time consuming: ", end - start)

# Pytorch Resnet34 in 1000 loops:
#  1 Bat:  635 Mib,   6.72 sec
# 32 Bat: 2715 Mib,  84.63 sec
# 64 Bat: 4839 Mib, 166.51 sec
# 
