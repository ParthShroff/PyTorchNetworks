import numpy as np
import torch
torch.set_printoptions(edgeitems=2, threshold=50)
import os
import imageio

batch_size = 3
batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)

#In order to read a set of images in a folder, consider the following example:

data_dir = '../image_cats'
filenames = [name for name in os.listdir(data_dir)
             if os.path.splitext(name)[-1] == '.png']

#filenames = [cat1.png, cat2.png, cat3.png]

#enumerate adds a counter to the filenames to form [[0, file0], [1, file1], etc.]


for i, filename in enumerate(filenames):
    img_arr = imageio.imread(os.path.join(data_dir, filename))
    img_final = torch.from_numpy(img_arr)
    img_final = img_final.permute(2, 0, 1)
    img_final = img_final[:3]
    batch[i] = img_final #The shape of batch is the dimension prepended so therefore the shape is 3 x 3 x 256 x 256

#print(batch)
#print(batch.shape)
batch = batch.float()
batch /= 255.0
n_channels = batch.shape[1]
for i in range(n_channels):
    mean = torch.mean(batch[:, i])
    print(mean)
    std = torch.std(batch[:, i])
    print(std)
    batch[:, i] = (batch[:, i] - mean)/std
    print(batch)

x = torch.tensor([1, 2, 3, 4])
x_mod = torch.unsqueeze(x, 1)
print(x_mod)

a = torch.randn(3, requires_grad=True)
print(a)

b = a + 2
print(b)

c = a * b
print(c)

d = c.mean()
print(d)

d.backward()

print(a.grad)

#Simple backpropagation:
x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)

y_pred = w * x
loss = (y_pred - y) ** 2

loss.backward()
print(w.grad)
