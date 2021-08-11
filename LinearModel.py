import torch
import torch.nn as nn

x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float)


test = torch.tensor([5.0], dtype=torch.float)
n_samples, n_features = x.shape
model = nn.Linear(n_features, n_features)

loss = nn.MSELoss()
alpha = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr = alpha) #Probably pass in what is set to requires_grad = True

for i in range(100):
    y_pred = model(test)

    L = loss(y, y_pred)

    L.backward()
    optimizer.step()

    optimizer.zero_grad()

    if(i % 10 == 0):
        [w, b] = model.parameters()
        print(f'i {i + 1}: w = {w[0][0].item():.3f}, loss={L:.8f}')

