import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device('cpu')

# hyper parameters
input_size = 784  # Flatten 28*28 tensor into 1d image

hidden_size = 500

num_class = 10  # digits from 1 - 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

#print(type(train_dataset))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = examples.next()
#print(samples.shape, labels.shape)  # samples shape = 100 (batch size) X 1 (Color Channel) X 28 X 28 (Image Array)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes): #use this to create layers and activation functions
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)  # First layer from input to hidden, number of incoming and outgoing edges
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes) #Don't need to apply the softmax here

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


model = NeuralNet(input_size, hidden_size, num_class)

# loss & optimizer
criterion = nn.CrossEntropyLoss()  # Remember, this function applies the softmax as well
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#print(model.parameters())
for p in model.parameters():
    if p.grad is not None:
        print("Not none data")
        print(p.grad.data)

# training loop
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  # iterator, (samples, labels)
        #print(type(images))
        images = images.reshape(-1, 28 * 28).to(device) #If we don't declare image as a variable, then autograd may not calculate the backward pass? Confirm with Jacob
        labels = labels.to(device)   #Ask if this is the same thing as images = Variable(images)

        # forward pass
        outputs = model(images)

        loss = criterion(outputs, labels)  # Labels are the actual output classes

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

# test
with torch.no_grad(): #prohibit calculations of the gradient since it should have been calculated by now
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        output = model(images)
        _, predictions = torch.max(output.data, 1)  # returns value, index
        n_samples += labels.size(0)
        n_correct += (predictions == labels).sum().item()

    accuracy = 100.0 * n_correct / n_samples

    print(f'accuracy = {accuracy}')
