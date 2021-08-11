import torch
import torchvision
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F


device = torch.device('cpu')

# hyper parameters
num_epochs = 6
batch_size = 4
learning_rate = 0.001

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



class ConvNet(nn.Module): #Input Dimension: 4 x 3 x 32 x 32
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), padding=1) # 4 x 16 x 32 x 32
        self.pool = nn.MaxPool2d(2, 2)                               # 4 x 16 x 16 x 16
        self.conv2 = nn.Conv2d(16, 8, kernel_size=(3, 3), padding=1) # 4 x 8 x 16 x 16 -> 4 x 8 x 8 x 8
        self.fc1 = nn.Linear(8*8*8, 32)            #Fully connected layer 4 x 8*8*8
        self.fc2 = nn.Linear(32, 10)               # Fully connected layer 4 x 32
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 8*8*8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()  # Remember, this function applies the softmax as well
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()

        loss.backward() #Always start the backpropagation from the loss function to compute gradients
        optimizer.step() #Update gradients

        if (i + 1) % 2000 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')


with torch.no_grad(): #prohibit calculations of the gradient since it should have been calculated by now
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        _, predictions = torch.max(output.data, 1)  # returns value, index
        n_samples += labels.size(0)
        n_correct += (predictions == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predictions[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')



torch.save(model.state_dict(), 'birds_vs_airplanes.pt')