# Import modules
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Set Hyperparameters
BATCH_SIZE = 256
LR = 0.01
EPOCHS = 5

# Data Preprocessing
transform=transforms.Compose(
          [ transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
          ])

# Data Loader
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Define Model
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
net = torchvision.models.resnet18(num_classes=10)
net = nn.DataParallel(net)
net = net.to(device)

# Define Loss Function
criterion = nn.CrossEntropyLoss()

# Define Optimizer
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

# Training
net.train()
for epoch in range(1, EPOCHS + 1):
    train_loss = correct = total = 0.0

    for idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total += labels.size(0)

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

        if (idx + 1) % 50 == 0 or (idx + 1) == len(train_loader):
            print(" Epoch: [{}/{}] | Batch: [{:3}/{}] | loss: {:.3f} | accuracy: {:6.3f}%".format(
              epoch,
              EPOCHS,
              idx + 1,
              len(train_loader),
              train_loss / (idx + 1),
              100.0 * correct / total,
              ))