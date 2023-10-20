'''
@author: Abder-Rahman Ali, PhD
@email: aali25@mgh.harvard.edu
@date: 2023
'''

import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from PIL import Image
import csv

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(250000, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

data_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.ImageFolder('./data/training/', transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=2)

validation_dataset = datasets.ImageFolder('./data/validation/', transform=data_transforms)
validation_loader = DataLoader(validation_dataset, batch_size=10, shuffle=False, num_workers=2)

test_dataset = datasets.ImageFolder('./data/test/', transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = SimpleClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

class UnlabeledImageDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.image_paths = sorted([os.path.join(root, img) for img in os.listdir(root) if self._is_image(img)])

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, os.path.basename(path)

    def __len__(self):
        return len(self.image_paths)

    def _is_image(self, filename):
        return any(filename.endswith(extension) for extension in ['.jpg', '.jpeg', '.png', '.bmp'])

unlabeled_dataset = UnlabeledImageDataset('./data/unlabeled/', transform=data_transforms)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=1, shuffle=False, num_workers=2)

def calculate_accuracy(net, data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_accuracy = calculate_accuracy(net, train_loader)
    validation_accuracy = calculate_accuracy(net, validation_loader)

    print(f'Epoch {epoch + 1}/{num_epochs}')

print('Training completed.')

test_accuracy = calculate_accuracy(net, test_loader)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')


def test_and_save_probabilities_unlabeled(net, unlabeled_loader, csv_filename):
    probability_list = []
    with torch.no_grad():
        net.eval()
        for data, img_paths in unlabeled_loader:
            images = data.to(device)
            outputs = net(images)
            probabilities = nn.Softmax(dim=1)(outputs)
            probability_list.append([img_paths[0]] + probabilities[0].tolist())

    # Save probabilities to a CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image', 'A', 'B'])
        csv_writer.writerows(probability_list)

unlabeled_probabilities_csv = 'unlabeled_probabilities.csv'
test_and_save_probabilities_unlabeled(net, unlabeled_loader, unlabeled_probabilities_csv)