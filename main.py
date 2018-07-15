# Adapted from
# https://medium.com/ml2vec/intro-to-pytorch-with-image-classification-on-a-fashion-clothes-dataset-e589682df0c5
import argparse
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch import IntTensor

use_cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
device = torch.device("cuda" if use_cuda else "cpu")
labels_map = {0: 'T-Shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt',
              7: 'Sneaker', 8: 'Bag', 9: 'Ankle Boot'}

parser = argparse.ArgumentParser()
parser.parse_args()
parser.add_argument("-batch_size", help="Batch size used", default=50)
parser.add_argument("-num-epochs", help="Number of epochs used", default=1)
parser.add_argument("-learning_rate", help="CNN learning rate", default=0.001)
args = parser.parse_args()
print(args)

def plot_kernels(tensor, num_cols=6):
    num_kernels = tensor.shape[0]
    num_rows = 1 + num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols, num_rows))
    for i in range(num_kernels):
        ax1 = fig.add_subplot(num_rows, num_cols, i + 1)
        ax1.imshow(tensor[i][0, :, :], cmap='gray')
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    plt.show()

def plot_losses(losses, results):
    fig = plt.figure()
    plt.xlabel('Batches #')
    plt.ylabel('Loss')
    plt.plot(losses, label="Total loss")
    for i in range(len(results)):
        print(results[i])
        plt.plot(results[i], label=labels_map[i])

    #Add plot of losses per category
    plt.show()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(3 * 3 * 64, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

training_set = datasets.FashionMNIST('data', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ]))
train_loader = torch.utils.data.DataLoader(
    training_set, batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('data', train=False, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ])), batch_size=args.batch_size, shuffle=True, **kwargs)

#Show some sample images
fig = plt.figure()

N = len(training_set)
#Show some sample images
fig = plt.figure()

for i in range(0,16):
        num = random.randint(0, N)
        sample = training_set[num]
        ax = plt.subplot(4, 4, i + 1)
        ax.set_title('Sample #{}'.format(labels_map[int(sample[1])]))
        ax.axis('off')
        plt.imshow(sample[0].view(sample[0].size(1), sample[0].size(2)))
plt.show()

cnn = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=args.learning_rate)
N = int(train_loader.__len__())
losses = []

for epoch in range(args.num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.float())
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.data.item())

        if (i + 1) % 100 == 0:
            print('Epoch : {epoch_no}/{epoch_total}, Batch : {batch_no}/{N},  Loss: {loss:.{digits}f}'.format(
                digits=2,
                epoch_no=epoch + 1,
                epoch_total=args.num_epochs,
                batch_no=i + 1,
                N=N,
                loss=loss.data.item())
            )
            break

cnn.eval()
correct = 0
total = 0
results = np.zeros(len(labels_map))

for images, labels in test_loader:
    images = Variable(images.float())
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    for i in labels_map:
        results[i] += (predicted == i).sum()
results = list(map(lambda x: x/total, results))

for i in labels_map:
    print('{item} Test Accuracy: {score:.4f}'.format(item=labels_map[i], score=100 * results[i]))
print('Overall Test Accuracy: {0:.4f}'.format(100 * correct / total))

filters = cnn.modules()
model_layers = [i for i in cnn.children()]

plot_losses(losses, results)

first_layer = model_layers[0]
first_kernels = first_layer[0].weight.data.numpy()
plot_kernels(first_kernels, 8)

second_layer = model_layers[1]
second_kernels = second_layer[0].weight.data.numpy()
plot_kernels(second_kernels, 8)

third_layer = model_layers[2]
third_kernels = third_layer[0].weight.data.numpy()
plot_kernels(third_kernels, 8)

# Line chart of losses based on certain classifications
