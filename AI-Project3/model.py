import copy

import torch
import torch.nn as nn
from typing import Tuple
from torch import optim
from torch.utils.data import DataLoader
'''
data = torch.load("data.pth")
batch_size = 10

label = data["label"]
feature = data["feature"]
train = []
for index in range(0, len(label)):
    train.append([feature[index], label[index]])
test = []
for index in range(0, len(label)):
    test.append([feature[index], label[index]])
train_loader = DataLoader(train, shuffle=True, batch_size = batch_size)
test_loader = DataLoader(test, shuffle=True, batch_size = batch_size)
'''


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(256, 128)
        # self.norm1 = nn.BatchNorm1d(128)
        self.linear2 = nn.Linear(128, 64)
        # self.norm2 = nn.BatchNorm1d(64)
        self.linear3 = nn.Linear(64, 10)
        self.activate = nn.ReLU()
    def forward(self, x):
        x = x.view(-1, 256)
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.linear3(x)
        return x

'''
model = Model()

def train():
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    optim_name = optimizer.__str__().split('(')[0].strip()
    print("optimizer name:", optim_name)

    for epoch in range(10):
        running_loss = 0
        for batch_idx, data in enumerate(train_loader, 0):
            feature, label = data
            optimizer.zero_grad()
            y_predict = model(feature)
            # print(y_predict)
            loss = criterion(y_predict, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 500 == 499:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 500))
                running_loss = 0.0
        print('Finished Training')
        train_test()
    torch.save(model.state_dict(), "classifier.pth") #save model parameter
    #Test
    


def train_test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            feature, label = data
            outputs = model(feature)
            _, predicted = torch.max(outputs.data, dim=1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    print('Accuracy: %d %%' % (100 * correct / total))
'''




