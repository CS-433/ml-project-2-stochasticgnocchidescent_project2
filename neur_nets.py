from torch import nn

import torch
import torchvision

# reference for this:
# https://github.com/ChawDoe/LeNet5-MNIST-PyTorch/blob/master/model.py
class LeNet(nn.Module):
    def __init__(self, cuda):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

        if cuda:
            self.cuda()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y

def ResNet(cuda, single_channel):
    net = torchvision.models.resnet18()
    
    # reference for this (needed for running ResNet on MNIST):
    # https://github.com/marrrcin/pytorch-resnet-mnist/blob/master/pytorch-resnet-mnist.ipynb
    if single_channel:
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    if cuda:
        net.cuda()
    
    return net

# reference:
# https://github.com/epfml/ML_course/blob/master/labs/ex08/solution/ex08.ipynb
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation_fn = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc2(self.activation_fn(self.fc1(x)))
        return out