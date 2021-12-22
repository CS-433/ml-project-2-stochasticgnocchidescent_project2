from torch.nn.modules import transformer
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import torch
from torchvision.transforms.transforms import Normalize

# get MNIST train x and train labels as torch tensors
def load_mnist_train():
    train_dataset = datasets.MNIST(root='./train', train=True, transform=ToTensor(), download = True)
    train_loader = DataLoader(train_dataset, batch_size=1000000)

    for (train_x, train_label) in train_loader:
        train_x, train_label = train_x, train_label
    
    return train_x, train_label

# get MNIST test x and test labels as torch tensors
def load_mnist_test():
    test_dataset = datasets.MNIST(root='./test', train=False, transform=ToTensor(), download = True)
    test_loader = DataLoader(test_dataset, batch_size=1000000)

    for (test_x, test_label) in test_loader:
        test_x, test_label = test_x, test_label
    
    return test_x, test_label

# get MNIST train x and train labels as torch tensors
# reshape to 60000*28*28*1, and normalize the entries in the range [0,255] using l2 norms.
# the purpose of this is for comparison with ProxSARAH, as in their code this is exactly what is happening (partially behind the scenes of TensorFlow)
def load_mnist_train_l2():
    train_dataset = datasets.MNIST(root='./train', train=True, transform=ToTensor(), download = True)
    train_loader = DataLoader(train_dataset, batch_size=1000000)

    for (train_x, train_label) in train_loader:
        train_x, train_label = train_x, train_label
    
    # reshape and "undo" division by 255.0 implicit in ToTensor()
    train_x = (torch.reshape(train_x, (60000, 28, 28, 1)))*255.0

    train_x = torch.nn.functional.normalize(train_x, p=2.0, dim=(1,2))
    
    return train_x, train_label

# get MNIST train x and train labels as torch tensors
# reshape to 10000*28*28*1, and normalize the entries in the range [0,255] using l2 norms.
# the purpose of this is for comparison with ProxSARAH, as in their code this is exactly what is happening (partially behind the scenes of TensorFlow)
def load_mnist_test_l2():
    test_dataset = datasets.MNIST(root='./test', train=False, transform=ToTensor(), download = True)
    test_loader = DataLoader(test_dataset, batch_size=1000000)

    for (test_x, test_label) in test_loader:
        test_x, test_label = test_x, test_label
    
    # reshape and "undo" division by 255.0 implicit in ToTensor()
    test_x = (torch.reshape(test_x, (10000, 28, 28, 1)))*255.0
    
    test_x = torch.nn.functional.normalize(test_x, p=2.0, dim=(1,2))
    return test_x, test_label

# get Fashion MNIST train x and train labels as torch tensors
def load_fashion_mnist_train():
    train_dataset = datasets.FashionMNIST(root='./train', train=True, transform=ToTensor(), download = True)
    train_loader = DataLoader(train_dataset, batch_size=1000000)

    for (train_x, train_label) in train_loader:
        train_x, train_label = train_x, train_label
    
    return train_x, train_label

# get Fashion MNIST test x and test labels as torch tensors
def load_fashion_mnist_test():
    test_dataset = datasets.FashionMNIST(root='./test', train=False, transform=ToTensor(), download = True)
    test_loader = DataLoader(test_dataset, batch_size=1000000)

    for (test_x, test_label) in test_loader:
        test_x, test_label = test_x, test_label
    
    return test_x, test_label

# get CIFAR10 train x and train labels as torch tensors
def load_cifar_train():
    train_dataset = datasets.CIFAR10(root='./train', train=True, transform=ToTensor(), download = True)
    train_loader = DataLoader(train_dataset, batch_size=1000000)

    for (train_x, train_label) in train_loader:
        train_x, train_label = train_x, train_label
    
    return train_x, train_label

# get CIFAR10 test x and test labels as torch tensors
def load_cifar_test():
    test_dataset = datasets.CIFAR10(root='./test', train=False, transform=ToTensor(), download = True)
    test_loader = DataLoader(test_dataset, batch_size=1000000)

    for (test_x, test_label) in test_loader:
        test_x, test_label = test_x, test_label
    
    return test_x, test_label