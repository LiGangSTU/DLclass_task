#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/25 15:27
# @Author  : 李刚
# @File    : simple_cnn.py
# @Func: 一个简单的CNN分类器
# 参考博客 ；https://blog.csdn.net/jining11/article/details/89114502


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms


class MyCNN(nn.Module):
    def __init__(self, img_size, num_classes):
        super(MyCNN, self).__init__()
        # conv1: Conv2d -> BN -> ReLU -> MaxPool
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # conv2: Conv2d -> BN -> ReLU -> MaxPool
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 全连接层
        self.fc = nn.Linear(32 * (img_size // 4) * (img_size // 4), num_classes)

    def forword(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output


def train(model, train_loader, loss_func, optimizer, device):
    """
    用损失函数和优化器训练模型
    :param model: CNN
    :param train_loader: 训练集
    :param loss_func: 损失函数
    :param optimizer: 优化器
    :param device: GPU device
    :return:
    """
    total_loss = 0
    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        # forward
        outputs = model(images)
        loss = loss_func(outputs, targets)

        # 后向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # 没一百次输出一次损失值
        if (i + 1) % 100 == 0:
            print("Step [{}/{}] Train Loss: {:.4f}"
                  .format(i + 1, len(train_loader), loss.item()))
        return total_loss / len(train_loader)


def evaluate(model, val_loader, device):
    """

    :param model: CNN
    :param val_loader: 校验集
    :param device: 指定GPU
    :return: 返回分类准确率
    """
    model.eval()
    # 禁用梯度计算
    with torch.no_grad():
        correct = 0
        total = 0

        for i, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
        accuracy = correct / total
        print('Accuracy on Test Set: {:.4f} %'.format(100 * accuracy))
        return accuracy


def save_model(model, save_path):
    # 保存模型至save_path路径下
    torch.save(model.state_dice(), save_path)


def show_curve(ys, title):
    """
    绘制图像
    :param ys: 损失 或者 准确率
    :param title:  损失 / 准确率
    :return:
    """
    x = np.array(range(len(ys)))
    y = np.array(ys)
    plt.plot(x, y, c='b')
    plt.title('{} curve'.format(title))
    plt.xlabel('epoch')
    plt.ylabel('{}'.format(title))
    plt.show()


# mean and std of cifar10 in 3 channels
cifar10_mean = (0.49, 0.48, 0.45)
cifar10_std = (0.25, 0.24, 0.26)

# define transform operations of train dataset
train_transform = transforms.Compose([
    # data augmentation
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),

    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)])

# mini train Cifar10 datasets: 1000 images each class
train_dataset = torchvision.datasets.ImageFolder(root='./data/path2cifar10/train', transform=train_transform)
# mini test Cifar10 datasets: 500 images each class
test_dataset = torchvision.datasets.ImageFolder(root='./data/path2cifar10/test', transform=test_transform)

"""
torchvision.datasets provide the full version of CIFAR-10 dataset
if you want to train the full version of cifar10 datasets, use codes below instead.
"""
# train_dataset = torchvision.datasets.CIFAR10(root='./data/',
#                                              train=True,
#                                              transform=train_transform,
#                                              download=True)
# test_dataset = torchvision.datasets.CIFAR10(root='./data/',
#                                             train=False,
#                                             transform=test_transform)

# Data loader: provides single- or multi-process iterators over the dataset.
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False)


def fit(model, num_epochs, optimizer, device):
    """
    train and evaluate an classifier num_epochs times.
    We use optimizer and cross entropy loss to train the model.
    Args:
        model: CNN network
        num_epochs: the number of training epochs
        optimizer: optimize the loss function
    """

    # loss and optimizer
    loss_func = nn.CrossEntropyLoss()

    model.to(device)
    loss_func.to(device)

    # log train loss and test accuracy
    losses = []
    accs = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch + 1, num_epochs))
        # train step
        loss = train(model, train_loader, loss_func, optimizer, device)
        losses.append(loss)

        # evaluate step
        accuracy = evaluate(model, test_loader, device)
        accs.append(accuracy)

    # show curve
    show_curve(losses, "train loss")
    show_curve(accs, "test accuracy")

# hyper parameters
num_epochs = 10
lr = 0.01
image_size = 32
num_classes = 10
# declare and define an objet of MyCNN
mycnn = MyCNN(image_size, num_classes)
print(mycnn)
device = torch.device('cuda:0')

optimizer = torch.optim.Adam(mycnn.parameters(), lr=lr)

# start training on cifar10 dataset
fit(mycnn, num_epochs, optimizer, device)


