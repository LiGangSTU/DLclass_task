#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/25 10:25
# @Author  : 李刚
# @File    : pytorch_cnn.py
# @Func: 用pytorch 构建CNN

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
# 对二维单通道图像的卷积
from PIL import Image


def conv(img, kernel, padding=1, stride=1):
    h, w = img.shape
    kernel_size = kernel.shape[0]

    # 获取扩增后的图像
    ph, pw = h + 2 * padding, w + 2 * padding
    padding_img = np.zeros((ph, pw))
    padding_img[padding: h + padding, padding:w + padding] = img

    # 获取卷积之后的完整图像
    res_h = (h + 2 * padding - kernel_size) // stride + 1
    res_w = (w + 2 * padding - kernel_size) // stride + 1
    res = np.zeros((res_h, res_w))

    # 进行卷积运算
    x, y = 0, 0
    for i in range(0, ph - kernel_size + 1, stride):
        for j in range(0, pw - kernel_size + 1, stride):
            roi = padding_img[i:i + kernel_size, j:j + kernel_size]
            res[x, y] = np.sum(roi * kernel)
            y += 1
        y = 0
        x += 1
    return res


def conv_test():
    img = Image.open('../img/lena.jpg').convert('L')
    plt.imshow(img, cmap='gray')
    # 拉普拉斯
    laplace_kernel = np.array([[-1, -1, -1],
                               [-1, 8, -1],
                               [-1, -1, -1]])

    # 高斯
    gauss_kernel3 = (1 / 16) * np.array([[1, 2, 1],
                                         [2, 4, 2],
                                         [1, 2, 1]])
    # size为5的高斯
    gauss_kernel5 = (1 / 84) * np.array([[1, 2, 3, 2, 1],
                                         [2, 5, 6, 5, 2],
                                         [3, 6, 8, 6, 3],
                                         [2, 5, 6, 5, 2],
                                         [1, 2, 3, 2, 1]])

    fig, ax = plt.subplots(1, 3, figsize=(12, 8))
    laplace_img = conv(np.array(img), laplace_kernel, padding=1, stride=1)
    ax[0].imshow(Image.fromarray(laplace_img), cmap='gray')
    ax[0].set_title('laplace')

    gauss3_img = conv(np.array(img), gauss_kernel3, padding=1, stride=1)
    ax[1].imshow(Image.fromarray(gauss3_img), cmap='gray')
    ax[1].set_title('gauss kernel_size=3')

    gauss5_img = conv(np.array(img), gauss_kernel5, padding=2, stride=1)
    ax[2].imshow(Image.fromarray(gauss5_img), cmap='gray')
    ax[2].set_title('gauss kernel_size=5')

    plt.show()


def myconv2d(features, weights, padding=0, stride=1):
    """
    实现多通道卷积
    :param features: d h w
    :param weights:
    :param padding:
    :param stride:
    :return:
    """
    in_channel, h, w = features.shape
    out_channel, _, kernel_size, _ = weights.shape

    # height and width of output image
    output_h = (h + 2 * padding - kernel_size) // stride + 1
    output_w = (w + 2 * padding - kernel_size) // stride + 1
    output = np.zeros((out_channel, output_h, output_w))

    # call convolution out_channel * in_channel times
    for i in range(out_channel):
        weight = weights[i]
        for j in range(in_channel):
            feature_map = features[j]
            kernel = weight[j]
            output[i] += conv(feature_map, kernel, padding, stride)
    return output


if __name__ == '__main__':
    input_data = [
        [[0, 0, 2, 2, 0, 1],
         [0, 2, 2, 0, 0, 2],
         [1, 1, 0, 2, 0, 0],
         [2, 2, 1, 1, 0, 0],
         [2, 0, 1, 2, 0, 1],
         [2, 0, 2, 1, 0, 1]],

        [[2, 0, 2, 1, 1, 1],
         [0, 1, 0, 0, 2, 2],
         [1, 0, 0, 2, 1, 0],
         [1, 1, 1, 1, 1, 1],
         [1, 0, 1, 1, 1, 2],
         [2, 1, 2, 1, 0, 2]]
    ]
    weights_data = [[
        [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]],

        [[-1, -1, -1],
         [-1, 8, -1],
         [-1, -1, -1]]
    ]]
    input_data = np.array(input_data)
    weights_data = np.array(weights_data)
    print(myconv2d(input_data, weights_data, padding=3, stride=3))

    # pytorch实现CNN
    input_tensor = torch.tensor(input_data).unsqueeze(0).float()
    F.conv2d(input_tensor, weight=torch.tensor(weights_data).float(), bias=None, stride=3, padding=3)
    