#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/25 9:21
# @Author  : 李刚
# @File    : orig_cnn.py
# @Func: 底层搭建一个CNN，拥有卷积层和池化层的网络

import numpy as np
import h5py
import matplotlib.pyplot as plt

np.random.seed(1)  # 指定随机种子

"""
不同模块要定义不同的功能
卷积模块：0边界填充，卷积窗口，前向卷积，反向卷积
池化模块：前向池化，创建掩码，值分配，反向卷积
"""


def zero_pad(X, pad):
    """
    把数据集X的图像边界全部使用0填充pad个宽度和高度
    :param X:
    :param pad:
    :return:
    """
    X_paded = np.pad(X, ((0, 0),  # 样本数，不填充
                         (pad, pad),  # 图像高度,你可以视为上面填充x个，下面填充y个(x,y)
                         (pad, pad),
                         (0, 0)),
                     'constant', constant_values=0)  # 连续一样的值填充
    return X_paded


def zero_pad_test():
    np.random.seed(1)
    x = np.random.randn(4, 3, 3, 2)
    x_paded = zero_pad(x, 2)
    # 查看信息
    print("x.shape =", x.shape)
    print("x_paded.shape =", x_paded.shape)
    print("x[1, 1] =", x[1, 1])
    print("x_paded[1, 1] =", x_paded[1, 1])

    # 绘制图
    fig, axarr = plt.subplots(1, 2)  # 一行两列
    axarr[0].set_title('x')
    axarr[0].imshow(x[0, :, :, 0])
    axarr[1].set_title('x_paded')
    axarr[1].imshow(x_paded[0, :, :, 0])


def conv_single_step(a_slice_prev, w, b):
    """
    在前一层的激活输出的一个片段上应用一个由参数w定义的过滤器
    :param a_slice_prev: 输入数据的一个片段，维度为（过滤器大小，过滤器大小，上一通道数）
    :param w: 权重参数
    :param b: 偏置参数
    :return:
        z - 在输入数据片 X 上卷积滑动窗口（w，b）的结果
    """
    s = np.multiply(a_slice_prev, w)+b
    z = np.sum(s)
    return z


if __name__ == '__main__':
    zero_pad_test()