#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/14 17:39
# @Author  : 李刚
# @File    : task1.py
# @Func: DL课后作业1


"""
- numpy的是主包和Python的科学计算。
- matplotlib是Python中的情节图形库。
-dnn_utils为此笔记本提供了一些必要的功能。
-testCases提供了一些测试用例来评估函数的正确性
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from DNN.testCases_v3 import *
from DNN.dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# np.random.seed（1）用于使所有随机函数调用保持一致.
np.random.seed(1)


# 初始化两层网络和网络的参数
# 模型结构为 线性——RELU——线性——SIGMOID
def initialize_parameters(n_x, n_h, n_y):
    """
    :param n_x:输入层数
    :param n_h:隐藏层数
    :param n_y:输出层数
    :return:
    """
    # 保证取到一致的随机数
    np.random.seed(1)
    # randn函数返回一个或一组样本，具有标准正态分布
    # n_h * n_x 的表格
    w1 = np.random.randn(n_h, n_x) * 0.01
    # 返回来一个给定形状和类型的用0填充的数组
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (w1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (w2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2
    }
    return parameters


def initialize_parameters_deep(layer_dims):
    """
    为L层神经网络实现初始化
    :param layer_dims: list 包含每一层维度
    :return:
    """
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['w' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['w' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters


# 线性——激活
def linear_forward(A, w, b):
    """
    实现前向传播模块
    :param A:  m x n
    :param w: 权重数据 a x m
    :param b: a x 1
    :return:
    """
    # dot()返回的是两个数组的点积(dot product)
    Z = np.dot(w, A) + b
    assert (Z.shape == (w.shape[0], A.shape[1]))
    cache = (A, w, b)

    return Z, cache


def linear_forward_test_case():
    np.random.seed(1)
    A = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    return A, W, b


#  线性激活正向
def linear_activation_forward(A_prev, W, b, activation):
    """
    实现正向传播 linear ——> 激活层
    :param A_prev: 前一层的数据
    :param W: 权重矩阵
    :param b: 偏差向量
    :param activation:
    :return:
    A : 激活函数的输出值
    cache：存储计算结果
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


# L层模型
def L_model_forward(X, parameters):
    caches = []
    L = len(parameters) // 2
    A = X
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['w' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['w' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)
    return AL, caches


# L层模型测试
def L_model_forward_test_case():
    np.random.seed(1)
    X = np.random.randn(4, 2)
    W1 = np.random.randn(3, 4)
    b1 = np.random.randn(3, 1)
    W2 = np.random.randn(1, 3)
    b2 = np.random.randn(1, 1)
    parameters = {"w1": W1,
                  "b1": b1,
                  "w2": W2,
                  "b2": b2}
    return X, parameters


# 计算损失，判断模型是否在学习
def compute_cost(AL, Y):
    """

    :param AL: 观测值
    :param Y: 表示真实值
    :return:
    """
    m = Y.shape[1]
    cost = -np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y)) / m
    # 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    return cost


# 损失函数测试
def compute_cost_test_case():
    Y = np.asarray([[1, 1, 1]])
    aL = np.array([[.8, .9, 0.4]])
    return Y, aL


# 实施反向传播模块
def linear_backward(dZ, cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_backward_test_case():
    np.random.seed(1)
    dZ = np.random.randn(1, 2)
    A = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    linear_cache = (A, W, b)
    return dZ, linear_cache


# 后向传播激活
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def linear_activation_backward_test_case():
    np.random.seed(2)
    dA = np.random.randn(1, 2)
    A = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    Z = np.random.randn(1, 2)
    linear_cache = (A, W, b)
    activation_cache = Z
    linear_activation_cache = (linear_cache, activation_cache)

    return dA, linear_activation_cache


# L层后向传播
def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,
                                                                                                  "sigmoid")
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def L_model_backward_test_case():
    np.random.seed(3)
    AL = np.random.randn(1, 2)
    Y = np.array([[1, 0]])

    A1 = np.random.randn(4, 2)
    W1 = np.random.randn(3, 4)
    b1 = np.random.randn(3, 1)
    Z1 = np.random.randn(3, 2)
    linear_cache_activation_1 = ((A1, W1, b1), Z1)

    A2 = np.random.randn(3, 2)
    W2 = np.random.randn(1, 3)
    b2 = np.random.randn(1, 1)
    Z2 = np.random.randn(1, 2)
    linear_cache_activation_2 = ((A2, W2, b2), Z2)

    caches = (linear_cache_activation_1, linear_cache_activation_2)

    return AL, Y, caches


# 更新参数
def update_parameters(parameters, grads, learning_rate):
    """

    :param parameters: 参数项
    :param grads: 包含后向传播中算出的梯度
    :param learning_rate: 学习速率
    :return:
    """
    L = len(parameters) // 2
    for l in range(L):
        parameters["w" + str(l + 1)] = parameters["w" + str(l + 1)] - learning_rate * grads["dw" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters


def update_parameters_test_case():
    np.random.seed(2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    np.random.seed(3)
    dW1 = np.random.randn(3,4)
    db1 = np.random.randn(3,1)
    dW2 = np.random.randn(1,3)
    db2 = np.random.randn(1,1)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return parameters, grads


if __name__ == '__main__':
    dZ, linear_cache = linear_backward_test_case()

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    print("dA_prev = " + str(dA_prev))
    print("dW = " + str(dW))
    print("db = " + str(db))
