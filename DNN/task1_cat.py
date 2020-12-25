#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/22 16:41
# @Author  : 李刚
# @File    : task1_cat.py
# @Func: 测试猫图像

import matplotlib.pyplot as plt
from DNN.dnn_utils_v2 import *
from DNN import lr_utils, task1 as t

np.random.seed(1)


def two_layer_model(X, Y, layer_dims, learning_rate=0.0075, num_iteration=3000, print_cost=False, isPolt=True):
    """
    实现一个两层的神经网络，【LINEAR->RELU】 -> 【LINEAR->SIGMOID】
    :param X: 输入的数据，维度为（n_x, 例子数）
    :param Y: 标签向量，0为不是猫，1为猫，维度为1
    :param layer_dims:层数的向量 维度为（n_x, n_h, n_y)
    :param learning_rate: 学习速率
    :param num_iteration: 迭代次数
    :param print_cost: 是否打印成本值
    :param isPolt: 是否绘制误差值的图
    :return:
        parameters - 一个包含参数的字典
    """
    np.random.seed(1)
    grads = {}
    costs = []
    (n_x, n_h, n_y) = layer_dims
    """
    初始化参数
    """
    parameters = t.initialize_parameters(n_x, n_h, n_y)
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    # 开始迭代
    for i in range(0, num_iteration):
        # 前向传播
        A1, cache1 = t.linear_activation_forward(X, w1, b1, "relu")
        A2, cache2 = t.linear_activation_forward(A1, w2, b2, "sigmoid")

        # 计算成本
        cost = t.compute_cost(A2, Y)

        # 反向传播
        # 初始化
        dA2 = -(np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        dA1, dw2, db2 = t.linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dw1, db1 = t.linear_activation_backward(dA1, cache1, "relu")
        # 向后传播完成后的数据保存到grads
        grads["dw1"] = dw1
        grads["db1"] = db1
        grads["dw2"] = dw2
        grads["db2"] = db2

        # 更新参数
        parameters = t.update_parameters(parameters, grads, learning_rate)
        w1 = parameters["w1"]
        b1 = parameters["b1"]
        w2 = parameters["w2"]
        b2 = parameters["b2"]

        # 打印成本值
        if i % 100 == 0:
            # 记录成本
            costs.append(cost)
            if print_cost:
                print("第", i, "次迭代，成本值为：", np.squeeze(cost))

    # 迭代完成，绘图
    if isPolt:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    return parameters


def predict(X, Y, parameters):
    """
    该函数预测L层神经网络的结果
    :param X: 测试集
    :param Y: 标签
    :param parameters:
    :return: 给定返回数据集X的预测
    """
    m = X.shape[1]
    n = len(parameters) // 2  # 神经网络的层数
    p = np.zeros((1, m))
    # 根据参数前向传播
    probas, caches = t.L_model_forward(X, parameters)
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    print("准确度为：" + str(float(np.sum((p == Y)) / m)))
    return p


# 加载数据
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()
train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
train_y = train_set_y
test_x = test_x_flatten / 255
test_y = test_set_y
n_x = 12288
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)
par = two_layer_model(train_x, train_y, layers_dims, 0.0075, 2500, True, True)
pred_train = predict
predictions_train = predict(train_x, train_y, par)  # 训练集
predictions_test = predict(test_x, test_y, par)  # 测试集
