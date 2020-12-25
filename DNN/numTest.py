#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/24 9:22
# @Author  : 李刚
# @File    : numTest.py
# @Func: 判断数字

import tensorflow as tf
import numpy as np

# 即显模式
tf.enable_eager_execution()

# 导入数据
train, test = tf.keras.datasets.mnist.load_data()
(train_x, train_y), (test_x, test_y) = train, test
# 形状和类型
train_x = train_x.reshape([-1, 784]).astype("float32")
test_x = test_x.reshape([-1, 784]).astype("float32")
train_y = train_y.astype("int32")
test_y = test_y.astype("int32")

# 训练，测试样本
print('测试集 %s, %s' %(train_x.shape, train_y.shape))
