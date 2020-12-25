#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/25 9:59
# @Author  : 李刚
# @File    : tensorflow_test.py
# @Func: TensorFlow 入门

import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import CNN.tf_utils
import time

tf.compat.v1.disable_eager_execution()
# %matplotlib inline #如果你使用的是jupyter notebook取消注释
np.random.seed(1)

# 实现一个线性功能
def linear_function():
    """
    初始化 w，x，b
    :return:
    """
    np.random.seed(1)
    X = np.random.randn(3, 1)
    w = np.random.randn(4, 3)
    b = np.random.randn(4, 1)
    y = tf.add(tf.matmul(w, X), b)  # matmul是矩阵乘法
    sess = tf.compat.v1.Session()
    res = sess.run(y)
    sess.close()
    return res


if __name__ == '__main__':
    print("result = " + str(linear_function()))
    # y_hat = tf.constant(36, name="y_hat")
    # y = tf.constant(39, name="y")
    # loss = tf.Variable((y - y_hat) ** 2, name="loss")
    # init = tf.compat.v1.global_variables_initializer()
    # with tf.compat.v1.Session() as session:
    #     session.run(init)
    #     print(session.run(loss))
    # x = tf.compat.v1.placeholder(tf.int64, name="x")
    # sess = tf.compat.v1.Session()
    # print(sess.run(2 * x, feed_dict={x: 3}))
    # sess.close()