#!/usr/bin/env python
# encoding: utf-8
"""
@author: WangFei
@license: (C) Copyright 2013-2020, Node Supply Chain Manager Corporation Limited.
@contact: wf18684531169@gmail.com
@software: pycharm
@file: LogisticRegression.py
@time: 2020/4/28 23:51
@desc
"""
import numpy as np


def sigmoid(input_data):
    """
    :param input_data: theta的转置和X的乘积
    :return:
    """
    return np.longfloat(1.0 / (1 + np.exp(-input_data)))


class Logistic_Regression(object):

    @staticmethod
    def grad_ascent(data_mat, label_mat, alpha, cycle_time):
        """
        :param data_mat: 特征矩阵
        :param label_mat: 标签矩阵
        :param alpha: 迭代步长
        :param cycle_time:迭代次数
        :return: 权重矩阵
        """
        data_matrix = np.mat(data_mat)
        label_matrix = np.mat(label_mat)
        m, n = np.shape(data_matrix)
        weights = np.ones((n, 1))
        for index in range(cycle_time):
            label_model = sigmoid(data_matrix * weights)
            error = label_matrix - label_model
            weights += alpha * data_matrix.transpose() * error
        return weights

    @staticmethod
    def stoc_grad_ascent(data_mat, label_mat, alpha):
        """
        :param data_mat: 特征矩阵
        :param label_mat: 标签矩阵
        :param alpha: 迭代步长
        :return:
        """
        data_matrix = np.array(data_mat)
        label_matrix = np.array(label_mat)
        m, n = np.shape(data_matrix)
        weights = np.ones(n)
        for i in range(m):
            h = sigmoid(sum(data_matrix[i] * weights))
            error = label_matrix[i] - h
            weights = weights + alpha * error * data_matrix[i]
        return weights

    @staticmethod
    def mini_batch_grad_ascent(data_mat, label_mat, cycle_time):
        """
        :param data_mat: 特征矩阵
        :param label_mat: 标签矩阵
        :param cycle_time: 循环次数
        :return:
        """
        # 将列表转化为array格式
        data_matrix = np.array(data_mat)
        label_matrix = np.mat(label_mat)
        # 获取dataMatrix的行、列数
        m, n = np.shape(data_matrix)
        # 初始化回归系数和步长
        weights = np.mat(np.ones(n).tolist())
        for j in range(cycle_time):
            data_index = list(range(m))
            for i in range(m):
                # 逐渐减小alpha，每次减小量为1.0/(j+i)
                alpha = 4 / (1.0 + j + i) + 0.01
                # 随机选取样本
                rand_index = int(np.random.uniform(0, len(data_index)))
                # 随机选取的一个样本，计算
                h = sigmoid(np.dot(data_matrix[rand_index], weights.tolist()[0]))
                # 计算误差
                error = label_matrix[rand_index] - h
                # 更新回归系数
                weights = weights + alpha * error * data_matrix[rand_index]
                # 删除已经使用过的样本
                del (data_index[rand_index])
        return weights

    @staticmethod
    def predict(data_mat, model_mat):
        prediction = []
        data_matrix = np.mat(data_mat)
        model_matrix = np.mat(model_mat)
        for item in data_matrix:
            if sigmoid(item * model_matrix) >= 0.5:
                prediction.append(1)
            else:
                prediction.append(0)
        return prediction
