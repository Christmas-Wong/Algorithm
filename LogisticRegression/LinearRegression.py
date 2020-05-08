#!/usr/bin/env python
# encoding: utf-8
"""
@author: WangFei
@license: (C) Copyright 2013-2020, Node Supply Chain Manager Corporation Limited.
@contact: wf18684531169@gmail.com
@software: pycharm
@file: LinearRegression.py
@time: 2020/5/8 22:21
@desc
"""
import numpy as np


class Linear_Regression(object):

    @staticmethod
    def least_squares_method(feature_matrix, label_matrix):
        """
        :param feature_matrix: 特征矩阵
        :param label_matrix: 标签矩阵
        :return:
        """
        feature_mat = np.mat(feature_matrix)
        mid_mat = np.linalg.inv(feature_mat.transpose() * feature_mat)
        return mid_mat * feature_mat.transpose() * np.mat(label_matrix)