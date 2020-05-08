#!/usr/bin/env python
# encoding: utf-8
"""
@author: WangFei
@license: (C) Copyright 2013-2020, Node Supply Chain Manager Corporation Limited.
@contact: wf18684531169@gmail.com
@software: pycharm
@file: function_test.py
@time: 2020/4/29 0:13
@desc
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import LinearRegression as lnr
import LogisticRegression as lr
import Evaluation as ev


data_path = "E:\\github\\LogisticRegression\\data\\"


if __name__ == "__main__":

    # 线性模型，最小二乘法测试*********************************************************************************
    # feature = [[1, 1, 1], [1, 2, 3], [3, 7, 3]]
    # label = [[3], [6], [13]]
    # linear_regression = lnr.Linear_Regression()
    # print(linear_regression.least_squares_method(feature, label))
    # *****************************************************************************************************

    # 逻辑回归模型*********************************************************************************
    data_csv = pd.read_csv(data_path + "test.csv")
    feature_mat = data_csv.iloc[:, 0:4]
    label_mat = data_csv["label"].to_frame()
    test_lr = lr.Logistic_Regression()
    # 梯度下降法
    grad_ascent_model = test_lr.grad_ascent(feature_mat, label_mat, 0.001, 500)
    print("梯度下降法:")
    print(grad_ascent_model)
    print("*******************************")
    # 随机梯度上升法
    origin_model = test_lr.stoc_grad_ascent(feature_mat, label_mat, 0.001)
    print("随机梯度上升法:")
    stoc_grad_ascent_model = []
    for item in origin_model:
        new_list = [item]
        stoc_grad_ascent_model.append(new_list)
    print(stoc_grad_ascent_model)
    print("*******************************")
    print("小样本随机梯度上升法")
    mini_batch_model_origin = test_lr.mini_batch_grad_ascent(feature_mat, label_mat, 150)
    mini_batch_model = []
    for item in mini_batch_model_origin.tolist()[0]:
        new_list = [item]
        mini_batch_model.append(new_list)
    print(mini_batch_model)
    print("*******************************")
    print("预测结果")
    prediction = test_lr.predict(feature_mat, mini_batch_model)
    test_ev = ev.Evaluation()
    confusion_matrix = test_ev.generate_confusion_matrix(np.array(label_mat["label"]).tolist(), prediction)
    print(confusion_matrix)
    sns.heatmap(confusion_matrix, annot=True)
    plt.show()

