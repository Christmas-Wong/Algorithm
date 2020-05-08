#!/usr/bin/env python
# encoding: utf-8
"""
@author: WangFei
@license: (C) Copyright 2013-2020, Node Supply Chain Manager Corporation Limited.
@contact: wf18684531169@gmail.com
@software: pycharm
@file: Evaluation.py
@time: 2020/4/30 0:54
@desc
@function: AUC Accuracy Recall F1_score
"""


class Evaluation(object):

    @staticmethod
    def generate_confusion_matrix(real_list, prediction_list):
        confusion_matrix = [[0, 0], [0, 0]]
        if len(real_list) != len(prediction_list):
            return None
        for index in range(len(real_list)):

            if real_list[index] == 1:
                if prediction_list[index] == 1:
                    confusion_matrix[0][0] += 1
                else:
                    confusion_matrix[0][1] += 1

            if real_list[index] == 0:
                if prediction_list[index] == 1:
                    confusion_matrix[1][0] += 1
                else:
                    confusion_matrix[1][1] += 1
        return confusion_matrix
