#!/usr/bin/env python
# encoding: utf-8
"""
@author: WangFei
@license: (C) Copyright 2013-2020, Node Supply Chain Manager Corporation Limited.
@contact: wf18684531169@gmail.com
@software: pycharm
@file: DecisionTree.py
@time: 2020/1/31 19:01
@desc
"""
import pandas as pd
import numpy as np
import math

classification_name = "类别"
theta = 0  # 阈值


class TreeNode:
    def __init__(self, fea_name=-1, fea_value=None, out_put=None, real_node=None, false_node=None):
        self.feature = fea_name
        self.value = fea_value
        self.result = out_put
        self.real_node = real_node
        self.false_node = false_node


class DecisionTree(object):

    def __init__(self):
        self._train_data = pd.read_csv("train.csv")
        self._test_data = pd.read_csv("test.csv")

    @staticmethod
    def _empirical_entropy(data):
        if len(data) == 0:
            return 0
        classification_types, classification_counts = \
            np.unique(data[classification_name], return_counts=True)
        probability = []
        classification_total = len(data)
        for item in classification_counts:
            probability.append(item / classification_total)
        empirical_entropy = 0.0
        for index in range(len(probability)):
            empirical_entropy -= probability[index] * math.log2(probability[index])
        return empirical_entropy

    def _information_gain_ratio(self, feature, input_data):
        feature_name, feature_counts = \
            np.unique(input_data[feature], return_counts=True)
        information_gain = self._empirical_entropy(input_data)
        feature_total = len(input_data[feature])
        for index in range(len(feature_name)):
            data = input_data.ix[input_data[feature] == feature_name[index]]
            probability = feature_counts[index] / feature_total
            information_gain -= probability * self._empirical_entropy(data)
        return information_gain

    def _best_feature(self, data):
        empirical_entropy = self._empirical_entropy(data)
        best_feature = ''
        best_gain = 0.0
        for item in data.columns:
            if item == classification_name:
                continue
            if self._information_gain_ratio(item, data) > best_gain:
                best_gain = self._information_gain_ratio(item, data) / empirical_entropy
                best_feature = item
        return best_feature, best_gain

    def _feature_rank(self):
        data = self._train_data
        feature_and_gain = []
        while len(data.columns) > 1:
            feature, gain = self._best_feature(data)
            feature_and_gain.append((feature, gain))
            del data[feature]
        return feature_and_gain

    @staticmethod
    def _divide_set(data, column, value):
        set1 = pd.DataFrame()
        set2 = pd.DataFrame()

        # 定义一个函数，判断当前数据行属于第一组还是第二组
        split_function = None
        if isinstance(value, int) or isinstance(value, float):
            split_function = lambda row: row[column] >= value
        else:
            split_function = lambda row: row[column] == value

        # 将数据集拆分成两个集合，并返回
        for index in range(0, data.shape[0]):
            if split_function(data.iloc[index]):
                set1 = set1.append(data.iloc[index], ignore_index=True)
            else:
                set2 = set2.append(data.iloc[index], ignore_index=True)
        return set1, set2

    @staticmethod
    def _unique_counts(data):
        result = {}
        for index in range(data.shape[0]):
            r = data[classification_name].iloc[index]
            if r not in result:
                result[r] = 0
            result[r] += 1
        return result

    def _build_tree(self, data):

        if len(data) == 0:
            return TreeNode()

        # 第一步：若数据只有一个分类，单节点树
        classification, number = np.unique(data[classification_name], return_counts=True)
        if len(classification) == 1:
            result = {classification[0]: number[0]}
            return TreeNode(out_put=result)
        #
        # 第二步：若数据没有特征，单节点树，最多的分类
        features = list(data.columns)
        features.remove(classification_name)
        number = list(number)
        index = number.index(max(number))
        if len(features) == 0:
            result = {classification[index]: number[index]}
            return TreeNode(out_put=result)

        # 判断信息增益阈值
        best_feature, best_gain = self._best_feature(data)
        if best_gain < theta:
            result = {classification[index]: number[index]}
            return TreeNode(out_put=result)

        # 获取最佳分割条件
        entropy = self._empirical_entropy(data)
        if entropy == 0:
            result = {classification[index]: number[index]}
            return TreeNode(out_put=result)
        feature_item = np.unique(data[best_feature])
        best_entropy = 0
        for item in feature_item:
            set1, set2 = self._divide_set(data, best_feature, item)
            probability1 = float(len(set1) / len(data))
            entropy = entropy - probability1 * self._empirical_entropy(set1)
            entropy = entropy - (1 - probability1) * self._empirical_entropy(set2)
            if entropy > best_entropy and len(set1) > 0 and len(set2) > 0:
                best_entropy = entropy
                best_criteria = (best_feature, item)
                best_sets = (set1, set2)

        # 递归构造树结构
        if best_entropy > 0:
            del best_sets[0][best_feature]
            del best_sets[1][best_feature]
            true_branch = self._build_tree(best_sets[0])
            false_branch = self._build_tree(best_sets[1])
            return \
                TreeNode(fea_name=best_criteria[0], fea_value=best_criteria[1], real_node=true_branch,
                         false_node=false_branch)
        else:
            return TreeNode(out_put=self._unique_counts(data))

    def main(self):
        tree = self._build_tree(self._train_data)
        self._print_tree(tree=tree)

    def _print_tree(self, tree, indent=''):
        # 是否是叶节点
        if tree.result is not None:
            print(str(tree.result))
        else:
            # 打印判断条件
            print(str(tree.feature) + ":" + str(tree.value) + "? ")
            # 打印分支
            print(indent + "T->", )
            self._print_tree(tree.real_node, indent + " ")
            print(indent + "F->", )
            self._print_tree(tree.false_node, indent + " ")


if __name__ == '__main__':
    solution = DecisionTree()
    solution.main()
