#!/usr/bin/env python
# encoding: utf-8
"""
@author: WangFei
@license: (C) Copyright 2013-2020, Node Supply Chain Manager Corporation Limited.
@contact: wf18684531169@gmail.com
@software: pycharm
@file: BayesClassifier.py
@time: 2020/1/29 15:52
@desc
"""
import pandas as pd
import numpy as np

classification_name = 'y'


class BayesClassifier(object):

    def __init__(self):
        self._train_data = pd.read_csv("train.csv")
        self._test_data = pd.read_csv("test.csv")

    def _priori_probability(self):
        """
        :return: Each classification's priori probability
        :return_type: map
        :return_example: ("element_1", 0.500, 10)
        """
        classifications, counts = np.unique(self._train_data[classification_name],
                                            return_counts=True)
        total_number = 0
        for item in counts:
            total_number += item
        classify_map = map(lambda index: (classifications[index],
                                          counts[index] / total_number,
                                          counts[index]),
                           range(len(classifications)))
        return classify_map

    def _conditional_probability(self, classify_name, classify_counts):
        """
        :param classify_map: ("element_1", 0.500, 10)
        :return: conditional probability for each feature
        :return_example:[0.2, 0.2, ..., 0.2]
        """
        features = self._train_data.columns
        combination = []
        combination_counts = []
        combination_probability = []
        for item in range(len(self._train_data)):
            classification = self._train_data.at[item, classification_name]
            denominator = classify_counts[classify_name.index(classification)]
            for element in features:
                if element == classification_name:
                    continue
                comb_ele = {'feature': element,
                            'value': self._train_data.at[item, element],
                            'classification': classification}
                if comb_ele in combination:
                    index = combination.index(comb_ele)
                    combination_counts[index] += 1
                    combination_probability[index] = combination_counts[index] / denominator
                    continue
                combination.append(comb_ele)
                combination_counts.append(1)
                combination_probability.append(1 / denominator)
        return combination, combination_probability

    def _classification_probability(self, combination, combination_probability,
                                    classify_name, classify_probability):
        classification_probability = []
        for index in range(len(self._test_data)):
            classification_probability.append(classify_probability*np.ones(len(classify_name)))
            for item in self._train_data.columns:
                if item == classification_name:
                    continue
                for element in combination:
                    if element["feature"] == item \
                            and element["value"] == self._test_data.at[index, item]:
                        print(element)
                        combination_index = combination.index(element)
                        print(combination_probability[combination_index])
                        print('***********************************************************')
                        classify_index = classify_name.index(element['classification'])
                        classification_probability[index][classify_index] *= combination_probability[combination_index]
        return classification_probability

    def main(self):
        classify_map = self._priori_probability()
        classify_name = []
        classify_probability = []
        classify_counts = []
        for x in classify_map:
            classify_name.append(x[0])
            classify_probability.append(x[1])
            classify_counts.append(x[-1])
        combination, combination_probability \
            = self._conditional_probability(classify_name, classify_counts)
        print(self._classification_probability(combination, combination_probability, classify_name, classify_probability))
        print(classify_name)


if __name__ == '__main__':
    solution = BayesClassifier()
    solution.main()
