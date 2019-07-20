#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Author   : 张晓辉
# @Email    : husterzxh@foxmail.com
# @GitHub   : https://github.com/husterzxh
# @Blog     : https://www.cnblogs.com/husterzxh/
# @Time     : 2019/7/20 19:07
# @File     : NodeClassification.py
# @Software : PyCharm
# 对embedding的结果进行节点分类
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return numpy.asarray(all_labels)


class Classifier(object):
    def __init__(self, embeddings, clf):
        self.embeddings = embeddings
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)
        averages = ["micro", "macro"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        return results

    def predict(self, X, top_k_list):
        X_ = numpy.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_precent, seed=0):
        state = numpy.random.get_state()

        training_size = int(train_precent * len(X))
        numpy.random.seed(seed)
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        self.train(X_train, Y_train, Y)
        numpy.random.set_state(state)
        return self.evaluate(X_test, Y_test)


def read_node_label(filename, skip_head=False):
    fin = open(filename, 'r')
    X = []
    Y = []
    if skip_head:
        fin.readline()
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y


def evaluate_embeddings(embeddings):
    X, Y = read_node_label('./data/cora/cora_labels.txt')
    str_result = ''
    list_tr_frac = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]
    for item_tr_frac in list_tr_frac:
        clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
        str_result = str_result + str(item_tr_frac) + str(clf.split_train_evaluate(X, Y, item_tr_frac)) + '\n'
    return str_result

if __name__ == '__main__':
    for i in range(1, 10):
        float_ratio = round(i * 0.1, 1)
        file_train_embedding_result = r'./data/' + str(float_ratio) + '/train_node_embedding_' + str(int_all_round) \
                                      + '_' + str(float_ratio) + '.embedding'
        file_test_embedding_result = r'./data/' + str(float_ratio) + '/test_node_embedding_' + str(int_all_round) \
                                     + '_' + str(float_ratio) + '.embedding'



    # 读取embedding结果数据
    file_name_embedding = r'./data/result_node_embedding_StreamNode_22166.embedding'
    with open(file_name_embedding, 'r', encoding='utf-8') as fp1:
        list_embedding = fp1.read().split('\n')
    del list_embedding[-1]
    dict_node_embedding = dict()
    for item_embedding in list_embedding:
        list_node_embedding = item_embedding.split(' ')
        str_key = list_node_embedding[0]
        list_value = list_node_embedding[1:]
        list_value = [float(i) for i in list_value]
        dict_node_embedding[str_key] = list_value
    str_result_f1 = evaluate_embeddings(dict_node_embedding)
    file_result = r'./data/classification_f1.txt'
    with open(file_result, 'w', encoding='utf-8') as fp2:
        fp2.write(str_result_f1)
    print(str_result_f1)

