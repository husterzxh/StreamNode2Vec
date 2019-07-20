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
    def __init__(self, train_embeddings, test_embeddings, clf):
        self.train_embeddings = train_embeddings
        self.test_embeddings = test_embeddings
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, Y_all):
        self.binarizer.fit(Y_all)
        X_train = []
        Y_train = []
        for node, embeddings in self.train_embeddings.items():
            X_train.append(embeddings)
            Y_train.append(Y_all[int(node)])
        Y = self.binarizer.transform(Y_train)
        self.clf.fit(X_train, Y)

    def evaluate(self, Y):
        list_test_embedding = []
        Y_test = []
        for node, embeddings in self.test_embeddings.items():
            list_test_embedding.append(embeddings)
            Y_test.append(Y[int(node)])

        top_k_list = [len(l) for l in Y_test]
        Y_ = self.predict(top_k_list)
        Y_test = self.binarizer.transform(Y_test)
        averages = ["micro", "macro"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y_test, Y_, average=average)
        return results

    def predict(self, top_k_list):
        list_test_embedding = []
        for node, embeddings in self.test_embeddings.items():
            list_test_embedding.append(embeddings)
        X_ = numpy.asarray(list_test_embedding)
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, Y, seed=0):
        state = numpy.random.get_state()
        numpy.random.seed(seed)

        self.train(Y)
        numpy.random.set_state(state)
        return self.evaluate(Y)


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


def evaluate_embeddings(train_embeddings, test_embeddings):
    X, Y = read_node_label('./data/cora/cora_labels.txt')
    str_result = ''
    clf = Classifier(train_embeddings=train_embeddings, test_embeddings=test_embeddings, clf=LogisticRegression())
    str_result = str_result + str(clf.split_train_evaluate(Y)) + '\n'
    return str_result

if __name__ == '__main__':
    for i in range(1, 10):
        int_all_round = 2
        float_ratio = round(i * 0.1, 1)

        # 读取train结果数据
        file_train_embedding_result = r'./data/' + str(float_ratio) + '/train_node_embedding_' + str(int_all_round) \
                                      + '_' + str(float_ratio) + '.embedding'
        with open(file_train_embedding_result, 'r', encoding='utf-8') as fp1:
            list_train_embedding = fp1.read().split('\n')
        del list_train_embedding[-1]
        dict_train_node_embedding = dict()
        for item_train_embedding in list_train_embedding:
            list_train_node_embedding = item_train_embedding.split(' ')
            str_key = list_train_node_embedding[0]
            list_value = list_train_node_embedding[1:]
            list_value = [float(i) for i in list_value]
            dict_train_node_embedding[str_key] = list_value

        # 读取test结果数据
        file_test_embedding_result = r'./data/' + str(float_ratio) + '/test_node_embedding_' + str(int_all_round) \
                                     + '_' + str(float_ratio) + '.embedding'
        with open(file_test_embedding_result, 'r', encoding='utf-8') as fp1:
            list_test_embedding = fp1.read().split('\n')
        del list_test_embedding[-1]
        dict_test_node_embedding = dict()
        for item_test_embedding in list_test_embedding:
            list_test_node_embedding = item_test_embedding.split(' ')
            str_key = list_test_node_embedding[0]
            list_value = list_test_node_embedding[1:]
            list_value = [float(i) for i in list_value]
            dict_test_node_embedding[str_key] = list_value

        str_result_f1 = str(float_ratio) + ' ' + evaluate_embeddings(dict_train_node_embedding, dict_test_node_embedding)
        file_result = r'./data/classification_f1.txt'
        with open(file_result, 'a', encoding='utf-8') as fp2:
            fp2.write(str_result_f1)
        print(str_result_f1)



