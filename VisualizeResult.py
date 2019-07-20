#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Author   : 张晓辉
# @Email    : husterzxh@foxmail.com
# @GitHub   : https://github.com/husterzxh
# @Blog     : https://www.cnblogs.com/husterzxh/
# @Time     : 2019/7/20 19:05
# @File     : VisualizeResult.py
# @Software : PyCharm
# 用t-SNE将embedding的结果可视化
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_result():
    # 读取content数据，将每个节点从0开始重新编号,cora数据集共有2708个节点，5429条边
    file_name_content = r'./data/cora/cora.content'
    df_raw_data_content = pd.read_csv(file_name_content, sep='\t', header=None)
    # 论文的新编号
    list_data_index = list(df_raw_data_content.index)
    # 论文的旧编号
    list_data_id = list(df_raw_data_content.iloc[:, 0])
    dict_id_index = dict(zip(list_data_id, list_data_index))
    # 论文的label
    list_data_label = list(df_raw_data_content.iloc[:, -1])
    # print(list_data_label)
    # 将label换成数字表示
    dict_label_id = {
        'Case_Based': '#0000FF',
        'Genetic_Algorithms': '#00FF00',
        'Neural_Networks': '#FFFF00',
        'Probabilistic_Methods': '#FF0000',
        'Reinforcement_Learning': '#808080',
        'Rule_Learning': '#FFC0CB',
        'Theory': '#008000'
    }
    list_data_label_id = []
    for item_data_label in list_data_label:
        list_data_label_id.append(dict_label_id[item_data_label])

    # 读取embedding结果数据
    file_name_embedding = r'./data/result_node_embedding_StreamNode_22166.embedding'
    with open(file_name_embedding, 'r', encoding='utf-8') as fp1:
        list_embedding = fp1.read().split('\n')
    del list_embedding[-1]
    dict_node_embedding = dict()
    for item_embedding in list_embedding:
        list_node_embedding = item_embedding.split(' ')
        int_key = int(list_node_embedding[0])
        list_value = list_node_embedding[1:]
        list_value = [float(i) for i in list_value]
        dict_node_embedding[int_key] = list_value

    list_data = []
    # 按顺序读取论文的旧编号
    for item_data_id in list_data_id:
        list_data.append(dict_node_embedding[dict_id_index[item_data_id]])
    print('数据规模：{}*{}'.format(len(list_data), len(list_data[0])))

    # 用t-SNE进行可视化
    area = np.pi * 2 ** 2  # 点面积
    X_tsne = TSNE(learning_rate=100).fit_transform(list_data)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=area, c=list_data_label_id)
    plt.title('t-SNE StreamNode2Vec in cora')
    plt.show()
    # plt.savefig(r'./embedding.png')


if __name__ == '__main__':
    visualize_result()