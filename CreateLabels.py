#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Author   : 张晓辉
# @Email    : husterzxh@foxmail.com
# @GitHub   : https://github.com/husterzxh
# @Blog     : https://www.cnblogs.com/husterzxh/
# @Time     : 2019/7/20 19:09
# @File     : CreateLabels.py
# @Software : PyCharm
# 生成每个节点的label
import pandas as pd

def create_label():
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
    print(list_data_label)
    # 将label换成数字表示
    dict_label_id = {
        'Case_Based': 0,
        'Genetic_Algorithms': 1,
        'Neural_Networks': 2,
        'Probabilistic_Methods': 3,
        'Reinforcement_Learning': 4,
        'Rule_Learning': 5,
        'Theory': 6
    }
    list_data_label_id = []
    for item_data_label in list_data_label:
        list_data_label_id.append(dict_label_id[item_data_label])

    with open('./data/cora/cora_labels.txt', 'a', encoding='utf-8') as fp:
        for node, label in zip(list_data_index, list_data_label_id):
            write_content = str(node) + ' ' + str(label) + '\n'
            fp.write(write_content)


if __name__ == '__main__':
    create_label()