#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Author   : 张晓辉
# @Email    : husterzxh@foxmail.com
# @GitHub   : https://github.com/husterzxh
# @Blog     : https://www.cnblogs.com/husterzxh/
# @Time     : 2019/7/20 17:26
# @File     : StreamNode2Vec.py
# @Software : PyCharm
# 1. 根据影响力传播算法找到所有受影响的节点
# 2. 得到到来的节点的embedding
# 3. 更新之前节点的embedding
import math
import random
import time

import pandas as pd
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec

# 最终每个节点的embedding结果，key是新编号
dict_node_embedding = dict()
# 训练节点（初始节点）的embedding
dict_train_node_embedding = dict()
# 测试节点（动态节点）的embedding
dict_test_node_embedding = dict()

# 读取数据
def read_data():
    # 读取content数据，将每个节点从0开始重新编号,cora数据集共有2708个节点，5429条边
    file_name_content = r'./data/cora/cora.content'
    df_raw_data_content = pd.read_csv(file_name_content, sep='\t', header=None)
    # 论文的新编号
    list_data_index = list(df_raw_data_content.index)
    # 论文的旧编号
    list_data_id = list(df_raw_data_content.iloc[:, 0])
    # 论文的个数
    data_size = df_raw_data_content.shape[0]
    dict_id_index = dict(zip(list_data_id, list_data_index))

    # 读取cites数据，创建邻接矩阵
    file_name_cites = r'./data/cora/cora.cites'
    df_raw_data_cites = pd.read_csv(file_name_cites, sep='\t', header=None)
    # 创建一个规模和邻接矩阵一样大小的矩阵
    matrix_adjacency = np.zeros((data_size, data_size))
    # 创建邻接矩阵
    for i in range(len(df_raw_data_cites)):
        index_x = dict_id_index[df_raw_data_cites.iloc[i, 0]]
        index_y = dict_id_index[df_raw_data_cites.iloc[i, 1]]
        matrix_adjacency[index_x][index_y] = matrix_adjacency[index_y][index_x] = 1

    # 为了方便后续操作，给邻接矩阵最后加一行，标记每个节点的id
    # 不加在邻接矩阵的开始是为了避免改变节点id与矩阵索引的对应关系
    list_node_ids = range(data_size)
    matrix_adjacency = np.insert(matrix_adjacency, data_size, values=list_node_ids, axis=0)
    print('邻接矩阵规模：{}'.format(matrix_adjacency.shape))
    return matrix_adjacency


# 动态embedding
def streaming_network_embedding(matrix_adjacency, float_ratio, int_all_round=2, int_embedding_dimension=90, int_walk_length=10, int_window_size=7):
    # 完整的图（所有节点和边都在）
    matrix_adjacency_delete_index = matrix_adjacency[:-1, :]
    # 为每个节点设定一个激活阈值
    nparray_sentiment = np.zeros((matrix_adjacency_delete_index.shape[0]))
    for n in range(matrix_adjacency_delete_index.shape[0]):
        nparray_sentiment[n] = random.random()  # 定义社交网络中节点对某一新闻的情感指数，随机独立

    # 在初始邻接矩阵中移除即将到来的节点，得到t0时刻的邻接矩阵
    # 对t0时刻的邻接矩阵进行node2vec，得到t0时刻各个节点的embedding
    # 读取事先生成的移除的节点
    file_remove_node = './data/' + str(float_ratio) + '/remove_node_' + str(float_rate) + '.txt'
    with open(file_remove_node, 'r', encoding='utf-8') as fp:
        read_content = fp.read()
        list_remove_nodes = read_content.split('\n')
    # 最后一个元素为空，所以需要删除
    del list_remove_nodes[-1]
    # 直接读进来list元素是str类型，转成int
    list_remove_nodes = [int(i) for i in list_remove_nodes]
    # t0时刻的初始邻接矩阵：不包括接下来即将到来的节点
    matrix_adjacency_t0 = np.delete(matrix_adjacency, list_remove_nodes, axis=0)
    matrix_adjacency_t0 = np.delete(matrix_adjacency_t0, list_remove_nodes, axis=1)
    print('t0时刻邻接矩阵规模：{}'.format(matrix_adjacency_t0.shape))

    # 通过t0时刻的邻接矩阵构建t0时刻的network(networkx版)
    # 由于t0时刻的邻接矩阵的最后一行是节点id，所以需要删除
    matrix_adjacency_t0_delete_index = matrix_adjacency_t0[:-1, :]
    nx_graph_t0 = nx.Graph(matrix_adjacency_t0_delete_index)
    # 通过邻接矩阵构建t0时刻的network(node2vec版)
    n2v_graph_t0 = node2vec.Graph(nx_G=nx_graph_t0, is_directed=False, p=1, q=1)
    n2v_graph_t0.preprocess_transition_probs()
    # 原文：dimension = 90; walk_length = 10; window_size = 7
    walks = n2v_graph_t0.simulate_walks(num_walks=10, walk_length=int_walk_length)
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=int_embedding_dimension, window=int_window_size, min_count=0, sg=1, workers=8, iter=1)
    # 读取embedding的结果，将各个节点的embedding存入dict_node_embedding中
    for word, vocab_ in sorted(model.wv.vocab.items(), key=lambda item: -item[1].count):
        row = model.wv.vectors[vocab_.index]
        # word是str类型，直接从matrix_adjacency_t0取元素是float64类型
        node_index = int(matrix_adjacency_t0[-1, int(word)])
        dict_node_embedding[node_index] = ' '.join(repr(val) for val in row)


    # 接下来，之前删除的节点依次到来(t1 t2 t3...时刻)，进行动态embedding
    # 打乱节点到来的顺序
    random.shuffle(list_remove_nodes)
    list_remove_nodes_cp = list_remove_nodes[:]
    for item_node in list_remove_nodes_cp:
        # 节点到来后，重新构建邻接矩阵
        list_remove_nodes.remove(item_node)
        # print('还剩下{}个节点'.format(len(list_remove_nodes)))
        matrix_adjacency_t1 = np.delete(matrix_adjacency, list_remove_nodes, axis=0)
        matrix_adjacency_t1 = np.delete(matrix_adjacency_t1, list_remove_nodes, axis=1)
        matrix_adjacency_t1_delete_index = matrix_adjacency_t1[:-1, :]
        nx_graph_t1 = nx.Graph(matrix_adjacency_t1_delete_index)

        # 标记当前图中节点是否被激活
        nparray_state = np.zeros((matrix_adjacency_t1_delete_index.shape[0]), dtype=np.int64)
        # 本次到来的节点被标记为激活
        nparray_state[matrix_adjacency_t1[-1, :].tolist().index(item_node)] = 1
        # 每次被激活的节点
        list_last_actived_node = [matrix_adjacency_t1[-1, :].tolist().index(item_node)]
        list_next_actived_node = []
        list_all_actived_node = []

        # 影响力传播
        for int_round in range(int_all_round):
            # 遍历被激活的节点，在当前图上找到被激活的节点的邻居，在邻居上用独立级联模型
            for item_actived_node in list_last_actived_node:
                dict_iter_node = nx.all_neighbors(nx_graph_t1, item_actived_node)
                for item_iter_node in dict_iter_node:
                    # 判断该节点是否已经被激活，
                    # 如果被激活，则跳过；否则判断是否要被激活
                    if nparray_state[item_iter_node] == 0:
                        if random.random() < nparray_sentiment[int(matrix_adjacency_t1[-1, item_iter_node])]:
                            nparray_state[item_iter_node] = 1
                            list_next_actived_node.append(item_iter_node)
                            list_all_actived_node.append(item_iter_node)
                    else:
                        continue
            list_last_actived_node = list_next_actived_node[:]
            list_next_actived_node.clear()
        # 可能对于当前节点来说，并没有对其他节点产生影响
        # 随机选取一个节点的embedding结果作为当前节点的embedding，并且不更新其他节点的embedding
        # 标记是否更新其他节点
        bool_flag_update = True
        while len(list_all_actived_node) == 0:
            bool_flag_update = False
            int_start = 0
            int_end = matrix_adjacency_t1_delete_index.shape[0] - 1
            int_random = random.randint(int_start, int_end)
            int_node = int(matrix_adjacency_t1[-1, int_random])
            if int_node in dict_node_embedding.keys():
                list_all_actived_node.append(int_random)
                break

        # 根据论文中的公式得到到来节点的embedding
        list_arrived_node_embedding = []
        for i in range(int_embedding_dimension):
            float_all_embedding = 0
            for item_all_actived_node in list_all_actived_node:
                str_embedding = dict_node_embedding[int(matrix_adjacency_t1[-1, item_all_actived_node])]
                list_embedding = str_embedding.split(' ')
                float_embedding = float(list_embedding[i])
                float_all_embedding += float_embedding
            float_all_embedding = float_all_embedding / len(list_all_actived_node)
            list_arrived_node_embedding.append(float_all_embedding)
        dict_node_embedding[item_node] = ' '.join(repr(val) for val in list_arrived_node_embedding)

        # 根据论文中的公式得到受影响节点的embedding
        if bool_flag_update:
            float_para = 1 - math.sqrt(1 - 1/len(list_all_actived_node))
            list_para_arrived_node = [float_para * i for i in list_arrived_node_embedding]
            for item_all_actived_node in list_all_actived_node:
                str_old_embedding = dict_node_embedding[int(matrix_adjacency_t1[-1, item_all_actived_node])]
                list_old_embedding = str_old_embedding.split(' ')
                list_old_embedding = [float(i) for i in list_old_embedding]
                list_new_embedding = list(map(lambda x: x[0]-x[1], zip(list_old_embedding, list_para_arrived_node)))
                dict_node_embedding[int(matrix_adjacency_t1[-1, item_all_actived_node])] = ' '.join(repr(val) for val in list_new_embedding)

    # 将最终的embedding结果写入文件
    # file_embedding_result = r'./data/result_node_embedding_StreamNode_' + str(int_all_round) \
    #                         + str(len(list_remove_nodes_cp)) + '.embedding'
    # for key, word in dict_node_embedding.items():
    #     with open(file_embedding_result, 'a', encoding='utf-8') as fp:
    #         write_content = str(key) + ' ' + str(word) + '\n'
    #         fp.write(write_content)


if __name__ == '__main__':
    matrix_adjacency = read_data()
    for i in range(1, 10):
        float_ratio = round(i * 0.1, 1)
        start_time = time.clock()
        streaming_network_embedding(matrix_adjacency, float_ratio)
        end_time = time.clock()
        print('运行时间：{}'.format(end_time-start_time))