#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Author   : 张晓辉
# @Email    : husterzxh@foxmail.com
# @GitHub   : https://github.com/husterzxh
# @Blog     : https://www.cnblogs.com/husterzxh/
# @Time     : 2019/7/20 17:00
# @File     : GenerateRemoveNode.py
# @Software : PyCharm
# 生成随机整数，作为要移除的节点，这些节点将以流的形式到来
import random

def generate_randint():
    # 生成的随机整数总数
    random_size = 2166
    int_start = 0
    int_end = 2707
    list_random = []
    while len(list_random) < random_size:
        int_random = random.randint(int_start, int_end)
        if int_random not in list_random:
            list_random.append(int_random)

    # 对列表进行升序排序
    list_random.sort()
    with open('./data/remove_node_2166.txt', 'w', encoding='utf-8') as fp:
        for item_random in list_random:
            fp.write(str(item_random))
            fp.write('\n')

if __name__ == '__main__':
    generate_randint()