# -*- coding: utf-8 -*-
"""
# @Time    : 2018/3/20 下午1:52
# @Author  : zhanzecheng
# @File    : text_utils.py
# @Software: PyCharm
"""
import re

def create_dico(items_list, dict):
    '''
    Create a dictionary of our dataset
    :param items_list:
    :return:
    '''
    for item in items_list:
        if item not in dict:
            dict[item] = 1
        else:
            dict[item] += 1
    return dict

def create_mapping(dico):
    '''
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    :param dico:
    :return:
    '''
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


# def zero_digits(s):
#     """
#     Replace every digit in a string by a zero.
#     """
#     return re.sub('\d', '0', s)

