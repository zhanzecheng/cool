# -*- coding: utf-8 -*-
"""
# @Time    : 2018/3/20 下午2:21
# @Author  : zhanzecheng
# @File    : make_dictionary_and_id.py
# @Software: PyCharm
"""
import pickle
import os
import sys
import sys, os
import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
# 增加了停用词、词频的筛选
from src.utils.text_utils import create_dico, create_mapping

FILENAME1 = '../../data/News_cut_validate_text.txt'
FILENAME2 = '../../data/News_cut_text.txt'
FILENAME3 = '../../data/News_cut_unlabel_text.txt'
SAVE_DICT = '../../data/word_dict.pkl'
ITEM_TO_ID = '../../data/item_to_id.pkl'
ID_TO_ITEM = '../../data/id_to_item.pkl'
STOPWORD_FILENAME = '../../data/stopword.txt'

STOPWORD = []
with open(STOPWORD_FILENAME, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        STOPWORD.append(line)

print('now begin to create dictionary')
dico = {}
with open(FILENAME1, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in tqdm.tqdm(lines):
        line = line.strip()
        line = line.split('\t')[1:]
        for l in line:
            l = l.strip().split(' ')
            dico = create_dico(l, dico)

with open(FILENAME2, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in tqdm.tqdm(lines):
        line = line.strip()
        line = line.split('\t')[1:]
        for l in line:
            l = l.strip().split(' ')
            dico = create_dico(l, dico)

with open(FILENAME3, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in tqdm.tqdm(lines):
        line = line.strip()
        line = line.split('\t')[1:]
        for l in line:
            l = l.strip().split(' ')
            dico = create_dico(l, dico)

print('before len : ', len(dico))

dic = dico.copy()
for word, count in dico.items():
    if count < 3:
        dic.pop(word)
        continue
    if word in STOPWORD:
        dic.pop(word)

print('after len : ', len(dic))

with open(SAVE_DICT, 'wb') as f:
    pickle.dump(dic, f)


item_to_id, id_to_item = create_mapping(dic)

with open(ITEM_TO_ID, 'wb') as f1, open(ID_TO_ITEM, 'wb') as f2:
    pickle.dump(item_to_id, f1)
    pickle.dump(id_to_item, f2)

