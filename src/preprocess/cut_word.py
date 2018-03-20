# -*- coding: utf-8 -*-
"""
# @Time    : 2018/3/20 上午11:26
# @Author  : zhanzecheng
# @File    : cut_word.py
# @Software: PyCharm
"""

import jieba
# FILENAME = '../../data/News_to_text.txt'
# SAVENAME = '../../data/News_cut_text.txt'
FILENAME = '../../data/News_to_validate_text.txt'
SAVENAME = '../../data/News_cut_validate_text.txt'
import tqdm
with open(FILENAME, 'r', encoding='utf-8') as f, open(SAVENAME, 'w', encoding='utf-8') as d:
    lines = f.readlines()
    for line in tqdm.tqdm(lines):
        line = line.strip()
        line = line.split('\t')
        result = line[0]
        for l in line[1:]:
            result += '\t'
            seg_list = jieba.cut(l, cut_all=False)
            for count, seg in enumerate(seg_list):
                if count == 0:
                    result += seg
                else:
                    result = result + ' ' + seg
        d.write(result + '\n')






