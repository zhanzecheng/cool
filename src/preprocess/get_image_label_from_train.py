# -*- coding: utf-8 -*-
"""
# @Time    : 2018/3/20 上午9:49
# @Author  : zhanzecheng
# @File    : get_image_label_from_train.py
# @Software: PyCharm
"""
'''
从原始文件中解析到label对于的images标签
'''
FILENAME = '/media/sohu/News_info_train.txt'
SAVENAME = '../../data/News_to_images.txt'
result = []
with open(FILENAME, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        # line = line.strip()
        line = line.split('\t')
        result.append(line[0] + '\t' + line[-1])

with open(SAVENAME, 'w', encoding='utf-8') as f:
    for line in result:
        f.write(line )