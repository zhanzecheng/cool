# -*- coding: utf-8 -*-
"""
# @Time    : 2018/3/20 下午12:36
# @Author  : zhanzecheng
# @File    : check_text.py
# @Software: PyCharm
"""

CHECK_ID = 'D0020255'
FILENAME = '/media/sohu/News_info_train.txt'

with open(FILENAME, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        # line = line.strip()
        line = line.split('\t')
        if line[0] == CHECK_ID:
            print(line)