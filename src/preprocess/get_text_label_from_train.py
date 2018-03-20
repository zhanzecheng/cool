# -*- coding: utf-8 -*-
"""
# @Time    : 2018/3/20 下午12:10
# @Author  : zhanzecheng
# @File    : get_text_label_from_train.py
# @Software: PyCharm
"""
import tqdm
FILENAME = '../../data/sohu/News_pic_label_train.txt'
TEXTNAME = '../../data/News_to_text.txt'
SAVENAME = '../../data/Text_label.txt'

label = {}
with open(FILENAME, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in tqdm.tqdm(lines):
        line = line.strip()
        line = line.split('\t')
        id = line[0]
        la = line[1]
        pattern = line[3]
        label[id] = [la, pattern]

print('label get done')

with open(TEXTNAME, 'r', encoding='utf-8') as f, open(SAVENAME, 'w', encoding='utf-8') as d:
    lines = f.readlines()
    for line in tqdm.tqdm(lines):
        result = ""
        line = line.strip()
        line = line.split('\t')
        if line[0] not in label:
            print('label not contain out text id')
            quit()
        la, pattern = label[line[0]][0], label[line[0]][1]
        if int(la) == 0:
            length = len(line[1:])
            for i in range(length):
                if i == 0:
                    result += '0'
                else:
                    result = result + ' 0'
            d.write(line[0] + '\t' + result + '\n')
        elif int(la) == 2:
            length = len(line[1:])
            for i in range(length):
                if i == 0:
                    result += '1'
                else:
                    result = result + ' 1'
            d.write(line[0] + '\t' + result + '\n')
        else:
            length = len(line[1:])
            result += '0'
            if length > 1:
                for t in line[2:]:
                    if t.strip() in pattern:
                        result += ' 1'
                    else:
                        result += ' 0'
            d.write(line[0] + '\t' + result + '\n')



