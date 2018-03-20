# -*- coding: utf-8 -*-
"""
# @Time    : 2018/3/20 上午10:20
# @Author  : zhanzecheng
# @File    : parse_html.py
# @Software: PyCharm
"""


# TODO:这里只是把title单纯的加入进去
# TODO: 还需要大量优化
# TODO: 可能p标签也需要修改

# FILENAME = '../../data/sohu/News_info_train.txt'
# SAVENAME = '../../data/News_to_text.txt'
# FILENAME = '../../data/sohu/News_info_unlabel.txt'
# SAVENAME = '../../data/News_to_unlabel_text.txt'
FILENAME = '../../data/sohu/News_info_validate.txt'
SAVENAME = '../../data/News_to_validate_text.txt'
from html.parser import HTMLParser
import tqdm

class MyHTMLParser(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)
        self.a_text = False
        self.links = []

    def handle_starttag(self, tag, attr):
        if tag == 'span' or tag == 'title':
            self.a_text = True
            # print (dict(attr))

    def handle_endtag(self, tag):
        if tag == 'span' or tag == 'title':
            self.a_text = False

    def handle_data(self, data):
        if self.a_text:
           self.links.append(data)
result = {}
with open(FILENAME, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in tqdm.tqdm(lines):
        line = line.split('\t')
        yk = MyHTMLParser()
        yk.feed(line[1])
        yk.close()
        result[line[0]] = yk.links

with open(SAVENAME, 'w', encoding='utf-8') as f:
    for key in result.keys():
        lines = result[key]
        line = ""
        if len(lines) < 1:
            print(key)
        for count, da in enumerate(lines):
            if count == 0:
                line += da
            else:
                line = line + '\t' + da
        f.write(key + '\t' + line + '\n')