# -*- coding: utf-8 -*-
# @Time    : 2018/3/20 下午2:15
# @File    : imgPreprocess.py

import time
import string
from PIL import Image, ImageChops
import re
from PIL.GifImagePlugin import getheader, getdata
import os
import pandas as pd
import tqdm

root_dir = '/home/zzc/cool/data/'


# 将gif图像第二帧拆成独立的位图
def split_gif(im, type = 'bmp'):
    print 'spliting', filename,
    im.seek(0)  # skip to the second frame
    type = string.lower(type)
    mode = 'RGB'  # image modea
    if type == 'bmp' or type == 'png':
        mode = 'P'  # image mode
    im.convert(mode)
    print '\n', filename, 'has been splited'
    return im

# 将rgba格式的图像转为rgb
def rgba2rgb(im):
    im_rgb = im.convert('RGB')
    print filename, 'has converted rgb'
    return im_rgb

def p2rgb(im):
    im_rgb = im.convert('RGB')
    return im_rgb

# df = pd.read_csv(root_dir + 'image_label.csv')
# f = df[df['image_label'] == 1]['image_name'].drop_duplicates().tolist()

flog = open('log_validate.txt', 'wr')

# for index in range(1, 4):
#     for root, dirs, files in os.walk(root_dir + 'sohu/Pic_info_train.part' + str(index)):
#         for filename in files:
#             tarname = filename
#
#             try:
#                 im = Image.open(root + '/' + filename)
#             except Exception, e:
#                 flog.write(tarname + '\t' + root + '\t' + e.message + '\t读取失败\n')
#                 continue
#
#             try:
#                 if filename[-3:] == 'GIF':
#                     im = split_gif(im)
#                     tarname = tarname[:-3] + 'JPEG'
#                 if im.mode == 'RGBA':
#                     im = rgba2rgb(im)
#                 elif im.mode == 'P':
#                     im = p2rgb(im)
#             except Exception, e:
#                 flog.write(tarname + '\t' + root + '\t' + e.message + '\t处理异常\n')
#                 continue
#
#             if not os.path.isdir(root_dir + 'sohu/negative'):
#                 os.makedirs(root_dir + 'sohu/negative')
#             if not os.path.isdir(root_dir + 'sohu/positive'):
#                 os.makedirs(root_dir + 'sohu/positive')
#
#             try:
#                 if filename in f: im.save(root_dir + 'sohu/negative/' + tarname)
#                 else: im.save(root_dir + 'sohu/positive/' + tarname)
#             except Exception, e:
#                 flog.write(tarname + '\t' + root + '\t' + e.message + '\t保存失败\n')
#                 continue
#
#             print tarname, 'has saved'
#
#         print root, 'has finished'
#
# flog.close()


for root, dirs, files in os.walk(root_dir + 'sohu/Pic_info_validate'):
    for filename in tqdm.tqdm(files):
        tarname = filename

        try:
            im = Image.open(root + '/' + filename)
        except Exception, e:
            flog.write(tarname + '\t' + root + '\t' + e.message + '\t读取失败\n')
            continue

        try:
            if filename[-3:] == 'GIF':
                im = split_gif(im)
                tarname = tarname[:-3] + 'JPEG'
            if im.mode == 'RGBA':
                im = rgba2rgb(im)
            elif im.mode == 'P':
                im = p2rgb(im)
        except Exception, e:
            flog.write(tarname + '\t' + root + '\t' + e.message + '\t处理异常\n')
            continue

        if not os.path.isdir(root_dir + 'sohu/validate'):
            os.makedirs(root_dir + 'sohu/validate')

        try:
            im.save(root_dir + 'sohu/validate/' + tarname)
        except Exception, e:
            flog.write(tarname + '\t' + root + '\t' + e.message + '\t保存失败\n')
            continue

        print tarname, 'has saved'

    print root, 'has finished'

flog.close()

print 'finished'