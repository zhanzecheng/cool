# -*- coding: utf-8 -*-
# @Time    : 2018/3/20 下午2:15
# @File    : imgPreprocess.py

import time
import string
from PIL import Image, ImageChops
from PIL.GifImagePlugin import getheader, getdata
import os

root_dir = '/home/zzc/cool/data/sohu'


## 将gif图像每一帧拆成独立的位图
def gif2images(filename, distDir='.', type='bmp'):
    if not os.path.exists(distDir):
        os.mkdir(distDir)
    print 'spliting', filename,
    im = Image.open(filename)
    im.seek(0)  # skip to the second frame
    cnt = 0
    type = string.lower(type)
    mode = 'RGB'  # image modea
    if type == 'bmp' or type == 'png':
        mode = 'P'  # image mode
    im.convert(mode).save(distDir + '/%d.' % cnt + type)
    print '\n', filename, 'has been splited to directory: [', distDir, ']'
    return cnt



frames = gif2images('P0009015.GIF', distDir='tmp', type='JPEG')
