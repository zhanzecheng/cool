# -*- coding: utf-8 -*-
"""
# @Time    : 2018/3/20 下午8:34
# @Author  : zhanzecheng
# @File    : get_images_from_serve.py
# @Software: PyCharm
"""

'''
我们使用这个脚本从服务器上把我们要的照片放在一个文件夹里面
'''
import os
import shutil
FILENAME = '../../data/check_image'
if os.path.exists(FILENAME):
    shutil.rmtree(FILENAME)
else:
    os.mkdir(FILENAME)

IMAGE_NAME = []
for i in range(1, 31):
    if i < 10 :
        image_name = 'P000000' + str(i)
    else:
        image_name = 'P00000' + str(i)
    IMAGE_NAME.append(image_name)

for image_name in IMAGE_NAME:
    findFlag = False
    try:
        os.system('cp /home/zzc/cool/data/sohu/Pic_info_train.part1/' + image_name + '* ' + FILENAME)
        findFlag = True
    except:
        print('We cant not find image in dataSet 1')

    if not findFlag:
        try:
            os.system('cp /home/zzc/cool/data/sohu/Pic_info_train.part2/' + image_name + ' ' + FILENAME)
            findFlag = True
        except:
            print('We cant not find image in dataSet 2')

    if not findFlag:
        try:
            os.system('cp /home/zzc/cool/data/sohu/Pic_info_train.part3/' + image_name + ' ' + FILENAME)
            findFlag = True
        except:
            print('We cant not find image in dataSet 3')


