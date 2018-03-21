# -*- coding: utf-8 -*-
# @Time    : 2018/3/21 下午1:51
# @File    : get_image_label_from_csv.py

import pandas as pd

def get_image_label(src_name, negative_list, error_list):
    tar_name = []
    label = []
    old_name = []
    positive_list = list(set(src_name) ^ set(negative_list))

    print 'negative begin'

    for image_name in negative_list:
        if image_name in error_list: continue
        old_name.append(image_name)
        if image_name[-3:] == 'GIF':
            tar_name.append(image_name[:-3] + 'JPEG')
        else:
            tar_name.append(image_name)
        label.append(1)

    print 'negative finished'

    for image_name in positive_list:
        if image_name in error_list: continue
        old_name.append(image_name)
        if image_name[-3:] == 'GIF':
            tar_name.append(image_name[:-3] + 'JPEG')
        else:
            tar_name.append(image_name)
        label.append(0)

    df_label = pd.DataFrame(data={'old_name': old_name, 'new_name': tar_name, 'label': label})
    print 'data size: ', len(df_label)
    df_label.to_csv('/home/zzc/cool/data/image_label_2.csv', index = False)

df = pd.read_csv('/home/zzc/cool/data/image_label.csv')
src_name = df['image_name'].drop_duplicates().tolist()
negative_list = df[df['image_label'] == 1]['image_name'].drop_duplicates().tolist()
error_list = ['P0085747.jpg']

get_image_label(src_name, negative_list, error_list)