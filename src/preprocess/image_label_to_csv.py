# encoding=utf8

import pandas as pd

image_label_file = '../../data/sohu/News_pic_label_train.txt'
image_name_file = '../../data/News_to_images.txt'

save_file = '../../data/image_label.csv'


def get_image_label():
    news_id = []
    image_name = []
    image_label = []
    with open(image_name_file, 'r') as fname, open(image_label_file, 'r') as flabel:
        name_lines = fname.readlines()
        label_lines = flabel.readlines()
        for name, label in zip(name_lines, label_lines):
            name_list = name.split('\t')
            label_list = label.split('\t')

            if name_list[1] == "NULL\n":
                continue

            name = name_list[1].strip().split(';')
            image_name.extend(name)

            label = label_list[2].strip().split(';')

            news_id.extend([name_list[0] for i in range(len(name))])

            if label_list[1] == '0':
                image_label.extend([0 for i in range(len(name))])
            if label_list[1] == '2':
                image_label.extend([1 for i in range(len(name))])
            if label_list[1] == '1':
                image_label.extend([1 if i in label else 0 for i in name])

    return pd.DataFrame(data={'image_label': image_label, 'news_id': news_id, 'image_name': image_name})


df = get_image_label()

df.to_csv(save_file, index=False)
