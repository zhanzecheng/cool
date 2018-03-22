# encoding=utf8

import pandas as pd

image_name_file = '../../data/News_to_images_validate.txt'

save_file = '../../data/validate_image.csv'


def get_image_label():
    news_id = []
    image_name = []
    with open(image_name_file, 'r') as fname:
        name_lines = fname.readlines()
        for name in name_lines:
            name_list = name.split('\t')

            if name_list[1] == "NULL\n":
                continue

            name = name_list[1].strip().split(';')
            image_name.extend(name)

            news_id.extend([name_list[0] for i in range(len(name))])

    return pd.DataFrame(data={'news_id': news_id, 'image_name': image_name})


df = get_image_label()

df.to_csv(save_file, index=False)
