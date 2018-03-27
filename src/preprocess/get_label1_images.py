import pandas as pd
import os
import tqdm
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

image_dir = "/home/zzc/cool/data/sohu/train/"
image_label_file = "../../data/image_label.csv"

image_label_df = pd.read_csv(image_label_file)

label_1_image_csv = '/home/zzc/cool/data/label_1_image.csv'

news_label1_id = []
with open('/home/zzc/cool/data/sohu/News_pic_label_train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        line = line.split('\t')
        if line[1] == '1':
            news_label1_id.append(line[0])

ids = []
names = []
labels = []
image_label_df = image_label_df
for i in tqdm.tqdm(range(len(image_label_df))):
    tmp = image_label_df.iloc[i]
    if tmp['news_id'] in news_label1_id:
        ids.append(tmp['news_id'])
        name = tmp['image_name']
        if name[-3:] == 'GIF':
            name = name[:-3] + 'JPEG'
        names.append(name)
        labels.append(tmp['image_label'])

image_label_df = pd.DataFrame({'news_id': ids, 'new_name': names, 'label': labels})

image_label_df.to_csv(label_1_image_csv, index=False)

label_1_image_path = '/home/zzc/cool/data/sohu/train_images_label1/'

if not os.path.exists(label_1_image_path):
    os.mkdir(label_1_image_path)
    os.mkdir(label_1_image_path + '/0')
    os.mkdir(label_1_image_path + '/1')

for i in tqdm.tqdm(range(0, len(ids))):
    src_file = ''
    for j in range(0, 2):
        src_file = '/home/zzc/cool/data/sohu/train/' + str(j) + '/' + names[i]
        if os.path.exists(src_file):
            break
    dst_file = label_1_image_path + str(labels[i]) + '/' + names[i]
    shutil.copyfile(src_file, dst_file)
