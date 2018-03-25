from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.vgg16 import VGG16
import tqdm
from keras.preprocessing import image
import pandas as pd
import random
from keras.models import Model
import numpy as np
import os
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
images_path = '/home/zzc/cool/data/sohu/train/'
image_label_csv = '/home/zzc/cool/data/image_label.csv'
image_label_clean_csv = '/home/zzc/cool/data/image_label_2.csv'
image_nums = 20
batch_size = 96
save_file = '../../data/images_features_validate.pkl'


def imagenet_scale(x):
    """
    ImageNet is trained with the following mean pixels.
    """
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    return x


def get_images_features(images_list, model):
    """
    get features of images for every news.
    :param images_list: the list of images
    :param model: the model to get the features
    :param images_nums: 每条新闻选取的图片数目
    :return:
    """
    images_features = []
    batch_nums = int(len(images_list) / batch_size)

    for i in tqdm.tqdm(range(batch_nums)):
        images = np.zeros(shape=(image_nums * batch_size, 224, 224, 3))
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        name_list = []
        for nl in images_list[batch_start:batch_end]:
            name_list.extend(nl)
        for count, name in enumerate(name_list):
            path = ''
            for i in range(0, 2):
                path = images_path + str(i) + '/' + name
                if os.path.exists(path):
                    break
            try:
                img = image.load_img(path, target_size=(224, 224))
            except OSError or AttributeError:
                img = np.zeros(shape=(224, 224, 3))

            x = image.img_to_array(img)
            x = imagenet_scale(x)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            images[count] = x
        features50 = model.predict(images, verbose=0)
        for j in range(batch_size):
            images_features.append(features50[j * image_nums:(j + 1) * image_nums])
    return images_features


if __name__ == '__main__':
    # ids = []
    #
    # with open('../../data/News_cut_text.txt', 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         line = line.strip()
    #         line = line.split('\t')[0]
    #         ids.append(line)
    #
    # images_name = pd.read_csv(image_label_csv)
    # images_name_clean = pd.read_csv(image_label_clean_csv)
    #
    # # news_train = images_name[['news_id']].drop_duplicates()
    #
    # images_name = images_name[images_name['image_name'] != 'P0085747.jpg']
    #
    # old_new_name = {}
    #
    # for old_name, new_name in zip(images_name_clean['old_name'], images_name_clean['new_name']):
    #     old_new_name[old_name] = new_name
    #
    # images_list = []
    # for id_ in tqdm.tqdm(ids):
    #     old_images_list = images_name[images_name['news_id'] == id_]['image_name']
    #     new_images_list = []
    #     for name in old_images_list:
    #         new_images_list.extend(old_new_name[name])
    #     if len(new_images_list) >= image_nums:
    #         # 随机选10个
    #         new_images_list = random.sample(new_images_list, image_nums)
    #     else:
    #         new_images_list.extend(['0'] * (image_nums - len(new_images_list)))
    #     images_list.append(new_images_list)
    # # resnet = ResNet50(include_top=False, weights='imagenet')
    #
    # base_model = VGG16(include_top=False, weights='imagenet', pooling='max')
    # # model = Model(inputs=base_model.inputs, outputs=base_model.get_layer())
    #
    # images_features_list = get_images_features(images_list, base_model)

    images_name = []
    with open('/home/zzc/cool/data/sohu/News_info_validate.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split('\t')
            if line[2] == 'NULL':
                images_name.append([])
            else:
                images_name.append(line[2].split(';'))
    images_list = []
    for name_list in images_name:
        if len(name_list) >= image_nums:
            name_list = random.sample(name_list, image_nums)
        else:
            name_list.extend(['0'] * (image_nums - len(name_list)))
        images_list.append(name_list)
    base_model = VGG16(include_top=False, weights='imagenet')

    images_features_list = get_images_features(images_list, base_model)

    with open(save_file, 'wb') as f:
        pickle.dump(images_features_list, f)
    print('done')
