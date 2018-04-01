# -*- coding: utf-8 -*-
# @Time    : 2018/3/31 下午3:37
# @File    : image_densenet121_model.py

from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
import numpy as np
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping
import tensorflow as tf
import pandas as pd
import tqdm
import glob
import os

# 指定显卡
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

TRAIN_DATA_PATH = '../../data/sohu/train/'
TEST_DATA_PATH = '../../data/sohu/validate/'
TRAIN_CSV_PATH = '../../data/image_label_3.csv'

def get_csv():

    positive_data_name = glob.glob(TRAIN_DATA_PATH + '0/*.*')
    negative_data_name = glob.glob(TRAIN_DATA_PATH + '1/*.*')
    name_list = positive_data_name + negative_data_name
    label_list = [0] * len(positive_data_name) + [1] * len(negative_data_name)

    print('size of csv: ', len(name_list))

    data = {'name': name_list, 'label': label_list}
    train_data = pd.DataFrame(data=data).sample(frac=1).reset_index(drop=True)
    train_data.to_csv(path_or_buf=TRAIN_CSV_PATH)

    print('done')

def get_model():
    input_image = Input(shape=(224, 224, 3), name='image_input')
    base_model = DenseNet121(input_tensor=input_image, include_top=False, weights='/home/zzc/.keras/models/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')
    flatten = Flatten()(base_model.output)
    dense_image1 = Dense(256, activation='relu', name='dense1')(flatten)
    dropout_image1 = Dropout(0.3, name='dropout1')(dense_image1)
    predictions = Dense(1, activation='sigmoid', name='dense2')(dropout_image1)
    dense_model = Model(inputs=[input_image], outputs=[predictions])
    optimizer = Adam(0.0001, 0.9, 0.999, None)
    dense_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return dense_model

def custom_generator(x_set, y_set=None, batch_size=32, epochs=1):
    print('epochs: ', epochs)
    for epoch in range(epochs):
        print('size of x_set: ', len(x_set))
        for i in range(int((len(x_set) - 1) / batch_size) + 1):

            train_image = []

            for name in tqdm.tqdm(x_set[i * batch_size:(i + 1) * batch_size]):

                try:
                    img = image.load_img(name, target_size=(224, 224))
                except OSError or AttributeError:
                    img = np.zeros(shape=(224, 224, 3))

                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                train_image.extend(x)
            print('y_set: ', y_set)
            if y_set == None:
                yield np.asarray(train_image)
            else:
                yield np.asarray(train_image), np.asarray(y_set[i * batch_size:(i + 1) * batch_size])

# get_csv()


train_data_csv = pd.read_csv(TRAIN_CSV_PATH)[:100]
train_data_size = len(train_data_csv)
batch_size = 32
steps_per_epoch = int((train_data_size - 1) / batch_size) + 1

# k折交叉
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=239)

for train_index, validation_index in kf.split(train_data_csv):

    print('train_index: ', train_index)

    train_data = list(train_data_csv['name'][train_index])
    train_label = list(train_data_csv['label'][train_index])
    validation_data = list(train_data_csv['name'][validation_index])
    validation_label = list(train_data_csv['label'][validation_index])

    model = get_model()

    earlystopping = EarlyStopping(monitor='val_loss', patience=6, verbose=1, mode='min')
    steps_per_epoch = int((len(train_data) - 1) / batch_size) + 1

    model.fit_generator(custom_generator(train_data, y_set=train_label, batch_size=batch_size, epochs=10),
                        steps_per_epoch=steps_per_epoch, epochs=10, callbacks=[earlystopping])

    steps_per_epoch = int((len(validation_data) - 1) / batch_size) + 1
    pred = model.predict_generator(custom_generator(validation_data, batch_size=batch_size), steps=steps_per_epoch)
    print('predict: ', pred)


# print('type: ', type(positive_data_name))
# print(positive_data_name)