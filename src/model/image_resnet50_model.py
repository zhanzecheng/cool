from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import Dropout
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
import pandas as pd
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

image_dir = "/home/zzc/cool/data/sohu/"
image_input_shape = (224, 224, 3)
image_batch_size = 32
image_nums = 0
image_label_file = "../../data/image_label.csv"

image_label_df = pd.read_csv(image_label_file)


def batch_iter(dir, train, batch_size, epoch, shuffle=True):
    """ Generates a batch iterator"""
    for epoch in range(epoch):
        data_size = len(train['image_name'])
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_image_name = train['image_name'][shuffle_indices]
            shuffle_image_label = train['image_label'][shuffle_indices]
        else:
            shuffle_image_name = train['image_name']
            shuffle_image_label = train['image_label']
        num_batches_per_epoch = int((len(train['image_name']) - 1) / batch_size) + 1
        for batch_num in range(num_batches_per_epoch):
            train_start = batch_num * batch_size
            train_end = min((batch_num + 1) * batch_size, data_size)

            train_image = []

            for name in shuffle_image_name[train_start:train_end]:
                path = ''
                for i in range(1, 4):
                    tmp_path = dir + 'Pic_info_train.part' + str(i) + '/' + name
                    if os.path.exists(tmp_path):
                        path = tmp_path
                        break

                img = image.load_img(path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                train_image.extend(x)

            yield np.array(train_image), shuffle_image_label[train_start:train_end]


def get_model():
    input_image = Input(shape=(224, 224, 3), name='image_input')
    base_model = ResNet50(input_tensor=input_image, weights='imagenet', include_top=False)
    dense_image1 = Dense(256, activation='relu', name='dense1')(base_model.output)
    dropout_image1 = Dropout(0.5, name='dropout1')(dense_image1)
    flatten = Flatten()(dropout_image1)
    predictions = Dense(1, activation='sigmoid', name='dense2')(flatten)
    res_model = Model(inputs=[input_image], outputs=[predictions])
    res_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return res_model


model = get_model()
model.fit_generator(batch_iter(image_dir, image_label_df, 32, 10), steps_per_epoch=11067)
model.save_weights(filepath='weights')






