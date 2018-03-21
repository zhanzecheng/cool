from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping
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
image_label_file = "../../data/image_label_2.csv"

image_label_df = pd.read_csv(image_label_file)

# image_label_df = image_label_df.head(1000)

image_name_column_name = 'new_name'
image_label_column_name = 'label'


def imagenet_scale(x):
    """ImageNet is trained with the following mean pixels.
    """
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    return x


def batch_iter(dir, train, batch_size, epochs, shuffle=True):
    """ Generates a batch iterator"""
    for epochs in range(epochs):
        data_size = len(train[image_name_column_name])
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_image_name = train[image_name_column_name][shuffle_indices]
            shuffle_image_label = train[image_label_column_name][shuffle_indices]
        else:
            shuffle_image_name = train[image_name_column_name]
            shuffle_image_label = train[image_label_column_name]
        num_batches_per_epoch = int((len(train[image_name_column_name]) - 1) / batch_size) + 1
        for batch_num in range(num_batches_per_epoch):
            train_start = batch_num * batch_size
            train_end = min((batch_num + 1) * batch_size, data_size)

            train_image = []

            for name in shuffle_image_name[train_start:train_end]:
                path = ''

                for i in ['positive', 'negative']:
                    tmp_path = dir + i + '/' + name
                    if os.path.exists(tmp_path):
                        path = tmp_path
                        break
                # for i in range(1, 4):
                #     tmp_path = dir + 'Pic_info_train.part' + str(i) + '/' + name
                #     if os.path.exists(tmp_path):
                #         path = tmp_path
                #         break

                try:
                    img = image.load_img(path, target_size=(224, 224))
                except OSError:
                    img = np.zeros(shape=(224, 224, 3))

                x = image.img_to_array(img)
                x = imagenet_scale(x)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                train_image.extend(x)

            yield (np.array(train_image), shuffle_image_label[train_start:train_end])


def get_model():
    input_image = Input(shape=(224, 224, 3), name='image_input')
    base_model = VGG16(input_tensor=input_image, weights='imagenet', include_top=False)
    # base_model = ResNet50(input_tensor=input_image, weights='imagenet', include_top=False)
    dense_image1 = Dense(256, activation='relu', name='dense1')(base_model.output)
    dropout_image1 = Dropout(0.3, name='dropout1')(dense_image1)
    flatten = Flatten()(dropout_image1)
    predictions = Dense(1, activation='sigmoid', name='dense2')(flatten)
    res_model = Model(inputs=[input_image], outputs=[predictions])
    optimizer = Adam(0.0001, 0.9, 0.999, None, 0.000001)
    res_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return res_model


model = get_model()
steps_per_epoch = int((len(image_label_df) - 1) / image_batch_size) + 1
model.fit_generator(batch_iter(image_dir, image_label_df, image_batch_size, epochs=10), steps_per_epoch=steps_per_epoch, epochs=10)
model.save_weights(filepath='/home/yph/weights')






