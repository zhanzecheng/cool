from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

image_dir = "/home/zzc/cool/data/sohu/train/"
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

                for i in range(0, 2):
                    tmp_path = dir + '/' + str(i) + name
                    if os.path.exists(tmp_path):
                        path = tmp_path
                        break

                try:
                    img = image.load_img(path, target_size=(224, 224))
                except OSError or AttributeError:
                    img = np.zeros(shape=(224, 224, 3))

                x = image.img_to_array(img)
                x = imagenet_scale(x)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                train_image.extend(x)

            yield (np.array(train_image), shuffle_image_label[train_start:train_end])


def predict_batch_iter(dir, test, batch_size):
    """ Generates a batch iterator"""
    data_size = len(test['image_name'])
    num_batches_per_epoch = int((len(test['image_name']) - 1) / batch_size) + 1
    for batch_num in range(num_batches_per_epoch):
        train_start = batch_num * batch_size
        train_end = min((batch_num + 1) * batch_size, data_size)
        validate_images = []

        for name in test['image_name'][train_start:train_end]:
            path = dir + name

            try:
                img = image.load_img(path, target_size=(224, 224))
            except OSError or AttributeError:
                img = np.zeros(shape=(224, 224, 3))

            x = image.img_to_array(img)
            x = imagenet_scale(x)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            validate_images.extend(x)
        yield np.array(validate_images)

def validation_batch_iter(dir, validation, batch_size):
    """ Generates a batch iterator"""
    data_size = len(validation['new_name'])
    num_batches_per_epoch = int((len(validation['new_name']) - 1) / batch_size) + 1
    for batch_num in range(num_batches_per_epoch):
        train_start = batch_num * batch_size
        train_end = min((batch_num + 1) * batch_size, data_size)
        validate_images = []

        for row in validation['new_name'][train_start:train_end]:

            if row['label'] == 0: path = dir + '0/' + row['new_name']
            else: path = dir + '1/' + row['new_name']

            try:
                img = image.load_img(path, target_size=(224, 224))
            except OSError or AttributeError:
                img = np.zeros(shape=(224, 224, 3))

            x = image.img_to_array(img)
            x = imagenet_scale(x)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            validate_images.extend(x)
        yield np.array(validate_images)

def get_model():
    input_image = Input(shape=(224, 224, 3), name='image_input')
    base_model = ResNet50(input_tensor=input_image, weights='imagenet', include_top=False)
    dense_image1 = Dense(256, activation='relu', name='dense1')(base_model.output)
    dropout_image1 = Dropout(0.3, name='dropout1')(dense_image1)
    flatten = Flatten()(dropout_image1)
    predictions = Dense(1, activation='sigmoid', name='dense2')(flatten)
    res_model = Model(inputs=[input_image], outputs=[predictions])
    optimizer = Adam(0.0001, 0.9, 0.999, None)
    res_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return res_model


# model = get_model()
# weights_path = '/home/yph/weights'
# if os.path.exists(weights_path):
#     print('---------loading weights---------------')
#     model.load_weights(weights_path)
#     print('---------loading success---------------')
#
# else:
#     steps_per_epoch = int((len(image_label_df) - 1) / image_batch_size) + 1
#     checkpoint = ModelCheckpoint(filepath='/home/yph/checkpoint/', monitor='loss', verbose=0, save_best_only=True, mode='auto', period=1)
#     model.fit_generator(batch_iter(image_dir, image_label_df, image_batch_size, epochs=10), steps_per_epoch=steps_per_epoch, epochs=10)
#     model.save_weights(filepath=weights_path)
#
# predict_dir = '/home/zzc/cool/data/sohu/Pic_info_validate/'
# image_nolabel_df = pd.read_csv('../../data/validate_image.csv')
#
# steps = int((len(image_nolabel_df) - 1) / image_batch_size) + 1
# predict = model.predict_generator(predict_batch_iter(predict_dir, image_nolabel_df, image_batch_size), steps=steps)
#
# image_nolabel_df['label'] = predict
# image_nolabel_df.to_csv('/home/yph/test_validate_images_predict.csv', index=False)


# k折交叉检验
image_label_df_size = len(image_label_df)
validation_ratio = 3

positive_image_label_df = image_label_df[image_label_df[image_label_column_name] == 0]
negative_image_label_df = image_label_df[image_label_df[image_label_column_name] == 1]
positive_image_label_df_size = len(positive_image_label_df)
negative_image_label_df_size = len(negative_image_label_df)
positive_validation_size = int(positive_image_label_df_size / validation_ratio)
negative_validation_size = int(negative_image_label_df_size / validation_ratio)

acc = []
for i in range(validation_ratio):
    validation_positive_image_label_df = positive_image_label_df[i * positive_validation_size:min((i + 1) * positive_validation_size, positive_image_label_df_size)]
    validation_negative_image_label_df = negative_image_label_df[i * negative_validation_size:min((i + 1) * negative_validation_size, negative_image_label_df_size)]
    train_positive_image_label_df = pd.concat([positive_image_label_df[:i * positive_validation_size],
                                               positive_image_label_df[(i + 1) * positive_validation_size:]])
    train_negative_image_label_df = pd.concat([negative_image_label_df[:i * negative_validation_size],
                                               negative_image_label_df[(i + 1) * negative_validation_size:]])

    train_image_label_df = pd.concat([train_positive_image_label_df, train_negative_image_label_df])
    validation_image_label_df = pd.concat([validation_positive_image_label_df, validation_negative_image_label_df])

    print('-------No.%d开始训练-----------'% i)
    model = get_model()
    steps_per_epoch = int((len(train_image_label_df) - 1) / image_batch_size) + 1
    # checkpoint = ModelCheckpoint(filepath='/home/yph/checkpoint/', monitor='loss', verbose=0, save_best_only=True, mode='auto', period=1)
    model.fit_generator(batch_iter(image_dir, train_image_label_df, image_batch_size, epochs=10), steps_per_epoch=steps_per_epoch, epochs=10)
    weights = weights + str(i)
    model.save_weights(filepath=weights)

    print('-------No.%d训练结束-----------'% i)

    validation_dir = '/home/zzc/cool/data/sohu/train/'
    steps = int((len(validation_image_label_df) - 1) / image_batch_size) + 1
    predict = model.predict_generator(validation_batch_iter(validation_dir, validation_image_label_df, image_batch_size), steps=steps)

    cont = 0.0
    for (label, predict_label) in zip(validation_image_label_df['label'], predict):
        if (label == 1 and predict_label >= 0.5) or (label == 0 and predict_label < 0.5): cont += 1

    acc.append(cont / len(validation_image_label_df))
    print('-------No.%d: validation_size:%d', len(validation_image_label_df))
    print('-------No.%d: acc:%lf----------'%(i, acc[-1]))

print('avg acc: %lf'%(sum(acc) / len(acc)))



