# -*- coding: utf-8 -*-
# @Time    : 2018/3/24 下午11:52
# @File    : svm_classification.py

import pickle
import numpy as np
import os
import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from keras.preprocessing import sequence
from keras.utils import np_utils
# from model.TextCNN import get_model_cnn
from keras.callbacks import EarlyStopping
from keras.layers import *
from keras.models import *
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import (
    TfidfVectorizer)
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import GridSearchCV

PREDICT_FILE = '../data/Text_Predict.txt'
TEXT_X = '../data/News_cut_validate_text.txt'
SAVE_DICT = '../data/word_dict.pkl'
ITEM_TO_ID = '../data/item_to_id.pkl'
ID_TO_ITEM = '../data/id_to_item.pkl'
EMBEDDING_FILE = "utils/chinese_txt"
TRAIN_X = '../data/All_train_file.txt'
ALL_FILE = '../data/All_file.txt'
TEST_FILE = '../data/All_validate_file.txt'
FEATURES_FILE = '../data/train_text_feature.csv'
TRAIN_X = '../data/All_train_file_new.txt'
ALL_FILE =  '../data/All_file_new.txt'
TEST_FILE = '../data/All_validate_file_new.txt'
######## define some hyper ##########
EMBED_SIZES = 300
MAX_LEN = 1000
BATCH_SIZE = 64
EPOCH = 70

#####################################
train_x = []
train_y = []
test_x = []
all_x = []

with open('../data/images_features.pkl', 'rb') as f:
    images_features = np.array(pickle.load(f), dtype=np.float32)
print(type(images_features))
print('images shape is', images_features.shape)

with open(ITEM_TO_ID, 'rb') as f:
    item_to_id = pickle.load(f)
with open(ID_TO_ITEM, 'rb') as f:
    id_to_item = pickle.load(f)
with open(SAVE_DICT, 'rb') as f:
    word_dict = pickle.load(f)
df = pd.read_csv(FEATURES_FILE)


STOP_WORD = set()
UNK = len(item_to_id)
word_count = {}
with open(ALL_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        all_x.append(line)

with open(TEST_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        line = line.split('\t')
        line = line[1]
        test_x.append(line)
ids = []
with open(TRAIN_X, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        line = line.split('\t')
        id = line[0]
        ids.append(id)
        label = int(line[2])
        line = line[1]
        # tmp = []
        #
        # for l in line:
        #     if l == '' or l == ' ':
        #         continue
        #
        #     if l in STOP_WORD:
        #         continue
        #
        #     if l in item_to_id:
        #         id = item_to_id[l]
        #         tmp.append(l)
        #
        # if len(tmp) == 0:
        #     tmp = [UNK]
        #
        # if len(tmp) > MAX_LEN:
        #     # print('Length Exceed!')
        #     tmp = tmp[:MAX_LEN]
        train_y.append(label)
        train_x.append(line)


# features = []
# for id in tqdm.tqdm(ids):
#     tmp = list(df.loc[df.id == id].values[0])
#     del tmp[6]
#     features.append(tmp)

with open('../data/features.pkl', 'rb') as f:
    features = pickle.load(f)

# 把list的list变成array
features = np.array(features, dtype=np.float32)

scaler = MinMaxScaler()
features = scaler.fit_transform(features)

train_x = np.asarray(train_x)
train_y = np.asarray(train_y)

with open('../data/train_x_250.pkl', 'rb') as f:
    train_x = pickle.load(f)
# with open('../data/train_x_500.pkl', 'wb') as f:
#     pickle.dump(train_x, f)
# with open('../data/test_x_500.pkl', 'wb') as f:
#     pickle.dump(test_x, f)
# with open('../data/test_x_500.pkl', 'rb') as f:
#     test_x = pickle.load(f)

# train_x = train_x[:1000]
# train_y = train_y[:1000]

print('train_x size: ', np.shape(train_x))
print('train_y size: ', np.shape(train_y))


# grid = GridSearchCV(SVC(), param_grid={"C":[0.1, 1, 10], "gamma": [1, 0.1, 0.01]}, cv=4)
# grid.fit(train_x, train_y)
# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))

oof_predict = np.zeros((train_x.shape[0], 1))
num_folds = 5
# 用KFold的形式来训练模型，具体可以自己卡看KFold的定义
# random_state 随机种子数
scores = []
kf = KFold(n_splits=num_folds, shuffle=True, random_state=239)
for train_index, test_index in kf.split(train_x):

    print('length of train_index: ', len(train_index))
    print('length of test_index: ', len(test_index))
    # 按照8：2的比例分成训练集和验证机
    y_train, y_test = train_y[train_index], train_y[test_index]

    kfold_X_train = train_x[train_index]
    kfold_X_test = train_x[test_index]

    # 训练model
    svc_model = SVC(kernel='linear', class_weight='balanced')
    svc_model.fit(kfold_X_train, y_train)

    # predict += model.predict(X_test, batch_size=1000) / num_folds

    predict = svc_model.predict(kfold_X_test).reshape((len(test_index), 1))
    oof_predict[test_index] = predict
    print('predict: ', predict)
    print('label: ', y_test)
    # # 评价函数，这里应该是accurancy
    # print('y_train', y_train)
    cv_score = accuracy_score(y_test, oof_predict[test_index])

    scores.append(cv_score)
    print('score: ', cv_score)

print('scores: ', scores)
print('mean score: ', np.mean(scores))