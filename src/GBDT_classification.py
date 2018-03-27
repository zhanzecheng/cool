# -*- coding: utf-8 -*-
# @Time    : 2018/3/27 下午1:35
# @File    : GBDT_classification.py

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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics

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
SAVE_RESULT_PATH = '../data/result/'
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
FEATURES_test_FILE = '../data/test_text_feature.csv'
STOP_WORD_FILE = '../data/stopword.txt'

#####################################
train_x = []
train_y = []
test_x = []
all_x = []

# load and depoment images_features train
with open('../data/images_features10.pkl', 'rb') as f:
    images_features = np.array(pickle.load(f), dtype=np.float32)
print(type(images_features))
print('images shape is', images_features.shape)

# 对图片特征进行降维
images_features = images_features.reshape((images_features.shape[0], -1))
print('images reshape is', images_features.shape)
svd = TruncatedSVD(n_components=2000)
images_features = svd.fit_transform(images_features)
# load and depoment images_features test

with open('../data/images_features10_validate.pkl', 'rb') as f:
    test_images_features = np.array(pickle.load(f), dtype=np.float32)
print(type(test_images_features))
print('images shape is', test_images_features.shape)

test_images_features = test_images_features.reshape((test_images_features.shape[0], -1))
test_images_features = svd.transform(test_images_features)



with open(ITEM_TO_ID, 'rb') as f:
    item_to_id = pickle.load(f)
with open(ID_TO_ITEM, 'rb') as f:
    id_to_item = pickle.load(f)
with open(SAVE_DICT, 'rb') as f:
    word_dict = pickle.load(f)

df = pd.read_csv(FEATURES_FILE)
df_test = pd.read_csv(FEATURES_test_FILE)

STOP_WORD = set()
UNK = len(item_to_id)
word_count = {}

with open(STOP_WORD_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        STOP_WORD.add(line)

with open(ALL_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        all_x.append(line)

test_ids = []
with open(TEST_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        line = line.split('\t')
        test_ids.append(line[0])
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
        train_y.append(label)
        train_x.append(line)


# for id in tqdm.tqdm(test_ids):
#     tmp = list(df_test.loc[df_test.id == id].values[0])
#     del tmp[6]
#     test_features.append(tmp)

with open('../data/test_features.pkl', 'rb') as f:
    test_features = pickle.load(f)

with open('../data/features.pkl', 'rb') as f:
    features = pickle.load(f)

# 把list的list变成array
features = np.array(features, dtype=np.float32)

scaler = MinMaxScaler()
features = scaler.fit_transform(features)
test_features = scaler.transform(test_features)

train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
# train_y = np_utils.to_categorical(train_y)
print('train_x shape is: ', train_x.shape)
print('train_y shape is: ', train_y.shape)
# vectorizer text
#
# vectorizer = TfidfVectorizer(ngram_range=(1,2),
#                              stop_words=STOP_WORD,
#                              sublinear_tf=True,
#                              use_idf=True,
#                              norm='l2',
#                              max_features=10000)

# LSA Pipeline
# svd = TruncatedSVD(n_components=250)
# lsa = make_pipeline(vectorizer, svd)
# #fit lsa
# print('begin')
# lsa.fit(all_x)
# print('end_fit')
# train_x = lsa.transform(train_x)
# test_x = lsa.transform(test_x)
# print('end_transform')
with open('../data/train_x_250.pkl', 'rb') as f:
    train_x = pickle.load(f)

with open('../data/test_x_250.pkl', 'rb') as f:
    test_x = pickle.load(f)
#
# with open('../data/train_x_250.pkl', 'wb') as f:
#     pickle.dump(train_x, f)
# with open('../data/test_x_250.pkl', 'wb') as f:
#     pickle.dump(test_x, f)
# quit()
# Now we concatenate our file
# train_x = np.concatenate((train_x, features, images_features), axis=-1)
# test_x = np.concatenate((test_x, test_features, test_images_features), axis=-1)
train_x = np.concatenate((train_x, features), axis=-1)
test_x = np.concatenate((test_x, test_features), axis=-1)


num_folds = 5
scores = []
kf = KFold(n_splits=num_folds, shuffle=True, random_state=239)

# train_x = train_x[:10]
# train_y = train_y[:10]

def check_accuracy(pred, label):
    right = 0
    total = len(pred)
    for count, re in enumerate(pred):
        flag = np.argmax(re)
        if int(flag) == int(label[count]):
            right += 1
    return right / total

scores = []
predict = np.zeros((test_x.shape[0], 3))
oof_predict = np.zeros((train_x.shape[0], 3))

for train_index, test_index in kf.split(train_x):

    train_data = train_x[train_index]
    train_label = train_y[train_index]
    test_data = train_x[test_index]
    test_label = train_y[test_index]

    model = GradientBoostingClassifier()
    model.fit(train_data, train_label)

    preds_class = model.predict_proba(test_data)

    print('preds_class: ', preds_class)

    print('test_label: ', test_label)

    # cv_score = accuracy_score(test_label, preds_class)
    accuracy_rate = check_accuracy(preds_class, test_label)
    print('Test error using softmax = {}'.format(accuracy_rate))

    results = model.predict_proba(test_x)
    predict += results / num_folds

    oof_predict[test_index] = preds_class

    scores.append(accuracy_rate)

print('total scores is: ', np.mean(scores))

with open(SAVE_RESULT_PATH + 'GBDT_oof_' + str(np.mean(scores)) + '.txt', 'wb') as f:
    pickle.dump(oof_predict, f)


with open(SAVE_RESULT_PATH + 'GBDT_pred_' +str(np.mean(scores)) + '.txt', 'wb') as f:
    pickle.dump(predict, f)

print('done')