# -*- coding: utf-8 -*-
"""
# @Time    : 2018/3/25 上午11:58
# @Author  : zhanzecheng
# @File    : train_all_text_svm.py
# @Software: PyCharm
"""

import pickle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import (
    TfidfVectorizer)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
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

images_features = images_features.reshape((images_features.shape[0], -1))
print('images reshape is', images_features.shape)
# svd = TruncatedSVD(n_components=2000)
# images_features = svd.fit_transform(images_features)
# load and depoment images_features test

# with open('../data/images_features10_validate.pkl', 'rb') as f:
#     test_images_features = np.array(pickle.load(f), dtype=np.float32)
# print(type(test_images_features))
# print('images shape is', test_images_features.shape)
#
# test_images_features = test_images_features.reshape((test_images_features.shape[0], -1))
# test_images_features = svd.transform(test_images_features)



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
train_x = np.concatenate((train_x, features), axis=-1)
test_x = np.concatenate((test_x, test_features), axis=-1)

'''
multi:softmax
num_class
'''

xgb_params ={
    'updater': 'grow_gpu',
    'booster': 'gbtree',
    'lambda': 0.1,
    'gamma': 0.7,
    'max_depth': 7,
    'nthread': -1,
    'subsample': 0.5,
    'silent': 0,
    'eta': 0.01,
    'scale_pos_weight': 1,
    'gpu_id': 3,
    'alpha': 0.1,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'objective': 'multi:softprob',
    'num_class': 3,
    }

print('begin to train xgboost')
num_folds = 4

scores = []
kf = KFold(n_splits=num_folds, shuffle=True, random_state=239)
predict = np.zeros((test_x.shape[0], 3))
oof_predict = np.zeros((train_x.shape[0], 3))

ddtest = xgb.DMatrix(test_x)

def check_accuracy(pred, label):
    right = 0
    total = len(pred)
    for count, re in enumerate(pred):
        flag = np.argmax(re)
        if int(flag) == int(label[count]):
            right += 1
    return right / total

# ids = []
for train_index, test_index in kf.split(train_x):
    # 按照8：2的比例分成训练集和验证集
    y_train, y_test = train_y[train_index], train_y[test_index]

    kfold_X_train = train_x[train_index]
    kfold_X_valid = train_x[test_index]
    # train_id = ids


    dtrain = xgb.DMatrix(kfold_X_train, label=y_train)
    dvalid = xgb.DMatrix(kfold_X_valid, label=y_test)

    watchlist = [(dtrain, 'train'), (dvalid, 'test')]

    best= xgb.train(xgb_params, dtrain, 2000, watchlist, verbose_eval=100, early_stopping_rounds=150)

    # 对验证集predict
    pred = best.predict(dvalid)
    # 对验证集的predict结果做accuracy评分
    accuracy_rate = check_accuracy(pred, y_test)
    print('Test error using softmax = {}'.format(accuracy_rate))

    results = best.predict(ddtest)
    predict += results / num_folds

    oof_predict[test_index] = pred

    scores.append(accuracy_rate)

print('total scores is ', np.mean(scores))

# dtrain = xgb.DMatrix(train_x, label=train_y)
# best= xgb.train(xgb_params, dtrain, 800, verbose_eval=100)
# predict = best.predict(ddtest)
with open(SAVE_RESULT_PATH + 'xgboost_change_oof_' + str(np.mean(scores)) + '.txt', 'wb') as f:
    pickle.dump(oof_predict, f)


with open(SAVE_RESULT_PATH + 'xgboost_change_pred_' + '.txt', 'wb') as f:
    pickle.dump(predict, f)

print('done')
