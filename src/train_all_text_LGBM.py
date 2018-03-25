# -*- coding: utf-8 -*-
"""
# @Time    : 2018/3/25 下午1:35
# @Author  : zhanzecheng
# @File    : train_all_text_LGBM.py
# @Software: PyCharm
"""

from lightgbm import LGBMClassifier
import pickle
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import (
    TfidfVectorizer)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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
######## define some hyper ##########
EMBED_SIZES = 300
MAX_LEN = 1000
BATCH_SIZE = 64
EPOCH = 10

#####################################
train_x = []
train_y = []
test_x = []
all_x = []

# with open('../data/images_features.pkl', 'rb') as f:
#     images_features = np.array(pickle.load(f), dtype=np.float32)
# print(type(images_features))
# print('images shape is', images_features.shape)





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

vectorizer = TfidfVectorizer(ngram_range=(1,2),
                             stop_words=STOP_WORD,
                             sublinear_tf=True,
                             use_idf=True,
                             norm='l2',
                             max_features=10000)

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
    'gamma': 4,
    'max_depth': 6,
    'nthread': -1,
    'subsample': 0.5,
    'silent': 0,
    'eta': 0.01,
    'scale_pos_weight': 1,
    'gpu_id': 2,
    'alpha': 0.01,
    'colsample_bytree': 0.5,
    'min_child_weight': 1,
    'objective': 'multi:softmax',
    'num_class': 3,
    }

print('begin to train xgboost')
num_folds = 4

scores = []
kf = KFold(n_splits=num_folds, shuffle=True, random_state=239)
for train_index, test_index in kf.split(train_x):
    # 按照8：2的比例分成训练集和验证集
    y_train, y_test = train_y[train_index], train_y[test_index]

    kfold_X_train = train_x[train_index]
    kfold_X_valid = train_x[test_index]

    clf = LGBMClassifier(objective='multiclass', max_depth=8, n_estimators=128, num_leaves=12, boosting_type="gbdt", learning_rate=0.1, feature_fraction=0.45, colsample_bytree=0.45, bagging_fraction=0.8, bagging_freq=5, reg_lambda=0.2)

    clf.fit(kfold_X_train, y_train, eval_set=[(kfold_X_valid, y_test)], eval_metric='multi_logloss', early_stopping_rounds=50)


    pred = clf.predict(kfold_X_train)
    print(pred)
    accuracy_rate = (np.sum(pred == y_test))/ y_test.shape[0]
    print('Test accuracy using softmax = {}'.format(accuracy_rate))
    scores.append(accuracy_rate)

print('total scores is ', np.mean(scores))




