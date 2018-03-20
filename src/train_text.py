# -*- coding: utf-8 -*-
"""
# @Time    : 2018/3/20 下午3:48
# @Author  : zhanzecheng
# @File    : train_text.py
# @Software: PyCharm
"""
import pickle
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.preprocessing import sequence
from model.Attention_lstm import get_model_cnn
from keras.callbacks import EarlyStopping

TRAIN_X = '../data/News_cut_text.txt'
TRAIN_Y = '../data/Text_label.txt'
SAVE_DICT = '../data/word_dict.pkl'
ITEM_TO_ID = '../data/item_to_id.pkl'
ID_TO_ITEM = '../data/id_to_item.pkl'
EMBEDDING_FILE = "../data/word2vec/news12g_bdbk20g_nov90g_dim128/news12g_bdbk20g_nov90g_dim64.txt"
######## define some hyper ##########
EMBED_SIZES = 64
MAX_LEN = 100
BATCH_SIZE = 128
EPOCH = 30

#####################################
train_x = []
train_y = []

with open(ITEM_TO_ID, 'rb') as f:
    item_to_id = pickle.load(f)
with open(ID_TO_ITEM, 'rb') as f:
    id_to_item = pickle.load(f)
with open(SAVE_DICT, 'rb') as f:
    word_dict = pickle.load(f)

STOP_WORD = set()
UNK = len(item_to_id)

with open(TRAIN_X, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        line = line.split('\t')
        for l in line[1:]:
            tmp = []
            for c in l.split(' '):
                # 对于空格的情况做处理
                if c == '' or c == ' ':
                    continue

                # 对于stopword进行处理
                if c in STOP_WORD:
                    continue
                if c in item_to_id:
                    id = item_to_id[c]
                    tmp.append(id)

            # if len(tmp) == 0:
            #     # print(l)
            #     # print('the len of sentence is 0 quit.....')
            #     tmp = [UNK]
            # if len(tmp) > MAX_LEN:
            #     print('Length Exceed!')
            #     tmp = tmp[:MAX_LEN]

            train_x.append(tmp)
train_x = sequence.pad_sequences(train_x, maxlen=MAX_LEN, padding='post',
                                                     truncating='post', value=UNK)

with open(TRAIN_Y, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        line = line.split('\t')[1]
        line = line.split(' ')
        for l in line:
            train_y.append(l)
train_x = np.asarray(train_x)
train_y = np.asarray(train_y)

print('train_x shape is: ', train_x.shape)
print('train_y shape is: ', train_y.shape)



print('vocabulary size : ', len(item_to_id) + 1)

print('begin to load word2vec file')
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()

print('create embedding matrix')
embedding_matrix = np.random.normal(emb_mean, emb_std, (len(item_to_id) + 1, EMBED_SIZES))
for word, i in item_to_id.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
print('end load word2vec')



model = get_model_cnn(MAX_LEN, (len(item_to_id) + 1), embedding_matrix, EMBED_SIZES)
model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')])
model.save('/home/zzc/cool/ckpt/baseline_text.md5')
