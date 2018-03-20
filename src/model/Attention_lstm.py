# -*- coding: utf-8 -*-
"""
# @Time    : 2018/3/20 下午3:35
# @Author  : zhanzecheng
# @File    : Attention_lstm.py
# @Software: PyCharm
"""
from keras.models import *
from keras.layers.core import *
from keras.layers import merge, Dense, Embedding, Input, Concatenate, Conv1D, GlobalMaxPooling1D, Activation, TimeDistributed, Flatten, \
    RepeatVector, Permute, multiply
from model.AttentionWithContext import AttentionWithContext
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, concatenate, GRU, GlobalAveragePooling1D, MaxPooling1D, \
    SpatialDropout1D, BatchNormalization


def get_model_cnn(maxlen, max_features, embedding_matrix, embed_size):
    main_input = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(main_input)
    x = SpatialDropout1D(0.5)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    x = AttentionWithContext()(x)
    # avg_pool = GlobalAveragePooling1D()(x)
    # max_pool = GlobalMaxPooling1D()(x)
    # concat2 = concatenate([avg_pool, num_vars, max_pool], axis=-1)
    dense2 = Dense(1, activation="sigmoid")(x)
    res_model = Model(inputs=[main_input], outputs=dense2)
    # res_model = Model(inputs=[main_input], outputs=main_output)
    res_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    res_model.summary()
    return res_model

