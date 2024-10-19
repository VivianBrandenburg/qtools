#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 20:04:42 2024

@author: viv
"""

from keras import Sequential 
from keras.layers import AveragePooling1D, Conv1D, Dropout, Flatten, Dense

def CNN_onehot(seq_len):
    
    model = Sequential(
        [Conv1D( filters=30, kernel_size=10, activation='relu', input_shape=(seq_len,4), strides=1),
          AveragePooling1D(pool_size=5),
         # MaxPooling1D(pool_size=5, strides=3),
         Flatten(),
         Dense(360, activation='relu'),
            Dense(30, activation='relu'),
         ])
    return model


def model_convert(seq_len=350, weights_path=None):
    model = CNN_onehot(seq_len)
    if not weights_path: 
        weights_path = "model_mod.h5"
    model.load_weights(weights_path)
    return model


