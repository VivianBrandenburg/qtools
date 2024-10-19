# -*- coding: utf-8 -*-
from qtools.encoding import onehot_encoding
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Flatten, Dense, Dropout

   
def CNN_onehot_model(seq_len):
    model = Sequential(
        [Conv1D(filters=30, kernel_size=10, strides=1,
                activation='relu', input_shape=(seq_len,4)),
           AveragePooling1D(pool_size=5),
            Dropout(0.2),
         Flatten(),
         Dense(360, activation='relu'),
         Dense(30, activation='relu')
         ])
    return model



class CNN_ONEHOT:
    def __init__(self, seq_length):
        self.model = CNN_onehot_model(seq_length)
        self.encoding_function = onehot_encoding        
        self.seq_len = seq_length
        self.input_dims = (1, seq_length, 4)
        self.output_dims = 30


