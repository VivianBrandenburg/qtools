#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import logomaker


from keras import Sequential 
from keras.layers import AveragePooling1D, Conv1D, Flatten, Dense

# =============================================================================
# get data for One Hot encoding
# =============================================================================
def onehot_encoding(seq_in):
    OneHot_encoder ={'A':np.array([0,0,0,1], np.float16),
                     'U':np.array([0,0,1,0], np.float16),
                     'G':np.array([0,1,0,0], np.float16),
                     'C':np.array([1,0,0,0], np.float16), 
                     'N':np.array([0,0,0,0], np.float16)}  
    seq = [OneHot_encoder[y] for y in seq_in ]
    return  np.array(seq)



# =============================================================================
# functions for loading the model
# =============================================================================

def CNN_onehot(seq_len):
    model = Sequential(
        [Conv1D( filters=30, kernel_size=10, activation='relu', input_shape=(seq_len,4), strides=1),
          AveragePooling1D(pool_size=5),
         Flatten(),
         Dense(360, activation='relu'),
            Dense(30, activation='relu'),
         ])
    return model


def model_convert(seq_len=35, weights_path=None):
    model = CNN_onehot(seq_len)
    if not weights_path: 
        weights_path = "model_mod.h5"
    model.load_weights(weights_path)
    return model




# =============================================================================
# some stuff for visualization
# =============================================================================


def make_logo(scores, outfile=None):
    alphabet = ['C', 'G', 'T', 'A']
    
    colorscheme = {
        'G': [0.004, 0.125, 0.305],
        'TU': [0.007, 0.514, 0.569],
        'C': [0.965, 0.863, 0.675],
        'A': [0.996, 0.68, 0.44]
    }
    plotscores = pd.DataFrame(scores, columns=alphabet)
    logo = logomaker.Logo(plotscores,color_scheme=colorscheme)
    plt.ylabel('attribution score')
    plt.xlabel('nt')
    if outfile: 
        plt.savefig(outfile, bbox_inches='tight')
    plt.plot()

