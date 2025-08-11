# -*- coding: utf-8 -*-

import numpy as np


OneHot_encoder = {'A':np.array([0,0,0,1], np.float16),
                 'U':np.array([0,0,1,0], np.float16),
                 'G':np.array([0,1,0,0], np.float16),
                 'C':np.array([1,0,0,0], np.float16)}  

Alphabet = ['C', 'G', 'U', 'A']

def onehot_encoding(seq_in):
    """
    one-hot encode data
    """
    seq = [OneHot_encoder[y] for y in seq_in ]
    return  np.array(seq)


