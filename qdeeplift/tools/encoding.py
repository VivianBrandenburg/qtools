# -*- coding: utf-8 -*-


from tensorflow.keras import metrics
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt 



Max_length=350




# =============================================================================
# get data for matrix encoding 
# =============================================================================


matrix_rules_2 = {'CG':1, 'AU':0.66, 'GU':0.33, 'GC':1, 'UA':0.66, 'UG':0.33,
                    'AA':0, 'GG':0, 'CC':0, 'UU':0, 'GA':0, 'AG':0, 'CA':0, 'AC':0, 'UC':0, 'CU':0 ,
                    'NA':0, 'NG':0, 'NC':0, 'NU':0, 'AN':0, 'GN':0, 'CN':0, 'UN':0, 'NN':0 }    

def matrix_encoding(seq):
    seq_rev = seq[::-1]
    myTable = []
    for i in seq:
        row=[matrix_rules_2[''.join([i,j])] for j in seq_rev]
        myTable.append(row)
    return np.array(myTable).reshape(Max_length,Max_length,1)
    



def get_data_matrix_encoded(infile):
    infile=pd.read_csv(infile)
    data = [matrix_encoding(seq) for seq in infile.sequence]
    label = infile.label
    return np.array(data), np.array(label)


