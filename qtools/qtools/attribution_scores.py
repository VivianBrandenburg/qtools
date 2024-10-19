#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy 
from sklearn import preprocessing
import numpy as np



hot_to_vector = {x: [1 if i == x else 0 for i in range(4)] for x in range(4)}
def mutagenesis(seq1):
    """
    Generate all possible single-point mutations for a given sequence.

    The function takes a sequence of one-hot encoded nucleotides and
    returns a list of sequences where each position is mutated to all 
    other possible nucleotides. 

    Parameters
    ----------
        seq1 : np.array of one-hot encoded nucleotides

    Returns
    -------
        list : list of mutated sequences
    """
    res = []
    for position, nt in enumerate(seq1):        
        for other_nt in range(4): 
            seq_mutated= deepcopy(seq1)
            seq_mutated[position] = hot_to_vector[other_nt] 
            res.append(seq_mutated)
    return res
    
    


def calculate_mutagensis_scores(predictions, input_length, output_nodes, feature ):
    """
    Calculate normalized mutagenesis scores for a specific feature from model predictions.

    The function reshapes the predictions, normalizes them using L1 normalization, and adjusts 
    the scores by subtracting 0.25.

    Parameters
    ----------
        predictions : ndarray
            model output scores to be reshaped and normalized
        input_length : int
            length of the input sequence
        output_nodes : int
            number of model output features
        feature : int
            index of the specific feature to calculate scores for

    Returns
    -------
        ndarray : normalized mutagenesis scores for the specified feature

    """
    
    # reshape the predictions - each nt has 4 different predictions
    predictions = np.reshape(predictions, (input_length, 4, output_nodes))
    predictions = predictions[:,:,feature]

    # normalized prediction 
    normalized = preprocessing.normalize(predictions, norm='l1')
    normalized = normalized - 0.25
    return normalized
    
