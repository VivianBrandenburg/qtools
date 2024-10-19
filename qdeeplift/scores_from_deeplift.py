#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

import deeplift
from deeplift.layers import NonlinearMxtsMode
from deeplift.conversion import kerasapi_conversion as kc

import utils


# =============================================================================
# set variables
# =============================================================================

# model and data to look at 
data = '../data/simdata/s8_n20m3_r7_run1/sequences.csv'
weights = '../models/simulated_data/quartet_plus_siamese/a/weights/m13_weights.h5'


# feature vector index
# feature vector indices in paper start with 1, here indices strat with 0 
# for seeing feature 10 from paper, choose feature=9
featureVector = 9 

# selected species and reference species
gammas = list(range(7))
alphas = list(range(7,13))
ingroup = alphas # sequences to calculate scores for
reference = gammas # reference sequences used for score calculation


# produce a name for outfiles from selected variables
outname = [f'f{featureVector}', 
           'in'+'.'.join([str(x) for x in ingroup]), 
           'out'+'.'.join([str(x) for x in reference])]
outname = '__'.join(outname)

# wether to check if the deeplift model was build correctly
sanity_check=True


# =============================================================================
# read in data and onehot encode them
# =============================================================================

data = pd.read_csv(data)
data['sequences'] = data.seq
onehot_data = np.array([utils.onehot_encoding(x) for x in data.seq]) 

seq_len = len(data.sequences.iloc[0])

# =============================================================================
# load the keras model
# =============================================================================


#load the convertet quartet model
keras_model = utils.model_convert(seq_len=seq_len, weights_path=weights)
model_weights_convert = weights.replace('.h5', '_tf1convert.h5')
keras_model.save(model_weights_convert)


# =============================================================================
# convert the model to deeplift model
# =============================================================================

dl_method = 'rescale_conv_revealcancel_fc'
dl_model = kc.convert_model_from_saved_files(h5_file=model_weights_convert,
    nonlinear_mxts_mode=NonlinearMxtsMode.DeepLIFT_GenomicsDefault)

# =============================================================================
# sanity check
# =============================================================================


if sanity_check: 
    #make sure predictions are the same as the original model
    from deeplift.util import compile_func
    dl_predict = compile_func([dl_model.get_layers()[0].get_activation_vars()],
                               dl_model.get_layers()[-1].get_activation_vars())
    og_predictions = keras_model.predict(onehot_data, batch_size=200)
    convert_predictions = deeplift.util.run_function_in_batches(
                                    input_data_list=[onehot_data],
                                    func=dl_predict,
                                    batch_size=200,
                                    progress_update=None)
    if np.max(np.array(convert_predictions)-np.array(og_predictions)) == 0:
              print('\nSANITY CHECK: OK\n')
    else:
        print("\nSANITY CHECK: FAILED.\nmaximum difference in predictions:",
              np.max(np.array(convert_predictions)-np.array(og_predictions)))
        assert np.max(np.array(convert_predictions)-np.array(og_predictions)) < 10**-5
    
    
    

# =============================================================================
# prepare score computation
# =============================================================================

#make lists of indices so that each sequence is tested against each reference 
input_idx = [[i]*len(reference) for i in ingroup]
input_idx = sum(input_idx, [])
ref_idx = reference*len(ingroup)

# get one-hot-encoded data
dl_input = [np.array([onehot_data[i,:,:] for i in input_idx])]
dl_ref = [np.array([onehot_data[i,:,:] for i in ref_idx])]

# get species names for visualization 
ingroup_species = [data.iloc[i].spec for i in ingroup]
outgroup_species = [data.iloc[i].spec for i in reference]



# =============================================================================
# calculate importance scores
# =============================================================================

# choose a function to calculate scores
scoring_function = dl_model.get_target_contribs_func(find_scores_layer_idx=0,
                                                     target_layer_idx=-2)
# actual  importance scores computation
scores = scoring_function(task_idx=featureVector, 
                          input_data_list=dl_input,
                          input_references_list=dl_ref, 
                          batch_size=10,
                          progress_update=None)

# calculate mean socres over all input sequences
scores = np.reshape(scores, [len(ingroup), len(reference)]+[seq_len,4])
mean_scores = np.mean(scores, axis=1)

# multiply scores with input
original_onehot = np.array([onehot_data[i] for i in ingroup])
mean_scores_corrected = np.multiply(mean_scores, original_onehot)
mean_scores_corrected = np.mean(mean_scores_corrected, axis=0)

# =============================================================================
# plot scores 
# =============================================================================
utils.make_logo(mean_scores_corrected, outfile=outname+'.png')







