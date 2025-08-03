#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Nadam

import qtools as qt
from qtools.lossfunctions import  siamloss_siamnet, Xsq_SiamReg, siamloss, eloss
from qtools.quartettroutines import siamesemodel, quartetmodel
from qtools.data_tracking import metadata, mutationscheme, Tracking

# =============================================================================
# setting up file names
# =============================================================================

# file with training sequences 
seqs_file = 'data/simdata/s8_n20m3_r7_run1/sequences.csv'

# file with mutation scheme 
mutation_scheme_file = seqs_file.replace('sequences.csv', 'mutationscheme.json')



# file with edge length distance matrix 
edge_distance = 'data/rRNA/train_patristic.csv'

# file with siamese minibatches 
minibatches_siamese = 'data/rRNA/train_siamesebatches.csv' 


# file with quartet minibatches 
minibatches_quartet = 'data/rRNA/train_minibatches.csv'


# path for writing results 
out_path = 'trained_models_test/'

# create a new subdir in your out_dir from local time
out_dir = qt.update_dir(out_path)


# =============================================================================
# setting up training variables 
# =============================================================================


# variables for training 
learning_rate = 0.001
batch_size = 1
epochs = 100



# choose which network to use by setting the mode to 'quartet' or 'siamese' 
# quartet mode can be used with or without siamese regulation. If you want no siamese regulation, set mode='quartet' and sigma=0
mode = 'quartet' 

# sigma is the relation between quartet loss and siamese correction. 
# loss = quartet loss + (siamese loss * sigma)
# If you use a siamese network, sigma will automatically be set to 'nan'
sigma = 0.1 




# if running as siamese model 
if mode == 'siamese':
    Multimodel = siamesemodel
    minibatch_file = minibatches_siamese
    sigma = 'nan'
    loss_function = siamloss_siamnet
    metrics = None

# if running as quartet model 
if mode == 'quartet':
    Multimodel = quartetmodel
    minibatch_file = minibatches_quartet 
    loss_function = Xsq_SiamReg(sigma)
    metrics = [eloss, siamloss] 

  


# =============================================================================
# prepare data 
# =============================================================================

# set up training data
data = pd.read_csv(seqs_file)
data = qt.qdata(data)


#  read minibatches 
minibatches = pd.read_csv(minibatch_file, index_col=0)
edge_distance = pd.read_csv(edge_distance, index_col=0)
scoring_batches = pd.read_csv(minibatches_quartet, index_col=0)


# =========================================================================
# set up model 
# =========================================================================


# set up model 
seq_len = data.get_seqlen()
singlemodel = qt.CNN_ONEHOT(seq_len, output_dims=32)


# build quartetnet or siamesenet
# I used `singlemodel` to keep track of the encoding function and some other stuff, but it should be easy to initialize the `Multimodel` with every other (Sequential) tensorflow model. If you did that, you might also want to replace the encoding function provided to data.encode in the next step.
multimodel = Multimodel.from_basemodel(singlemodel.model)
multimodel.compile(optimizer=Nadam(learning_rate=learning_rate),
                    loss=loss_function,
                    metrics=metrics
                    )


# encode the training data and split them in encoded data and species names
data.encode(singlemodel.encoding_function)
x_encoded, x_species = data.get_data()





# =============================================================================
# record metadata and mutation scheme 
# =============================================================================
    

# get infos for simulated data
# The mutation scheme is written when you simulated data with this package.
# I use it for easier data selection in the dashboard, but
# you do not necessarily need it, everything works without it as well 
try:
    ms = mutationscheme.from_json(mutation_scheme_file)
    mutation_scheme = ms.scheme
except FileNotFoundError:
    print("---> I could not find the mutation scheme. Writing 'unknown' to metadata.")
    mutation_scheme = 'unknown'
    


# track the metadata
# you can inspect which variables are written to metadata with metadata.collected_keys
# you can modify which data are tracked in qtools.data_tracking.metadata
print(locals()['out_dir'])
metadata.record(locals()).write()



# =========================================================================
# evaluate model before first run
# =========================================================================

# get feature vectors
prediction = multimodel.predict(np.array(x_encoded))

# calculate the distance matrix (euclidean distances) from the feature vectores
matrix_i = multimodel.get_distance_matrix(prediction)  

# create a splits diagram from the distance matrix with splitstree.
# the default directory for splitstree is '~/splitstree4/SplitsTree' 
# but you can change it with option 'splitstree_location'
qt.matrix2nexus(matrix_i, x_species,  out_dir + 'nexus/0.nex',
                plot_now=True)
        

# evaluate quartet scores
scores = qt.get_qscores(matrix_i, x_species, scoring_batches)


# evaluate loss
batches = data.batchmaker(minibatches, edge_distance)
losses = multimodel.evaluate(batches, return_dict=True)


# initialize the tracking
t = Tracking(out_dir, y_names=x_species)

# track evaluation before first run 
t.trackall(epoch=0, feature_vectors=prediction, score=scores, loss=losses)
t.writeall()
t.write_species_names()



# =============================================================================
# train model 
# =============================================================================


for e in range(epochs):
    
   
    # some quartet models without siamese regulation tend to collapse 
    # after some epochs and return only nan. 
    if type(losses['loss'])==float or np.isfinite(losses['loss'][-1]):  # stop if model collapsed
        
        # prepare training batches
        batches = data.batchmaker(minibatches, edge_distance)
        
        # train quartetnet (or siamesenet)
        multimodel.fit(batches, batch_size = batch_size)
        
        # evaluate epoche
        losses = multimodel.history.history
    
        # calculate matrix with euclidean distances
        prediction = multimodel.predict(x_encoded)
        matrix_i = multimodel.get_distance_matrix(prediction)
        
        # check how many quartets are in right split (calculate quartet scores)
        scores = qt.get_qscores(matrix_i, x_species, scoring_batches)
        
        # make splitstree diagram from distacne matrix
        qt.matrix2nexus(matrix=matrix_i, taxa=x_species, 
                        nexusfile=f'{out_dir}nexus/{e+1}.nex', plot_now=True)
        

        # track the evaluated measures 
        t.trackall(e+1, feature_vectors=prediction, loss=losses, score=scores)
        t.writeall()
    
    
        # you can immediately plot the results if you want 
        t.plotall(e+1, 'test', plot_live=True)
        
        
        # save model weights 
        multimodel.basemodel.save_weights(f'{out_dir}/weights/m{e}_weights.h5')
