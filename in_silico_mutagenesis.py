#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import qtools as qt
from qtools.attribution_scores import calculate_mutagensis_scores, mutagenesis



# =============================================================================
# set up input and output files
# =============================================================================
seqs_file = '.data/simdata/s8_n20m3_r7_run1/sequences.csv'
weights = 'models/simulated_data/quartet_plus_siamese/a/weights/m10_weights.h5'
outdir = 'plots/mutagenesis'


# =========================================================================
# set up data and model 
# =========================================================================

#set up training data
data = pd.read_csv(seqs_file)
data = qt.qdata(data)


# create the model and load in the weights
model = qt.CNN_ONEHOT(data.get_seqlen())
coremodel = model.model
coremodel.load_weights(weights)


# encode data, split data in encoded data and species names
data.encode(model.encoding_function)
x_encoded, x_species = data.get_data()

# =============================================================================
# choose feature vectors to analyze
# =============================================================================

alphas = list(range(0,6))
gammas = list(range(6,13))

# put feature = 2 if you want to see feature 3 from the paper. 
# feature vector in paper starts with 1, index starts with 0
specs =  gammas
feature = 2
            
# =============================================================================
# mutagenesis and attribution score calculation 
# =============================================================================

# we are iterating over all selected species to calculate average attributions scores for them 
attribution_res = []
for spec in specs:
    # make all mutagenesis for the seuqence
    mutated_seqs = mutagenesis(x_encoded[spec])
    
    # forward pass for all mutated sequences
    predictions = coremodel.predict(np.array(mutated_seqs))
    
    
    #calculate attribution scores 
    attributions = calculate_mutagensis_scores(predictions, 
                                             input_length = model.input_dims[1], 
                                             output_nodes = model.output_dims,
                                             feature = feature)
    
    attribution_res.append( attributions)

    
  
# average over all attribution scores for all species
attribution_res = np.array(attribution_res).sum(axis=0)/len(specs)
  

# =============================================================================
# plot results
# =============================================================================
mymotif = pd.DataFrame(attribution_res, columns= ['C', 'G', 'T', 'A'])

qt.motif.make_logo(mymotif)
plotname = f"{outdir}/{'.'.join([str(x) for x in specs])}_f{feature}__normalized.svg"

plt.savefig(plotname, bbox_inches='tight')
plt.show()

