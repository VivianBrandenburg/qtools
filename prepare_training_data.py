#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# prep minibatches and edgelength distances for testing data
# you need the names of the species you want to involve and a reference tree 
# the species names here are taken from csv files with columns [species name],[sequence]
"""
import pandas as pd
import  qtools.data_prepper as dp
import itertools

# get reference tree 
ref_tree = 'data/rRNA/entire_tree.ph'
ref_tree = dp.read_tree(ref_tree)


# prepare names of species used for testing
train_names = 'data/15spec.2/train.csv'
train_names = pd.read_csv(train_names).spec.to_list()



# name the outfiles 
outfile_prefix = 'data/rRNA/test'


# prune tree to names from testing and write to file 
tree = dp.prune_tree(ref_tree, train_names)
dp.write_tree(tree, outfile_prefix+'.ph')

# plot the tree (just for visual control)
dp.plot(tree, outfile_prefix + '.png' )


# calculate minibatches for quartet training
minibatches = dp.get_quartets_from_tree(tree)
minibatches.to_csv(outfile_prefix  + '_minibatches.csv')

# calculate minibatches for siamese training 
siamesebatches = itertools.combinations(train_names, 2)
pd.DataFrame(siamesebatches).to_csv(outfile_prefix  + '_siamesebatches.csv')

# calculate edge distances and write to file 
distances = dp.get_edgelengths(tree, train_names)
distances.to_csv(outfile_prefix  + '_patristic.csv')
