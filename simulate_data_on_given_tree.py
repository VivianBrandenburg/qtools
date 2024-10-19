#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from Bio import Phylo
from qtools.simulate_data import patch_tree
from qtools.utils import create_dir


# get names of species used for trainng
selection = 'data/rRNA/train.csv'
selection = pd.read_csv(selection).spec.to_list()

# get reference tree 
tree = 'data/rRNA/train.ph'
tree = Phylo.read(tree, "newick")

# add the SimTree functions to the tree object
tree = patch_tree(tree)

# prune the tree to get a tree with only used species 
tree.prune_from_list(selection)



    
# make mutated sequences
tree.simulate_stable('stable', 8)
tree.simulate_evolving('evolving', 20,3)
tree.simulate_random('random',7)

# prepare path for outfiles
scheme = tree.mutationscheme.as_str()
outdir = 'data/some_example_output/'
create_dir(outdir)

# write results to path
tree.write_as_tree(outdir)
tree.write_as_csv(outdir)

# if you create sequences in a loop, make sure to 
# delete the dictionaries and mutation scheme to prep for next run. 
# to avoid sequence mixups
tree.clear_sequences()
