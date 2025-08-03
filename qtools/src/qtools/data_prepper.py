#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from Bio import Phylo 
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from copy import deepcopy



def read_tree(file):
    return Phylo.read(file, "newick")

def write_tree(tree, file):
    Phylo.write(tree, file, 'newick')


def get_edgelength(tree, names):
    """
    Generate pairwise distance matrix with the edge length between all pairs of taxa

    Parameters
    ----------
    tree : Bio.Phylo.Newick.Tree, reference tree 
    names : list of strings, names of all training sequences 

    Returns
    -------
    distances : distance matrix as pandas.DataFrame 

    """
    # iter over all combinations and get pairwise tree distance
    select_combs = combinations(names, 2)
    # initialize the dataframe for distances with zeros 
    distances = pd.DataFrame(0, index=names, columns=names)
    for a_n, b_n in select_combs: 
        a = next(tree.find_clades(name=a_n))
        b = next(tree.find_clades(name=b_n))
        distances.loc[a_n, b_n] = distances.loc[b_n, a_n] = tree.distance(a, b)
    return distances
        
    


def plot(tree, outfile=None):
    fig, ax = plt.subplots(1,1,figsize=(15, 15))
    Phylo.draw(tree, axes=ax)
    if outfile: 
        fig.savefig(outfile)

  
def prune_tree(tree, selection):
    tree = deepcopy(tree)
    # prune tree to get only terminal leafs that are in training set
    for clade in tree.get_terminals():
        if clade.name not in selection:
            tree.prune(clade)
    return tree 



def get_quartets_from_tree(tree):
    """
    finds all non-redundant quartets for a given tree. 

    Parameters
    ----------
    tree : Bio.Phylo.Newick.Tree object


    Returns
    -------
    minibatches : pandas.DataFrame object.
        list of all possible quartets as split (0,1)|(2,3)

    """
    minibatches = []
    leaf_names = [x.name for x in tree.get_terminals()]
    
    
    # for each internal node   
    for clade in tree.get_nonterminals():
        # choose an inner group 
        for i in clade.clades: 
            ingroup = i.get_terminals()
            ingroup = [x.name for x in ingroup] 
            outgroup = [x for x in leaf_names if x not in ingroup]
            # form pairs for the quartets from both sides of the node
            lefts = [sorted(x) for x in combinations(ingroup, 2)]
            rights = [sorted(x) for x in combinations(outgroup, 2)]
            # create a list of possible quartets 
            quartets = [sorted([x,y]) for x in lefts for y in rights ]
            
            # add the quartets to the minibatches list 
            for i in quartets:
                if i not in minibatches:
                    minibatches.append(i)
    test_minibatching(minibatches)
    print(f'found {len(minibatches)} quartets')
    
    # from pairs to csv-ready dataframe
    minibatches = [[x[0], x[1], y[0], y[1]] for x,y in minibatches]
    minibatches = pd.DataFrame(minibatches)
    return minibatches
    

def sort_quartet(x,y):
    x = sorted(x)
    y = sorted(y)
    
    x = '.'.join(x)
    y = '.'.join(y)
    
    xy ='.'.join(sorted([x,y]))
    return xy




def test_minibatching(minibatches):
    test = sorted([sort_quartet(x,y)for x,y in minibatches])
    assert len(set(test)) == len(test), 'WARNING: minibatches are NOT unique'
        
      
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
