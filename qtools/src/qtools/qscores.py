#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd 

def quartet2score(dm, split):
    """
    Calculate whether the predicted distance matrix is in accordance with the provided split

    Parameters
    ----------
    dm : distance matrix with distances between predicted y (aka distances between feature vectors)
    split : list of four ints, referring to the positions in the distance matrix 

    Returns
    -------
    int
        0 if split is optimal
        1 if split is suboptimal (f > e)
        2 if split is entirely wrong (A is opponent of B, no edge could be minimized to get the right split) 
       

    """
    
    # return 
    
    
    ab = dm.loc[split[0], split[1]]
    ac = dm.loc[split[0], split[2]]
    ad = dm.loc[split[0], split[3]]
    bc = dm.loc[split[1], split[2]]
    bd = dm.loc[split[1], split[3]]
    cd = dm.loc[split[2], split[3]]
    
    
    # correct split = ab|cd 
    #incorrect split = ac|bd or ad|bc
    
    
    
    # this split is optimal if e < f  
    e1 = (ad + bc - ac - bd) / 2
    f1 = (ad + bc - ab - cd) / 2
    
    # this split is optimal if e < f  
    e2 = e1 * -1 
    f2 = (ac + bd - ab - cd) /2
    
    # this split is never optimal. it is the totally wrong split 
    e3 = f1 * -1 
    f3 = f2 * -1 
    
    if e1>=0 and f1 >=0 :
        res = 0 if e1 < f1 else 1
        return res 
    elif e2 >= 0 and f2 >= 0:
        res = 0 if e2 < f2 else 1
        return res        
    elif e3 >= 0 and f3 >= 0:
        return 2
    else:
        assert False, 'There was no correct split found.'
    




def get_qscores(dm, x_species, minibatches):
    """
    Calculate whether the predicted distance matrix is in accordance with the provided minibatches

    Parameters
    ----------
    dm : distance matrix with distances between predicted y (aka distances between feature vectors)
    minibatches : list of allowed minibatches (as splits ab|cd)
    x_names : names of species as ordered in distance matrix 

    Returns
    -------
    scores : list of percentage for [optimal, suboptimal, wrong] quartetts

    """
    dm=pd.DataFrame(dm, index=x_species, columns=x_species)
    scores = [0,0,0]
    for batch in minibatches.values: 
        score_int = quartet2score(dm, batch)
        scores[score_int] += 1
    total = sum(scores)
    scores = [x/total for x in scores]
    return scores 
