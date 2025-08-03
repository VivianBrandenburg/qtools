#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import itertools



class qdata(pd.DataFrame):
    
    def __init__(self, data, sort=True, encoding_function=False):
        """
        Parameters
        ----------
        data : pandas.DataFrame 
            Dataframe including the columns 'spec' and 'seq' (species names and sequences)
        sort : bool
            Sort alphabetically by spec. The default is True. 
        encoding : bool
            If encoding function is provided, encodes the data and adds them to DataFrame.
            The default is False.

        Returns
        -------
        pandas Dataframe with data for quartetnet.

        """
        
        super(qdata, self).__init__()
        if sort:
            data = data.sort_values('spec')
        data.index = data['spec']
        pd.DataFrame.__init__(self, data)
        
        if encoding_function:
            self.encode(encoding_function)
        
    def get_seqlen(self):
        return len(self['seq'][0])
    
    def encode(self, encoding_function):
        """ adds encoded sequences to DataFrame."""
        self['encoded'] = [encoding_function(x) for x in self.seq] #??
        
    def get_data(self):    
        x = list(self.encoded)
        y = list(self.spec)
        return x, y
           
     

    def __get_y_true(self, minibatches, distance_matrix): 
        """reads list of minibatches and distacne matrix and returns pairwise distances in minibatch-order 
         [i, j, k, l] ->  [Dij, Dik, Dil, Djk, Djl, Dkl]
        """
        pairsDistances = self.qName2pairDist(minibatches, distance_matrix)
        pairsDistances = pd.DataFrame(pairsDistances)
        
        return pairsDistances

    
    
    def qName2pairDist(self, minibatches, dm):
        """Gives a list of all pairwise distances for a list of species
         [i, j, k, l] ->  [Dij, Dik, Dil, Djk, Djl, Dkl]
        """
        pairs = [[x for x in itertools.combinations(r, 2)] 
                 for i,r in minibatches.iterrows()]
        pairsDistances = [[dm.loc[x,y] for x,y in batch] for batch in pairs]
        return pairsDistances


        
        
        
    def batchmaker(self, minibatches, distance_matrix, epochs=1, batch_size=1):
        """
        Splits dataFrame in batches, according to a file in which minibatches are listet.

        Parameters
        ----------
        minibatches : list of minibatches as pandas dataframe
        batch_size : int
        distance_matrix : distance matrix as pandas dataframe
        epochs : int, optional. The default is 1

        Yields
        ------
        x_true : encoded seqeunce for minibatches in batch 
        y_true : pairwise distaces fro minibatches in batch 

        """
        import numpy as np 
        
        if not hasattr(self, 'encoded'):
            raise KeyError('no encoded seqs found. Please encode with qdata.encode before using batchmaker')
        
        for _ in range(epochs): 
        
            # reorder minibatches for this run 
            minibatches = minibatches.sample(frac=1)
            # get pairwise distances for all minibatches
            quartet_y_true = self.__get_y_true(minibatches, distance_matrix).values
            
            # make batches 
            batches_len = len(minibatches) - len(minibatches)%batch_size        
            for n in range(0, batches_len, batch_size):   
            
                y_true = list(quartet_y_true[n:n+batch_size]) 
                x_names = minibatches[n:n+batch_size].values
                x_true = [[self.loc[i]['encoded'] for i in j] for j in x_names]
                x_true = [[np.expand_dims(i, axis=0) for i in j] for j  in x_true]
                yield x_true, y_true
     


