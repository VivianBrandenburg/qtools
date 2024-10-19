#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json
import pandas as pd


class metadata():  

    collected_keys = [
        'epochs', 'learning_rate', 'batch_size', 'date',
                  'model_type', 'sigma',
                   'seq_type', 'mutation_scheme', 'seq_len', 'seqs_file', 'out_dir']
        

    collected_automatic = ['epochs', 'learning_rate', 'batch_size',  'sigma',
                   'seq_len', 'seqs_file', 'out_dir', 'mutation_scheme']
    
    def __init__(self, metadata={}):
        self.metadata = metadata
    
    
    @classmethod
    def record(cls, local_input):
        """
        Automatically records metadata. Returns metadata object.
        See metadata.collected_keys for a list of collected metadata and 
        metadata.metadata for the collected records
        See metadata.collected_from_locals for a list of variables that are 
        collected from locals()
        
        Usage:
            myrecords = metadata.record(locals()).write()
        
        This function assumes you have defined local variables: 
            
            epochs : int 
                tensorflow variable
            learning_rate : float
                tensorflow variable
            batch_size : int
                tensorflow variable
            sigma : float 
                used for quartet+siamese loss 
            seq_len : int 
                length of the input sequences
            seqs_file : str 
                file with input data
            out_dir : str 
                directory your output is written to 
            mutation_scheme : str 
                indicator on which simulated seqeuence was used for training           
            
        """
        collected = local_input #.items()
        
        # update model type
        if 'sigma' in collected.keys(): 
            if collected['sigma'] == 'nan':  collected['model_type'] = 'siamese'
            elif collected['sigma'] == 0 : collected['model_type'] = 'quartet'
            elif type(collected['sigma'] ) == float: collected['model_type'] = 'q+s'
            else: collected['model_type'] = 'unknown'
        else: 
            collected['sigma'] = collected['model_type']  = 'nan'
            
        
        # update date
        if 'out_dir' in collected.keys(): 
            collected['date'] = collected['out_dir'].strip('/').split('/')[-1]
        else: 
            collected['date'] = collected['out_dir']  = 'nan'

                
        # update seq_type with known mutation schemes
        if 'mutation_scheme' in collected.keys(): 
            if collected['mutation_scheme'] == 'n350': collected['seq_type'] = 'rRNA'
            elif collected['mutation_scheme'] == 's8_n20m3_r7': collected['seq_type'] = 'simulated'
            else:
                collected['mutation_scheme'] = collected['seq_type']  = 'nan'
        else: 
            collected['mutation_scheme'] = collected['seq_type']  = 'nan'
            
        
        metadata = {k:collected[k] for k in cls.collected_keys }
        print(metadata)
        
        
        return cls( metadata)

    
    def write(self, outdir=None):
        """ Writes metadata to a 'meta.json'. If not filename was provided metadata are written to 'out_dir' variable.        Returns : None
        """
        metadata = self.metadata
        if 'out_dir' in metadata.keys():
            date = metadata['out_dir'].strip('/').split('/')[-1]
            metadata.update({'date': date})
        if outdir:     
            print(f'writing metadata to {outdir}')
        else:
            assert os.path.exists(metadata['out_dir']), "Did not find an out_dir in your local variables. Please add 'out_dir=path/to/outfile/' to your local variables or provide a path to this function"
            outdir = metadata['out_dir']
        with open(outdir + '/meta.json', 'w') as outf:
            json.dump(metadata, outf)
            
            
    
    @staticmethod
    def __collect_metafiles(path, pattern):
        for dirpath, dirnames, filenames in os.walk(path):
            files = [f for f in filenames if f.find(pattern) != -1]
            for f in files: 
                with open(os.path.join(dirpath, f), 'r') as inf:
                    d = json.load(inf)
                    d.update({'path':dirpath})
                    yield d
    
             
    @classmethod
    def read(cls, path=os.getcwd(), pattern='meta.json'):
        """
        Automatically read all 'meta.json' files from the current dir and subdirs and merge them to one dictionary. 

        Parameters
        ----------
        path : str, optional
            The default is the current working directory.
        pattern : str, optional
            Pattern to select metadata files. The default is 'meta.json'.\

        Returns
        -------
        metadata object with collected data in metadata.metadata 

        """
        collected_files = [x for x in cls.__collect_metafiles(path, pattern)]
        collected_data = [pd.DataFrame(x, index=[0]) for x in collected_files]
        data_merged = pd.concat(collected_data, ignore_index=True)
        
        data_merged_asDict = data_merged.to_dict(orient='list') # If the format of the resulting dict is not convinient, you might want to change 'orient' here
        return cls(data_merged_asDict)




class mutationscheme():
    
    def __init__(self, infos):
        self.scheme = infos['scheme']
        self.names = infos['names']
        self.seqlength = self.__get_seqlength()
    
    
    @classmethod
    def from_json(cls, file):
        with open(file,'r') as inf: 
            inf = json.load(inf)
        return cls(inf)
    
    
    def __get_seqlength(self):
        parts  = self.scheme.split('_')
        parts = [x.split('m')[0] for x in parts]
        part_lengths = [int(x[1:]) for x in parts]
        seqlength = sum(part_lengths)
        return seqlength
        
    
    


def get_metadata(path):    
    meta = metadata.read(path).metadata
    meta = pd.DataFrame(meta)
    meta = meta.fillna('nan')
    meta['trainseq'] = [x[-2] for x in meta['seqs_file'].str.split('/')]
    return meta 
