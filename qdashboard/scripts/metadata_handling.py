#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json
import pandas as pd


class metadata():  


    def __init__(self, metadata={}):
        self.metadata = metadata
    
    
    
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






