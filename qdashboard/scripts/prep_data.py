#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .metadata_handling import metadata
import pandas as pd
import base64



def get_metadata(path):    
    meta = metadata.read(path).metadata
    meta = pd.DataFrame(meta)
    meta = meta.fillna('nan')
    meta['trainseq'] = [x[-2] for x in meta['seqs_file'].str.split('/')]
    return meta 


    

def get_path(meta, selected_date):
    meta = pd.DataFrame(meta)
    load_from, = meta[meta['date'] == selected_date].path.values
    return load_from




def load_json_as_table(meta_selected, file_to_load):
    data = pd.read_json(meta_selected['path'] + '/' + file_to_load)
    data['epoch'] = data.index
    data = pd.melt(data, id_vars='epoch')
    return data

       
            

def prep_b64_src(path):
    encoded_image = base64.b64encode(open(path, 'rb').read())
    new_image_src = 'data:image/png;base64,{}'.format(encoded_image.decode())
    return new_image_src

    
    












