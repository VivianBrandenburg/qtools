#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from dash import  Input, Output, callback
import pandas as pd 
from scripts.utils import order_by_species


# =============================================================================
# callback for creating and 
# =============================================================================



@callback(
    Output('datatable-seqs', 'data'),
    Output('datatable-seqs', 'columns'),
    Input('meta_selected', 'data'),
    Input('traindatasource', 'data'))
def update_table(meta_selected, traindatasource):
    """
    Make a overview table to select sequences that will be used for motif creation.

    Parameters
    ----------
    meta : assembled meta files from results directory, stored in dcc.Store
    traindatasource : path to training data, stored in dcc.Store
    selected_date : date for which data should be shown, selected in selectionzone

    Returns
    -------
    df : data to be shown
    df_col : columns to be shown
    """
    
    # read in data table
    trainseq = pd.read_csv(
        traindatasource + '/' + meta_selected['seqs_file'])
    
    # select columns for display 
    controlseq_cols = [x for x in trainseq.columns if not x.startswith('control_') and x not in  ['seq', 'spec']]
    selected_cols = ['spec', 'seq'] +   controlseq_cols 
    trainseq = trainseq[selected_cols]
    
    # rename columns for display 
    rename_seqnames = {x:x.replace('control_','') for x in controlseq_cols}
    trainseq = trainseq.rename(columns=rename_seqnames)
    
    
    # reorder rows 
    trainseq = order_by_species(trainseq.set_index(trainseq['spec'], drop=True))
    
    # prepare data for sequences table 
    df =  trainseq.to_dict('records')
    
    # format columns for selectable table 
    df_col = [ {"name": i, "id": i,  "selectable": True} if i != 'spec' 
              else {"name": i, "id": i}
              for i in trainseq.columns ]
    

    return df, df_col


