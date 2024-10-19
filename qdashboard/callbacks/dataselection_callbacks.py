#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from dash import html, Input, Output,  callback
import os


def get_available_options(meta, searched_options, **conditions):    
    meta =  pd.DataFrame(meta)
    all_options = list(meta[searched_options].unique())
    
    # sorting options for conistency
    options_str = sorted([x for x in all_options if type(x) == str])
    options_else = sorted([x for x in all_options if x not in options_str])
    all_options = options_else + options_str
        
    
    # check all conditions for available options 
    available = meta
    for cond_name, cond_value in conditions.items(): 
        available = available[available[cond_name] == cond_value]
    available = list(available[searched_options].unique()) 
    value = [x for x in all_options if x in available][0]
    labels = []
    
    for x in all_options: 
        if x in available: 
            labels.append({'label': html.Div(x, style={'display': 'inline-block'}),
                            'value':x})
        else: 
            labels.append({'label': html.Div(x, style={'color': 'lightgrey', 'display': 'inline-block'}),
                           'value':x, 'disabled': True})
    return labels, value
    


# make radio with all available sigma settings, select first in list
@callback(
    Output('sigma-radio', 'options'),
    Output('sigma-radio', 'value'),
    Input('meta', 'data'), 
    Input('schemes-radio', 'value'),
)
def set_sigma_radio(meta, selected_scheme):  
    options, value = get_available_options(meta, 'model_type', seq_type = selected_scheme)
    return options, value



# list available dates depending on radio selections
@callback(
    Output('dates-dropdown', 'options'),
    Input('meta', 'data'),
    Input('schemes-radio', 'value'),
    Input('sigma-radio', 'value'),
    )
def set_dates_options(meta, selected_scheme, selected_sigma):   
    meta = pd.DataFrame(meta)    
    available_data = meta[(meta['seq_type']==selected_scheme) &
                           (meta['model_type']==selected_sigma) 
                           ].sort_values(by='path')
    available_dates = available_data['date']
    labels = [x.split('/')[-1] for x in available_data['path']]
    return [{'label': i, 'value': j} for i, j in zip(labels, available_dates)]


# when changing list of available dates, then set first date as selected 
@callback(
    Output('dates-dropdown', 'value'),
    Output('epoch-selected', 'value'),
    Input('dates-dropdown', 'options'))
def set_dates_value(available_options):
    return available_options[0]['value'], 0


# set title for epoch selector including the number of trained epochs
@callback(
    Output("epoch-range-info", "children"),
    Input("epoch-selected", "min"),
    Input("epoch-selected", "max"),
)
def number_render(rangemin, rangemax):
    return "of {}".format(rangemax)


# when changing selected date, then update the selected metadata 
@callback(
    Output('meta_selected', 'data'),
    Input('meta', 'data'),
    Input('dates-dropdown', 'value'))
def set_selected_metadata(meta, selected_date):
    print(f'selected data: {selected_date}')
    meta = pd.DataFrame(meta)
    selected_meta = meta[meta['date'] == selected_date].iloc[0]
    return selected_meta.to_dict()


# get info for additional info box 
@callback(Output('info-box', 'children'),    
          Input('meta_selected', 'data'))
def update_info_box(selected_meta):
    infotext = selected_meta['path']+'/INFO.txt'
    if os.path.isfile(infotext):
        with open(infotext, 'r') as inf: 
            infotext = inf.read()
        return infotext
    return 'There is no additional information available for this model.'
    
