#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from dash import Input, Output, callback, html
import json
import plotly.express as px
import numpy as np
import scripts.styles as c
from scripts.prep_data import load_json_as_table
from scripts.utils import order_by_species



# =============================================================================
# simple lineplots 
# =============================================================================

def plot_simple_lines(data):
    fig = px.line(data, x='epoch', y='value', color='variable', 
                  color_discrete_sequence=[c.blue, c.lightorange, c.orange])
    fig.update_traces(line={'width': 3})    
    fig.update_layout(margin=dict(l=0,r=0,b=0,t=0), legend_title=None,
                      font=dict(size=10),
                      legend=dict(orientation="h",  yanchor="bottom", y=1.02,
                          xanchor="right", bgcolor="rgba(0,0,0,0)", x=1))
    return fig




# plot quartet loss
@callback(
    Output('plot-losses', 'figure'),
    Output('epoch-selected', 'max'),
    Input('meta_selected', 'data'),
    Input('loss-toggle-logscale', 'value'))
def update_loss(meta_selected, yaxis_type):
    data = load_json_as_table(meta_selected,  'losses.json')
    fig = plot_simple_lines(data)
    fig.update_yaxes(title='',
                     type='linear' if yaxis_type == 'Linear' else 'log')
    return fig , data['epoch'].max()


# plot quartet scores
@callback(
    Output('plot-quartet-scores', 'figure'),
    Input('meta_selected', 'data'),
    Input('scores-toggle-logscale', 'value'))
def update_scores(meta_selected, yaxis_type):
    data = load_json_as_table(meta_selected,  'scores.json')
    fig = plot_simple_lines(data)
    fig.update_yaxes(title='', type='linear' if yaxis_type == 'Linear' else 'log')
    return fig 




# =============================================================================
# heatmaps 
# =============================================================================


def make_heatmap(data, labels):
    data[data == 0] = np.nan
    
    # reorder the dataframe
    data = data.set_index(labels['y_labels'], drop=True)
    data = order_by_species(data) 
    # change feature vector indices from 0...29 to 1...30
    data = data.rename(columns={x:x+1 for x in range(30)}) 
    
    # plot the heatmap
    fig = px.imshow(data, color_continuous_scale='darkmint', labels={'x':'feature vector position', 'color': 'expr.'})
    
    # make figure look nice
    fig.update_layout(margin=dict(l=10,r=10,b=10,t=10), yaxis_title=None, 
                      font=dict(size=10), height=230)
    fig.update_xaxes(tick0=1, dtick=2)
    fig.update_yaxes(ticklabelstep=1)
    return fig


# load the feature vectors 
def get_featureVectors(meta_selected, infile = '/featureVectors.json'):
    load_from = meta_selected['path']
    with open(load_from + infile , 'r') as inf:
        data = json.load(inf)
    data = {k: pd.DataFrame(v) for k, v in data.items()}
    labels = pd.read_json(load_from + infile.replace('featureVectors.json', 'y_labels.json'))
    return data, labels


# update heatmap plot when changing to new run / new epoch
@callback(
    Output("heatmap-figure", "figure"),
    Input('meta_selected', 'data'),
    Input("epoch-selected", "value"))
def filter_heatmap(meta_selected, selected_epoch):
    data, labels = get_featureVectors(meta_selected) # load feature vectors
    data = data[str(selected_epoch)]
    fig = make_heatmap(data, labels) # plot feature vectors
    return fig




# =============================================================================
# Callback to update the reference tree image based on model type
# =============================================================================
@callback(
    Output('reftree-img-container', 'children'),
    Input('schemes-radio', 'value'),
)
def update_reftree_img(schemes_value):
    print('schemes_value:', schemes_value)
    if schemes_value and 'randomized' in str(schemes_value).lower():
        src = 'assets/reftree_randomized.png'
        print('gotcha')
    else:
        print('not gotcha')
        src = 'assets/reftree.png'
    return html.Img(src=src, alt='image', style={'height': '180px', 'margin-top': '10px'})

