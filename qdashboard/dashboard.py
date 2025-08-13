#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dash_bootstrap_templates import load_figure_template
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, dash_table

# =============================================================================
# import callbacks and components
# I tranfered some code to modules to unclutter and keep it maintainable. 
# =============================================================================


# The callbacks and their helper functions are in the callback module 
from callbacks import in_silico_mutagenesis_callbacks
from callbacks import dataselection_callbacks
from callbacks import table_callbacks
from callbacks import motifcreation_callbacks
from callbacks import plotting_callbacks
from callbacks import splitstree_callbacks

# some other helper functions are in the scripts module
from scripts import prep_data, utils

# load custom color scheme
import scripts.styles as c

# =============================================================================
# prepare path variables and metadata
# =============================================================================

# path for data for selectable table and the motif generation  
# must fit to 'seq_file' in metadata so that 
# traindatasource + seqs_file is the path from here to the training sequences
traindatasource =  '../' # keep '../' if you want to run the app from the qdashboard directory

# path from here to trained models 
# all metadata files from this path (including subdirectories) will be collected 
resultsdatasource = '/home/viv/Documents/GitHub/projects/qtools_reste/models_review'


# =============================================================================
# collect metadata 
# =============================================================================

# collect all metadata from resultsdatasource and prepare them.
# Most data selections use the 'date' from metadata as selector
meta = prep_data.get_metadata(resultsdatasource)


# =============================================================================
# app
# =============================================================================

# styling
load_figure_template("cerulean")
app = Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])
app.title='Quartet Dashboard'

app.layout = html.Div([

# =============================================================================
# store metadata and paths 
# =============================================================================
dcc.Store(id='meta', data=meta.to_dict("records")),
dcc.Store(id='meta_selected'),
dcc.Store(id='traindatasource', data=traindatasource),

# =============================================================================
# layout head 
# =============================================================================
  
# title
html.H1(children='siamese & quartet training',
        className="app-header--title", style={'textAlign': 'center', 'color':c.blue}),
html.Hr(),

# =============================================================================
# container 1 - select a dataset and an epoch.
#               plot loss, scores, splitstree and feature vectors for selection 
# =============================================================================

html.Div([

    # select dataset (input sequence, sigma, run) and  epoche 
    # =============
    # callbacks are in callbacks.dataselection_callbacks
    html.Div([ 
        html.H5('Data'),

        # select input sequence  
        dcc.RadioItems(
            meta['seq_type'].unique(),
            meta.loc[0]['seq_type'],
            id='schemes-radio', 
            className='selectionitem',
            inputStyle={"margin-left": "5px",   "margin-right": "2px",}
        ),

        # select sigma
        html.Label('Model Type'), 
        dcc.RadioItems(
            id='sigma-radio', 
            className='selectionitem',
            inputStyle={"margin-left": "5px",   "margin-right": "2px",}
        ),
        
        # select output_dims
        html.Label('Output Dimensions'),
        dcc.Dropdown(
            id='outputdims-dropdown',
            clearable=False,
            style={'height': '30px', 'width': '80px', 'margin-bottom': '10px'},
            className='selectionitem'
        ),
            
        # select date(run)
        html.Label('Run'),
        dcc.Dropdown(id='dates-dropdown', clearable=False, 
                     style={'height': '30px', 'width':
                            '60px', 'margin-bottom': '10px'},
                     className='selectionitem'
        ), 

        # select epoch
        html.Label('Epoch'),
        html.Div([dcc.Input(
            id="epoch-selected", type="number", 
            min=0, max=20, value=0, 
            style={'height': '30px', 'width': '60px'}, 
            className='selectionitem'
        ),  
        html.P(id="epoch-range-info", style={'display':'inline'}) ]),
        
        
    ], className='selectionzone'), 


    # show info text if available
    # =============
    # callback is in callbacks.dataselection_callbacks
    html.Div([
        html.H5('Info'),
        dcc.Markdown(id="info-box", style={'font-size': '9pt'}),
    ]),
                  
 
    # line plots 
    # =============
    # callbacks are in callbacks.plotting_callbacks          
    
    # plot loss
    html.Div([
        html.H5('Losses'),
        
        dcc.Graph(id='plot-losses', className='smallfig', 
                  config={'toImageButtonOptions': {'format':'svg',
                                                   'filename': 'losses'}}), 
        utils.LinearLogRadio(id_str='loss-toggle-logscale'),
    ], className='logtoggle'),
    
    # plot score
    html.Div([
        html.H5('Scores'),
        dcc.Graph(id='plot-quartet-scores', className='smallfig', 
                  config={'toImageButtonOptions': {'format':'svg',
                                                   'filename': 'losses'}}),
        utils.LinearLogRadio(id_str='scores-toggle-logscale'),
    ]),
    
    
    # splitstree
    # =============
    # callbacks are in callbacks.splitstree_callbacks
    html.Div([
        html.H5('Splitstree'),
            html.Img(id='splitstree-figure', style={'width':'400px',
                                                    'max-height':'500px',
                                                    'float':'left'})
    ], className='splitstreeimg'),
    
    
    
        
    # heatmap 
    # =============
    # callbacks are in callbacks.plotting_callbacks    
    html.Div([
            
    html.H5('Ref. Tree'),
    html.Img(src=r'assets/reftree.png', alt='image', style={'height':'180px', 'margin-top':'10px'})
        ], className='reftree'),
    

    html.Div([
        html.H5('Feature Vectors'), 
        dcc.Graph(id="heatmap-figure",
                  config={'toImageButtonOptions': {'format':'svg',
                                                   'filename': 'losses'}}), 
     ], className='featurevectors'),

    # Mutagenesis feature selection in its own row

    html.Div([
        html.Div([
            html.H5('Mutagenesis Feature'),
            dcc.Dropdown(id='mutagenesis-feature-dropdown', clearable=False, className='mutagenesis-dropdown', style={'width': '120px'}),
        ], style={'display': 'inline-block', 'verticalAlign': 'top', 'width': '160px'}),
        html.Div([
            html.H5('Motif'),
            html.Img(id='mutagenesis-motif-logo', style={'height': '100px', 'margin-left': '20px'})
        ], style={'display': 'inline-block', 'verticalAlign': 'top', 'margin-left': '30px'})
    ], id='mutagenesis-row', className='mutagenesisrow', style={'display': 'none', 'margin-top': '10px'}),
    
    

], className='grid-container'),
    

    
# =============================================================================
#  container 2 - select sequences and plot a motif from these sequences 
# =============================================================================
    
html.Div([

    # table with selectable taxa and sequences  
    # =============
    # callbacks are in callbacks.table_callbacks 
    html.Div([
        html.H5('Training Sequences'),
        dash_table.DataTable(
            id='datatable-seqs',
            editable=True,
            column_selectable="single",
            row_selectable="multi",
            style_cell = {  'font-family': 'Courier New', 
                          'font-size': '12px', 'padding-left': '10px',
                          'overflow': 'hidden',
                          'textOverflow': 'ellipsis',
                          'maxWidth': 150
                          },
            style_cell_conditional=[
                {'if': {'column_id': 'spec'},'width': '130px'},],
   
            css=[{'selector': '.dash-spreadsheet tr', 'rule': 'height: 1px;'}],
        )
    ], className='seqtable'),
    
    
    
    # buttons and plotting area 
    # =============  
    # callbacks are in callbacks.motifcreation_callbacks
    html.Div([ 
        
        html.H5('Alignment Motifs'),
        
        # create a motif from the sequences you selected in datatable-seqs. 
        # the motif will be added to the motifs displayed in  motif-plots
        html.Button('Create Motif', id='button-create-motif', n_clicks=0,
                    className='button', 
                    style={'background-color':c.lightorange, 
                           'border-color':c.lightorange}),
        
        # delete all created motifs 
        html.Button('Delete Motifs', id='button-delete-motif', n_clicks=0,
                    className='button',
                    style={'background-color':c.lightorange, 
                           'border-color':c.lightorange}),
        
        # display area for created motifs 
        html.Div(id='motif-plot', 
            style={'padding-top':'20pt'})

    ], className='showmotif'),


], className='grid-container-2'),


])




# =============================================================================
# run app 
# =============================================================================
if __name__ == '__main__':
    app.run(debug=True)
