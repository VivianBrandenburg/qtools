#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# use write-only mode for matplotlib to surpress UserWarning caused by logomaker: 
# UserWarning:Starting a Matplotlib GUI outside of the main thread will likely fail.
import matplotlib
matplotlib.use('agg')


from dash import  Input, Output, callback, State, html, ctx
import os 
from scripts.motif import motif
import matplotlib.pyplot as plt 
from scripts.prep_data import  prep_b64_src
import pandas as pd 


# =============================================================================
# helper functions to generate temporary motif pngs 
# =============================================================================

def collect_motif_images(motif_path):
    image_paths = sorted([motif_path+ x for x in os.listdir(motif_path) 
                          if x.startswith('motif') and x.endswith('.png')])
    image_paths.reverse()
    return image_paths

def make_motif_filename(motif_path, n_clicks):
    return f'{motif_path}motif.{n_clicks}.png'


def clear_motif_cache(motif_path):
    images = collect_motif_images(motif_path)
    for i in images: 
        os.remove(i)


def generate_motif_image(img):
    return html.Div([
            html.Img(
                src = img,
                style = {'height': '60pt','padding-top': '5pt', 'padding-right': 0},
                draggable='True'),
           
            ])

def prepare_spec_names(selected_data):
    specs = [('_').join(x.split('_')[-2:]) for x in
             selected_data['spec'].to_list()]
    spec_per_line = 2
    specs = [specs[x:x+spec_per_line] for x in range(0, len(specs),spec_per_line)]
    specs = [', '.join(x) for x in specs]
    specs = '\n'.join(specs)
    return specs

# =============================================================================
# default for 'no motif to show'
# =============================================================================

motif_button_default = html.Center([html.P('Select species and sequence from the data table'), html.P(' and hit submit to create a motif')], style={'padding':'10%', "color": "gray", "fontSize": "14px"})


# =============================================================================
# callbacks 
# =============================================================================


@callback(
    Output('motif-plot', 'children'),
    Input('button-create-motif', 'n_clicks'),
    Input('button-delete-motif', 'n_clicks'),
    State('datatable-seqs', 'data'),
    State('datatable-seqs', 'selected_rows'),
    State('datatable-seqs', 'selected_columns'),

    prevent_initial_call=False
)
def update_motif_plots(n_clicks_create, n_clicks_delete, data, selected_rows, selected_columns):
    """
    take selected sequences, pass them to create a motif. Plot the motif with matplotlib, print it as png as temporary file into motif_path. Show all temporary files in motf_path. 

    Parameters
    ----------
    motif_path : path to dir where to store temporary pngs of motifs 
    n_clicks : trigger this function 
    data : all sequences from selected date 
    selected_rows : taxa to select from data 
    selected_columns : sequence to select from data 

    Returns
    -------
    a list of all temp motif pngs 

    """
     
    motif_path = './'
    if "button-delete-motif" == ctx.triggered_id:
        clear_motif_cache(motif_path)
        return motif_button_default
    
    
    else: 
    # elif "button-create-motif" == ctx.triggered_id: 
        if selected_rows is None or selected_columns is None:
            return motif_button_default
        else:
            # prepare selected sequences 
            data = pd.DataFrame(data)
            selected_data = data.loc[sorted(selected_rows), :]
            seqs = selected_data[selected_columns[0]].to_list()
            
            
            # prepare spec names for plot title generation
            specs = prepare_spec_names(selected_data)
            
            # make a new motif 
            m = motif(seqs)
            logo = m.logo()
            
            # plot the motif and save it as temp png for the dashboard 
            motif_file = make_motif_filename(motif_path, n_clicks_create)
            plt.figtext(0.91, 0.2, f'{selected_columns[0]}\n{specs}', 
                        fontsize=8, fontfamily='monospace')
            plt.savefig(motif_file,  bbox_inches='tight')
            plt.close()
            
            # collect all motifs in current motif_path 
            images = collect_motif_images(motif_path)
            images = [prep_b64_src(x) for x in images]
            
            # create a list of all motifs 
            images_div = []
            for i in images:
                images_div.append(generate_motif_image(i))
            return images_div
        
