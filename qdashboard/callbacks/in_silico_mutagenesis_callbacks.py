# Helper for vertically centered, styled message in the mutagenesis logo container
def mutagenesis_message(msg):
    return html.Div(
        html.P(msg, style={"color": "gray", "fontSize": "14px", "margin": "0", "fontFamily": "inherit"}),
        style={"display": "flex", "flexDirection": "column", "justifyContent": "center", "height": "165px"}
    )
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from dash import Input, Output, State, callback, html
import dash

import numpy as np
import pandas as pd
import logomaker
import matplotlib.pyplot as plt
import base64
import io
from scripts.motif import motif

def find_mutagenesis_features(mutagenesis_path, epoch):
    """
    Returns a sorted list of feature numbers for files matching e{epoch}_f{feature}.npy in the given path.
    """
    if not os.path.isdir(mutagenesis_path):
        return []
    pattern = re.compile(rf"e{epoch}_f(\d+)\.npy")
    features = []
    for fname in os.listdir(mutagenesis_path):
        match = pattern.match(fname)
        if match:
            features.append(int(match.group(1)))
    return sorted(features)


def mutagenesis_exists(path):
    new_path = os.path.isdir(os.path.join(path, 'mutagenesis/'))
    return new_path


# Callback to update the feature dropdown if mutagenesis exists
@callback(
    Output('mutagenesis-row', 'style'),
    Output('mutagenesis-feature-dropdown', 'options'),
    Output('mutagenesis-feature-dropdown', 'value'),
    Input('meta_selected', 'data'),
    Input('epoch-selected', 'value'),
)
def update_mutagenesis_dropdown(meta_selected, selected_epoch):
    path = meta_selected['path']
    mutagenesis_path = os.path.join(path, 'mutagenesis/')
    if not os.path.isdir(mutagenesis_path):
        return {'display': 'none'}, [], None
    features = find_mutagenesis_features(mutagenesis_path, selected_epoch)
    if features:
        options = [{'label': f'Feature {f+1}', 'value': f} for f in features]
        value = features[0]
    else:
        options = [{'label': f'epoch {selected_epoch} has features', 'value': 'no_features', 'disabled': True}]
        value = 'no_features'
    return {'display': 'block'}, options, value

# Callback to plot the selected mutagenesis feature as a logo
@callback(
    Output('mutagenesis-logo-container', 'children'),
    Input('meta_selected', 'data'),
    Input('epoch-selected', 'value'),
    Input('mutagenesis-feature-dropdown', 'value'),
)
def plot_mutagenesis_logo(meta_selected, selected_epoch, selected_feature):
    if not meta_selected:
        return None
    path = meta_selected['path']
    mutagenesis_path = os.path.join(path, 'mutagenesis')
    # If no mutagenesis folder at all
    if not os.path.isdir(mutagenesis_path):
        msg = f"no in-silico mutagenesis found for this model at {path}"
        return mutagenesis_message(msg)
    # If no features for this epoch
    if selected_feature is None or selected_feature == 'no_features':
        msg = f"no in-silico mutagenesis motif found for epoch {selected_epoch}"
        return mutagenesis_message(msg)
    npy_path = os.path.join(mutagenesis_path, f"e{selected_epoch}_f{selected_feature}.npy")
    if not os.path.isfile(npy_path):
        msg = f"no in-silico mutagenesis motif found for epoch {selected_epoch}"
        return mutagenesis_message(msg)
    arr = np.load(npy_path)
    # Only handle (L, 4) arrays
    if arr.ndim != 2 or arr.shape[1] != 4:
        msg = f"no in-silico mutagenesis motif found for epoch {selected_epoch}. I found a npy file at {npy_path}, but it was broken."
        return mutagenesis_message(msg)
    # Use motif class for color scheme and nucleotide order
    m = motif(['A'*arr.shape[0]])  # dummy motif to access class vars
    def rgb_to_hex(rgb):
        return '#%02x%02x%02x' % tuple(int(255*x) for x in rgb)
    color_scheme = {}
    for k, v in m.colorscheme.items():
        if k == 'TU':
            color_scheme['T'] = rgb_to_hex(v)
            color_scheme['U'] = rgb_to_hex(v)
        else:
            color_scheme[k] = rgb_to_hex(v)
    # Map arr columns to motif.alphabet order, then relabel T->U for display
    # If arr columns are [T, C, G, A], and motif.alphabet is [A, C, G, T],
    # we need to find the mapping from motif.alphabet to arr columns
    arr_nts = ['C', 'G', 'T', 'A']  # assumed order in arr (update if needed)
    # Build index mapping: for each nt in motif.alphabet, find its index in arr_nts
    idx_map = []
    for nt in m.alphabet:
        if nt == 'U':
            idx_map.append(arr_nts.index('T'))
        else:
            idx_map.append(arr_nts.index(nt))
    arr_reordered = arr[:, idx_map]
    motif_nts = [nt if nt != 'U' else 'T' for nt in m.alphabet]
    df = pd.DataFrame(arr_reordered, columns=motif_nts)
    fig, ax = plt.subplots(figsize=(min(12, arr.shape[0] / 5), 2))
    logomaker.Logo(df, ax=ax, color_scheme=color_scheme)
    ax.set_ylabel('Score')
    ax.set_xlabel('Position')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return html.Img(src=f"data:image/png;base64,{img_base64}", style={"height": "165px"})
