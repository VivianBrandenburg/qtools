#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from dash import Input, Output, State, callback
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
        options = [{'label': f'Feature {f}', 'value': f} for f in features]
        value = features[0]
    else:
        options = [{'label': f'epoch {selected_epoch} has features', 'value': 'no_features', 'disabled': True}]
        value = 'no_features'
    return {'display': 'block'}, options, value

# Callback to plot the selected mutagenesis feature as a logo
@callback(
    Output('mutagenesis-logo', 'src'),
    Input('meta_selected', 'data'),
    Input('epoch-selected', 'value'),
    Input('mutagenesis-feature-dropdown', 'value'),
)
def plot_mutagenesis_logo(meta_selected, selected_epoch, selected_feature):
    if not meta_selected or selected_feature is None or selected_feature == 'no_features':
        return None
    path = meta_selected['path']
    npy_path = os.path.join(path, 'mutagenesis', f"e{selected_epoch}_f{selected_feature}.npy")
    if not os.path.isfile(npy_path):
        return None
    arr = np.load(npy_path)
    # Only handle (L, 4) arrays
    if arr.ndim != 2 or arr.shape[1] != 4:
        return None
    # Use motif class for color scheme and alphabet
    m = motif(['A'*arr.shape[0]])  # dummy motif to access class vars
    # Convert color scheme to logomaker format (dict of letter to RGB tuple)
    def rgb_to_hex(rgb):
        return '#%02x%02x%02x' % tuple(int(255*x) for x in rgb)
    color_scheme = {}
    for k, v in m.colorscheme.items():
        if k == 'TU':
            color_scheme['T'] = rgb_to_hex(v)
            color_scheme['U'] = rgb_to_hex(v)
        else:
            color_scheme[k] = rgb_to_hex(v)
    # Use the motif alphabet, but replace T with U if needed
    columns = [nt if nt in ['A', 'C', 'G', 'U'] else 'U' for nt in m.alphabet]
    df = pd.DataFrame(arr, columns=columns)
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
    return f"data:image/png;base64,{img_base64}"
