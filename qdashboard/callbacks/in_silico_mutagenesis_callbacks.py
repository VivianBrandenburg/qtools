#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from dash import Input, Output, State, callback
import dash

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
