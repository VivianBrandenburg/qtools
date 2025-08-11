#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd 
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt
import json


def load_json_as_table(metadata_row, filename):
    """Load JSON file and convert to long format table."""
    filepath = metadata_row['path'] + '/' + filename
    data = pd.read_json(filepath)
    data['epoch'] = data.index
    melted_data = pd.melt(data, id_vars='epoch')
    return melted_data

def calculate_quartet_scores(scores_data, output_dims, model_name):
    """Calculate quartet scores from raw score data."""
    pivoted_scores = scores_data.pivot(index='epoch', columns='variable', values='value')
    
    quartet_scores = pd.DataFrame({
        'quartet_score': pivoted_scores['opt'] + pivoted_scores['subopt'], 
        'epoch': pivoted_scores.index
    })
    quartet_scores['output_dims'] = output_dims
    quartet_scores['model_name'] = model_name
    return quartet_scores

def calculate_losses(losses_data, output_dims, model_name):
    """Calculate losses from raw loss data."""
    pivoted_losses = losses_data.pivot(index='epoch', columns='variable', values='value')
    
    processed_losses = pd.DataFrame({
        'loss': pivoted_losses['loss'], 
        'epoch': pivoted_losses.index
    })
    processed_losses.dropna(subset=['loss'], inplace=True)
    processed_losses['output_dims'] = output_dims
    processed_losses['model_name'] = model_name
    return processed_losses

def transpose_feature_vectors(feature_data):
    """Transpose feature vector data structure."""
    transposed_features = {}
    for epoch, feature_rows in feature_data.items():
        if not feature_rows: 
            transposed_features[epoch] = []
            continue
        # Transpose rows to columns
        transposed = list(zip(*feature_rows))
        transposed_features[epoch] = [list(col) for col in transposed]
    return transposed_features

def is_feature_vector_non_empty(vector):
    """Check if a feature vector contains any non-zero values."""
    return len([x for x in vector if x != 0]) > 0

def calculate_non_zero_features(metadata_df):
    """Calculate non-zero feature statistics for all models."""
    feature_statistics = pd.DataFrame()
    
    for idx in metadata_df.index:
        model_metadata = metadata_df.iloc[idx]
        
        # Load and transpose feature vectors
        feature_vectors_path = model_metadata['path'] + '/featureVectors.json'
        with open(feature_vectors_path) as f:
            raw_vectors = json.load(f)
        
        transposed_vectors = transpose_feature_vectors(raw_vectors)
        
        # Calculate statistics for each epoch
        for epoch, feature_vectors in transposed_vectors.items():
            non_zero_count = len([v for v in feature_vectors if is_feature_vector_non_empty(v)])
            total_vectors = len(feature_vectors)
            non_zero_fraction = non_zero_count / total_vectors if total_vectors > 0 else 0
            
            epoch_stats = pd.DataFrame({
                'epoch': [int(epoch)], 
                'model_name': [metadata_df.loc[idx, 'out_dir']],
                'output_dims': [model_metadata['output_dims']],
                'non_zero_features': [non_zero_count],
                'non_zero_fraction': [non_zero_fraction]
            })
            feature_statistics = pd.concat([feature_statistics, epoch_stats], ignore_index=True)
    
    return feature_statistics

def create_lineplot(dataframe, y_column, model_type=None, title=None, ylabel=None,
                    plottype = ''): 
    """Create a line plot for the given data."""
    # Filter data (only include models with output_dims > 4)
    filtered_df = dataframe[dataframe['output_dims'] > 4]
    
    plt.figure(figsize=(9*0.3, 6*0.35))
    
    # Select color palette
    palette =  sns.color_palette('viridis', 4)
    
    # Create the line plot
    ax = sns.lineplot(
        data=filtered_df,
        x='epoch',
        y=y_column, 
        hue='output_dims',
        estimator=np.mean,    
        linewidth=2,
        palette=palette
    )
    
    # Set title and labels
    if title: 
        plot_title = f'{model_type}' 
        plt.title(plot_title)
    
    plt.xlabel('Epoch')
    plt.ylabel(ylabel if ylabel else y_column)
    
    # Customize legend
    legend = plt.legend(title='Output Dims')
    for line in legend.get_lines():
        line.set_linewidth(2) 
        
    # plt.ylim(0,50)
    
    plt.tight_layout()
    plt.show()


