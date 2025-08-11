
import json
import pandas as pd
import numpy as np
import qtools as qt
from qtools.encoding import onehot_encoding



def create_df_every_fifth(data, label):
    """
    Build a DataFrame with every fifth epoch's data for plotting.
    Args:
        data (list of lists): Effect sizes per epoch.
        label (str): Condition label.
    Returns:
        pd.DataFrame
    """
    df = []
    for epoch_idx, values in enumerate(data):
        if epoch_idx % 5 == 0:
            for v in values:
                df.append({'Epoch': epoch_idx + 1, 'EffectSize': v, 'Condition': label})
    return pd.DataFrame(df)

def merge_runs_by_epoch(model_paths, num_epochs=100):
    """
    Merge and combine results from multiple runs by epoch.
    Args:
        model_paths (dict): Model path to data mapping.
        num_epochs (int): Number of epochs to merge.
    Returns:
        merged_generative, merged_random (list of lists)
    """
    merged_generative, merged_random = [], []
    for (path, data), res in zip(model_paths.items(), [merged_random, merged_generative]):
        for i in range(num_epochs):
            combined = []
            for run in data:
                if len(run) > i:
                    combined.extend(run[i])
            res.append(combined)
    return merged_generative, merged_random

def calculate_p_values(merged_generative, merged_random, num_epochs=100):
    """
    Calculate p-values for each epoch using Mann-Whitney U test.
    Args:
        merged_generative, merged_random (list of lists): Effect sizes per epoch.
        num_epochs (int): Number of epochs.
    Returns:
        p_vals (list), stats (list)
    """
    from scipy.stats import mannwhitneyu
    p_vals = []
    stats = []
    for i in range(num_epochs):
        left = merged_generative[i]
        right = merged_random[i]
        if left and right:
            u_stat, p_val = mannwhitneyu(left, right, alternative='two-sided')
        else:
            u_stat = p_val = 1
        p_vals.append(p_val)
        stats.append(u_stat)
    return p_vals, stats


def is_larger_subtree(index_list, subtree_groups='normal'):
    generative_groups = {'A': list(range(6)), 'B': list(range(7,13))}
    random_groups = {'A': [5, 8, 10, 2, 0, 12], 'B': [4, 7, 1, 9, 3, 11]}
    if subtree_groups == 'generative':
        groups_selected = generative_groups
    elif subtree_groups == 'random':
        groups_selected = random_groups
    else:
        # fallback for legacy or typo
        groups_selected = generative_groups
    input_set = set([i for i in index_list if i != 6])
    for group in groups_selected.values():
        if input_set == set(group):
            return True
    return False

def find_folders_with_subfolder(root_dir, target_subfolder):
    import os
    matching_folders = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        if target_subfolder in dirnames:
            matching_folders.append(dirpath)
    return matching_folders

def get_non_zero_features_by_epoch(feature_vectors_path):
    """
    Read featureVectors from path and return indices of non-zero features for each epoch.
    Returns dict: {epoch: [list of indices of non-zero features]} for each epoch
    """
    with open(feature_vectors_path) as f:
        raw_vectors = json.load(f)
    non_zero_indices_by_epoch = {}
    transposed_vectors = {}
    for epoch, feature_rows in raw_vectors.items():
        if not feature_rows:
            non_zero_indices_by_epoch[int(epoch)] = []
            continue
        transposed = list(zip(*feature_rows))
        transposed_vectors[epoch] = transposed
        feature_vectors = [list(col) for col in transposed]
        non_zero_indices = [i for i, feature_vector in enumerate(feature_vectors) if any(x != 0 for x in feature_vector)]
        non_zero_indices_by_epoch[int(epoch)] = non_zero_indices
    return transposed_vectors, non_zero_indices_by_epoch

def get_data(seqs_file):
    data = pd.read_csv(seqs_file)
    data = qt.qdata(data)
    data.encode(onehot_encoding)
    x_encoded, x_species = data.get_data()
    return x_encoded, x_species

def get_background_dist(encoded_seqs, pseudocount):
    encoded_seqs = np.array(encoded_seqs) + pseudocount
    position_counts = np.sum(encoded_seqs, axis=0)
    background_dist = position_counts / position_counts.sum(axis=1, keepdims=True)
    return background_dist

def get_non_zero_features_and_positions(feature_vectors_path):
    """
    Read featureVectors from path and return positions of non-zero values for each feature in each epoch.
    Returns dict: {epoch: {feature_idx: [positions with non-zero values]}}
    """
    with open(feature_vectors_path) as f:
        raw_vectors = json.load(f)
    non_zero_positions_by_epoch = {}
    for epoch, feature_rows in raw_vectors.items():
        if not feature_rows:
            non_zero_positions_by_epoch[int(epoch)] = {}
            continue
        transposed = list(zip(*feature_rows))
        feature_vectors = [list(col) for col in transposed]
        feature_positions = {}
        for feature_idx, feature_vector in enumerate(feature_vectors):
            nonzero_positions = [pos for pos, val in enumerate(feature_vector) if val != 0]
            if nonzero_positions:
                feature_positions[feature_idx] = nonzero_positions
        non_zero_positions_by_epoch[int(epoch)] = feature_positions
    return non_zero_positions_by_epoch
