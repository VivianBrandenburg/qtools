from qtools.relative_relevance.utils import get_non_zero_features_and_positions, find_folders_with_subfolder, is_larger_subtree, merge_runs_by_epoch, create_df_every_fifth
from qtools.relative_relevance.visualize import plot_relative_relevance, plot_delta_relevance_boxplot

import numpy as np
from scipy.stats import mannwhitneyu
import pandas as pd


model_paths = {'trained_models/qsloss_randomTree/outnodes_16':[], 
               'trained_models/qsloss/outnodes_16': []}



for model_path, cummulated_difference in model_paths.items(): 
    
    basepaths = find_folders_with_subfolder(model_path, 'mutagenesis')
    treetype = 'generative' if model_path.split('/')[-3].find('random') == -1 else 'randomized'
    
    for basepath in basepaths: 
        cummulated_difference += [[]]
            
        all_attribution = []
        
        # get non-empty input features 
        featureVectors_nonzero = get_non_zero_features_and_positions(basepath + '/featureVectors.json')
        del featureVectors_nonzero[0]
        
        # find setting for subtree selection 
        subtree_groups = 'generative' if basepath.find('random') == -1 else 'random'
        
        
        # get motif dist
        for epoch, features in featureVectors_nonzero.items(): 
            cummulated_difference[-1] += [[]]
            attribution_epoch = []
            if features: 
                for feature, positions in features.items():
                    
                    if is_larger_subtree(positions, subtree_groups=subtree_groups): 
                        motif = np.load(basepath + f'/mutagenesis/e{epoch}_f{feature}.npy')
                        positional_attribution = np.sum(np.abs(motif), axis=1)
                        normalized_attribution = positional_attribution / np.sum(positional_attribution)
                        attribution_epoch.append(normalized_attribution)
                        
                        slice1 = np.mean(normalized_attribution[8:28])
                        slice2 = np.mean(normalized_attribution[28:])
                        diff = slice1-slice2 
                        cummulated_difference[-1][-1].append(diff)
            if attribution_epoch: 
                all_attribution.append(np.nanmean(np.array(attribution_epoch), axis=0))
            else:
                all_attribution.append(np.full((35,), np.nan))
                        
        # plotting 
        plot_relative_relevance(all_attribution, f'{treetype} tree')

merged_generative, merged_random = merge_runs_by_epoch(model_paths, num_epochs=100)



# print stats for each epoch
for i in range(100):
    left = merged_generative[i]
    right = merged_random[i]
    if left and right:
        u_stat, p_val = mannwhitneyu(left, right, alternative='two-sided')
    else:
        u_stat = p_val = 1
    print(f"epoch {i+1} | generative {np.mean(left):.4f}, random {np.mean(right):.4f} | U = {u_stat}, p = {p_val:.4f}")


# make boxplots for every fifth epoch
df_generative = create_df_every_fifth(merged_generative, 'generative')
df_random = create_df_every_fifth(merged_random, 'random')
df_all = pd.concat([df_generative, df_random])

plot_delta_relevance_boxplot(df_all, merged_generative, merged_random)



