import seaborn as sns 
import pandas as pd
from qtools.data_tracking import metadata

from qtools.variable_output_dims import (
    create_lineplot,
    calculate_quartet_scores,
    calculate_losses,
    calculate_non_zero_features,
    load_json_as_table
)

# Configuration
MODEL_TYPES = ['siameseloss', 'quartetloss', 'qsloss']
CURRENT_MODEL_TYPE = MODEL_TYPES[0]  


metadata_path = f'/trained_models/{CURRENT_MODEL_TYPE}/'
model_metadata = pd.DataFrame(metadata.read(metadata_path).metadata)
model_metadata['output_dims'] = model_metadata['output_dims'].astype(int)

# Process scores and losses for all models
combined_metrics = pd.DataFrame()

for idx in model_metadata.index: 
    # Load and calculate scores
    scores_data = load_json_as_table(model_metadata.loc[idx], 'scores.json')
    quartet_scores = calculate_quartet_scores(
        scores_data,
        output_dims=model_metadata.loc[idx, 'output_dims'],
        model_name=model_metadata.loc[idx, 'out_dir']
    ).reset_index(drop=True)
    
    # Load and calculate losses
    losses_data = load_json_as_table(model_metadata.loc[idx], 'losses.json')
    processed_losses = calculate_losses(
        losses_data,
        output_dims=model_metadata.loc[idx, 'output_dims'], 
        model_name=model_metadata.loc[idx, 'out_dir']
    ).reset_index(drop=True)
 
    # Merge scores and losses
    merged_metrics = pd.merge(
        processed_losses, quartet_scores, 
        on=['model_name', 'epoch', 'output_dims'], 
        how='outer'
    )
    
    combined_metrics = pd.concat([combined_metrics, merged_metrics]).reset_index(drop=True)

# Calculate feature vector statistics
feature_statistics = calculate_non_zero_features(model_metadata)

# Create visualizations
print("Creating quartet score visualization...")
valid_metrics = combined_metrics[~combined_metrics.loss.isna()]
create_lineplot(
    valid_metrics, 
    'quartet_score',
    model_type=CURRENT_MODEL_TYPE,
    title='Quartet Score Over Epochs by Output Dims', 
    ylabel='Quartet Score',
    plottype='quartetscore'
)

# Merge metrics with feature statistics
complete_data = pd.merge(
    combined_metrics, feature_statistics,
    on=['model_name', 'epoch', 'output_dims'], 
    how='outer'
)

print("Creating non-zero features visualization...")
valid_complete_data = complete_data[~complete_data.loss.isna()]
create_lineplot(
    valid_complete_data, 
    'non_zero_features',
    model_type=CURRENT_MODEL_TYPE,
    title='Number of Non-Empty Features Over Epochs by Output Dims',
    ylabel='# Non-Empty Featurs',
    plottype='emptyfeatures'
)
