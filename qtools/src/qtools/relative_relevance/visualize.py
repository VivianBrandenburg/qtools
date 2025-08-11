import numpy as np
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns


def plot_delta_relevance_boxplot(df_all, merged_generative, merged_random):
    """
    Plot boxplot of delta relevance and annotate significance.
    Args:
        df_all (pd.DataFrame): Data for plotting.
        merged_generative, merged_random (list of lists): Effect sizes per epoch.
        outdir (str): Output directory for saving plot.
        used_significance_levels (set): Set to track used significance levels.
    """

    plt.figure(figsize=(16*0.42, 6*0.6))
    ax = sns.boxplot(data=df_all, x='Epoch', y='EffectSize', hue='Condition', palette='Set2')
    x_positions = [i for i, _ in enumerate(ax.get_xticklabels())]
    colors = {'p < 0.001': '#F52500', 'p < 0.01': '#F5573B', 'p < 0.05': '#F58773'}
    used_significance_levels = set()
    
    # Calculate p-values for every fifth epoch
    num_epochs = len(merged_generative)
    p_vals = []
    epoch_indices = []
    for i in range(0, num_epochs, 5):
        vals_n = merged_generative[i]
        vals_s = merged_random[i]
        if vals_n and vals_s:
            _, p = mannwhitneyu(vals_n, vals_s, alternative='two-sided')
        else:
            p = 1
        p_vals.append(p)
        epoch_indices.append(i)
        
    # Annotate significance
    for j, (i, p) in enumerate(zip(epoch_indices, p_vals)):
        if p < 0.05:
            vals_n = merged_generative[i]
            vals_s = merged_random[i]
            y_max = max(max(vals_n, default=0), max(vals_s, default=0))
            if p < 0.001:
                symbol, level = 'D', 'p < 0.001'
            elif p < 0.01:
                symbol, level = '^', 'p < 0.01'
            else:
                symbol, level = 'o', 'p < 0.05'
            used_significance_levels.add(level)
            if j < len(x_positions):
                plt.scatter(x_positions[j], y_max + 0.005, marker=symbol, color=colors[level], s=50, zorder=10)
    plt.ylabel('Δ Rel Nt Relevance (Seg. 2 − Seg. 3)')
    plt.xlabel('Epoch')
    plt.grid(True)
    condition_legend = plt.legend(title='Reference Tree', loc='lower left', bbox_to_anchor=(0, 0))
    plt.gca().add_artist(condition_legend)
    if used_significance_levels:
        significance_order = [
            ('p < 0.05', 'o'),
            ('p < 0.01', '^'),
            ('p < 0.001', 'D')
        ]
        significance_handles = []
        for label, symbol in significance_order:
            if label in used_significance_levels:
                handle = plt.Line2D([0], [0], marker=symbol, color='w', markerfacecolor=colors[label],
                                   markersize=8, label=label, linestyle='None')
                significance_handles.append(handle)
        plt.legend(handles=significance_handles, title='Significance', loc='lower left',
                  bbox_to_anchor=(0.25, 0))
    plt.tight_layout()
    plt.show()




def plot_segments(data):
    avg_0_8 = np.nanmean(data[:, 0:8])
    avg_8_28 = np.nanmean(data[:, 8:28])
    avg_28_35 = np.nanmean(data[:, 28:-1])
    line1 = plt.hlines(avg_0_8, 0, 8, colors='blue', linestyles='--', label=f'seg 1: {avg_0_8:.3f}', lw=2)
    line2 = plt.hlines(avg_8_28, 8, 30, colors='red', linestyles='--', label=f'seg 2: {avg_8_28:.3f}', lw=2)
    line3 = plt.hlines(avg_28_35, 30, 35, colors='green', linestyles='--', label=f'seg 3: {avg_28_35:.3f}', lw=2)
    plt.legend(handles=[line1, line2, line3], loc='upper right')
    

def test_segments(data, title):
    segment2 = data[:, 8:28]
    segment3 = data[:, 28:-1]
    valid_rows = ~np.isnan(segment2).any(axis=1) & ~np.isnan(segment3).any(axis=1)
    seg2_clean = segment2[valid_rows]
    seg3_clean = segment3[valid_rows]
    mean_seg2 = np.nanmean(seg2_clean, axis=1)
    mean_seg3 = np.nanmean(seg3_clean, axis=1)
    t_stat, p_val = ttest_rel(mean_seg2, mean_seg3)
    w_stat, w_p_val = wilcoxon(mean_seg2, mean_seg3)
    print(f"{title} [{len(valid_rows)} epochs] | Paired t: t = {t_stat:.3f}, p = {p_val:.4f} | Wilcoxon: W = {w_stat:.3f}, p = {w_p_val:.4f}")

  



def plot_relative_relevance(all_attribution, title):
    """
    Plot attribution data
    """
    data = np.array(all_attribution)
    num_lines = data.shape[0]
    colors = cm.viridis(np.linspace(0, 1, num_lines))
    plt.figure(figsize=(5, 2))
    for i in range(num_lines):
        plt.plot(data[i], color=colors[i], alpha=0.8)
    
    norm = mcolors.Normalize(vmin=0, vmax=num_lines)
    sm = cm.ScalarMappable(cmap='viridis', norm=norm)
    cbar = plt.colorbar(sm, label='Epoch')
    cbar.set_ticks([0, num_lines // 2, num_lines - 1])
    cbar.set_ticklabels([1, num_lines // 2 + 1, num_lines])
    
    plot_segments(data)
    test_segments(data, title)
    
    plt.xlabel('nt')
    plt.ylabel('relative nt relevance')
    plt.title(title)
    plt.grid(True)
    plt.show()
    




