
"""
ASD/ADHD/AD Benchmark Plotting Utilities (with mean ± std)
----------------------------------------------------------
Author: (you)
Requirements: pandas, numpy, matplotlib

Data format (LONG/TIDY):
    Columns: ['dataset', 'atlas', 'method', 'metric', 'value'] (+ optional 'seed')
    - Multiple rows per (dataset, atlas, method, metric) are allowed (e.g., different random seeds).
    - The functions below aggregate internally: mean and std over repeated rows.

Design choices:
- Uses matplotlib only (no seaborn).
- One chart per figure (no subplots in a single call).
- No explicit colors set (uses defaults).
- Plot types:
    1) Heatmap of MEANS (optional text annotations can show mean or mean±std)
    2) Grouped DOT plot with vertical error bars (mean ± std) per method×atlas
    3) Average-RANK summary with horizontal error bars over seeds/groups

Metric direction:
- Configure whether "higher is better" for ranking via METRIC_HIGHER_IS_BETTER.
"""

from typing import Iterable, Optional, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Which metrics are "higher is better"? Set to False for losses.
METRIC_HIGHER_IS_BETTER: Dict[str, bool] = {
    "BCE": False,
    "AUC (%)": True,
    "F1 (%)": True,
}

def _validate_columns(df: pd.DataFrame):
    required = {'dataset', 'atlas', 'method', 'metric', 'value'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

def _filter_df(df: pd.DataFrame, dataset: str, metric: str) -> pd.DataFrame:
    _validate_columns(df)
    out = df[(df['dataset'] == dataset) & (df['metric'] == metric)].copy()
    if out.empty:
        raise ValueError(f"No rows found for dataset='{dataset}' and metric='{metric}'")
    return out

def _agg_mean_std(df: pd.DataFrame, by_cols) -> pd.DataFrame:
    """Aggregate to mean/std over duplicates (e.g., multiple seeds)."""
    g = df.groupby(by_cols)['value']
    agg = g.agg(['mean', 'std']).reset_index()
    return agg

def plot_metric_heatmap(
    df: pd.DataFrame,
    dataset: str,
    metric: str,
    annotate: str = "none",
    figsize: Tuple[float, float] = (8, 6),
    ticks_font: int = 12,
    labels_font: int = 14,
    title_font: int = 16
) -> plt.Figure:
    """
    Heatmap with rows=methods, cols=atlases for a given dataset+metric.
    The heatmap shows MEANS. You can annotate cells with:
      annotate = "none" | "mean" | "mean_std"  (mean±std as text)

    Note: Showing std as a separate heatmap is often cleaner if you need both
    maps in a paper; this function focuses on means for the color scale.
    """    
    sub = _filter_df(df, dataset, metric)
    agg = _agg_mean_std(sub, by_cols=['dataset', 'atlas', 'method', 'metric'])

    methods = sorted(agg['method'].unique().tolist())
    atlases = sorted(agg['atlas'].unique().tolist())

    piv_mean = agg.pivot_table(index='method', columns='atlas', values='mean').reindex(index=methods, columns=atlases)
    piv_std  = agg.pivot_table(index='method', columns='atlas', values='std').reindex(index=methods, columns=atlases)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    im = ax.imshow(piv_mean.values, aspect='auto')
    ax.set_xticks(np.arange(len(atlases)))
    ax.set_xticklabels(atlases, rotation=45, ha='right', fontsize=ticks_font)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods, fontsize=ticks_font)
    ax.set_xlabel('Atlas', fontsize=labels_font)
    ax.set_ylabel('Method', fontsize=labels_font)
    if dataset == "ABIDEI":
        dataset = "ABIDE I"
    elif dataset == "ABIDEII":
        dataset = "ABIDE II"    
    elif dataset == "ADHD200":
        dataset = "ADHD-200"    
    if not 'BCE' in metric:
        ax.set_title(f"{dataset} - {metric[:-4]} (mean ± std)", fontsize=title_font)
    else:
        ax.set_title(f"{dataset} - {metric} (mean ± std)", fontsize=title_font)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)

    if annotate.lower() in {"mean", "std", "mean_std"}:
        for i in range(len(methods)):
            for j in range(len(atlases)):
                m = piv_mean.values[i, j]
                s = piv_std.values[i, j]
                if np.isfinite(m):
                    if annotate.lower() == "mean":
                        txt = f"{m:.3f}"
                    elif annotate.lower() == "mean_std":
                        if np.isfinite(s):
                            txt = f"{m:.3f}±{s:.3f}"
                        else:
                            txt = f"{m:.3f}"
                    else:
                        txt = f"{s:.3f}"
                    ax.text(j, i, txt, ha="center", va="center")

    fig.tight_layout()
    return fig


def plot_grouped_dotplot(
    df: pd.DataFrame,
    dataset: str,
    metric: str,
    jitter: float = 0.08,
    capsize: float = 3.0,
    figsize: Tuple[float, float] = (10, 6),
    title_font: int = 16,
    ticks_font: int = 12,
    labels_font: int = 14,
    include_legend: bool = False, 
    legend_font: int = 12,
    legend_x: float = 1.0,
    legend_y: float = 1.0,
    ylim: Tuple[float, float] = (0,0)
) -> plt.Figure:
    """
    Grouped dot plot: x-axis=methods, separate dots per atlas for a given dataset+metric.
    Shows mean ± std as vertical error bars.
    """
    sub = _filter_df(df, dataset, metric)
    agg = _agg_mean_std(sub, by_cols=['atlas', 'method'])

    methods = sorted(agg['method'].unique().tolist())
    atlases = sorted(agg['atlas'].unique().tolist())
    x_positions = np.arange(len(methods))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    for k, atlas in enumerate(atlases):
        s = agg[agg['atlas'] == atlas]
        # Align with method order
        means = [s.loc[s['method'] == m, 'mean'].values[0] if m in s['method'].values else np.nan for m in methods]
        stds  = [s.loc[s['method'] == m, 'std'].values[0]  if m in s['method'].values else np.nan for m in methods]

        x = x_positions + (k - (len(atlases) - 1)/2) * jitter
        # errorbar draws markers with vertical bars: mean ± std
        ax.errorbar(x, means, yerr=stds, fmt='o', capsize=capsize, label=atlas)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=ticks_font)
    ax.tick_params(axis='y', labelsize=ticks_font)
    #ax.set_xlabel('Method', fontsize=labels_font)
    ax.set_ylabel(metric, fontsize=labels_font)
    if dataset == "ABIDEI":
        dataset = "ABIDE I"
    elif dataset == "ABIDEII":
        dataset = "ABIDE II"    
    elif dataset == "ADHD200":
        dataset = "ADHD-200"
    if not 'BCE' in metric:
        ax.set_title(f"{dataset} - {metric[:-4]} (mean ± std)", fontsize=title_font)
    else:
        ax.set_title(f"{dataset} - {metric} (mean ± std)", fontsize=title_font)
    if ylim != (0,0):
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
    if include_legend:
        ax.legend(title='Atlas', bbox_to_anchor=(legend_x, legend_y), loc='upper left', fontsize=legend_font,title_fontsize=14)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    fig.tight_layout()
    return fig


def plot_average_rank_summary(
    df: pd.DataFrame,
    metric: str,
    datasets: Optional[Iterable[str]] = None,
    atlases: Optional[Iterable[str]] = None,
    seed_col: str = "seed",
    figsize: Tuple[float, float] = (8, 6),
) -> plt.Figure:
    """
    Compute ranks per (dataset, atlas, seed) across methods for the given metric, then
    plot average rank (mean) ± std across all groups×seeds as horizontal error bars.

    Fallback: if no 'seed' column exists, ranking is done per (dataset, atlas) using mean values,
    and std of ranks is computed across those groups only.
    """
    _validate_columns(df)
    if metric not in df['metric'].unique():
        raise ValueError(f"Metric '{metric}' not found in df['metric'].")

    sub = df[df['metric'] == metric].copy()
    if datasets is not None:
        sub = sub[sub['dataset'].isin(list(datasets))]
    if atlases is not None:
        sub = sub[sub['atlas'].isin(list(atlases))]

    if sub.empty:
        raise ValueError("No rows to rank after filtering datasets/atlases.")

    higher_is_better = METRIC_HIGHER_IS_BETTER.get(metric, True)
    ascending = not higher_is_better

    # Decide grouping keys for ranking
    if seed_col in sub.columns:
        group_keys = ['dataset', 'atlas', seed_col]
    else:
        group_keys = ['dataset', 'atlas']

    def _rank_group(g: pd.DataFrame) -> pd.DataFrame:
        # mean over duplicates within group (e.g., replicate rows for a method)
        p = g.groupby('method', as_index=False)['value'].mean()
        p['rank'] = p['value'].rank(ascending=ascending, method='average')
        return p[['method', 'rank']]

    ranks = sub.groupby(group_keys, as_index=False).apply(_rank_group).reset_index(drop=True)

    stats = ranks.groupby('method', as_index=False)['rank'].agg(['mean', 'std']).reset_index()
    stats = stats.sort_values('mean', ascending=True)

    # Plot: horizontal scatter with xerr=std
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    y = np.arange(len(stats))
    ax.errorbar(stats['mean'], y, xerr=stats['std'], fmt='o', capsize=3.0)

    # Label each point
    for xi, yi, name in zip(stats['mean'], y, stats['method']):
        ax.text(float(xi), int(yi), f"  {name}", va='center')

    ax.set_yticks([])
    ax.set_xlabel('Average Rank ± SD (lower is better)')
    ax.set_title(f"Average Rank Summary · {metric}")
    ax.set_xlim(left=max(0.9, float(stats['mean'].min()) - 0.8), right=float(stats['mean'].max()) + 0.8)
    fig.tight_layout()
    return fig

# ------------------ Demo with Synthetic Data (with seeds) ------------------ #
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    datasets = ["ASD", "ADHD", "AD"]
    atlases = [f"Atlas{i}" for i in range(1, 7)]
    methods = [f"Method{i}" for i in range(1, 10)]
    metrics = ["BCE", "AUC", "F1"]
    seeds = list(range(5))  # 5 random seeds

    rows = []
    for d in datasets:
        for a in atlases:
            for m in methods:
                for s in seeds:
                    bce = rng.normal(loc=0.60, scale=0.10)
                    auc = rng.normal(loc=0.78, scale=0.05)
                    f1  = rng.normal(loc=0.72, scale=0.06)
                    rows.extend([
                        (d, a, m, "BCE", float(np.clip(bce, 0, 2)), s),
                        (d, a, m, "AUC", float(np.clip(auc, 0, 1)), s),
                        (d, a, m, "F1",  float(np.clip(f1, 0, 1)), s),
                    ])
    demo_df = pd.DataFrame(rows, columns=['dataset', 'atlas', 'method', 'metric', 'value', 'seed'])

    # Example usage with seeds:
    fig1 = plot_metric_heatmap(demo_df, dataset="ASD", metric="AUC", annotate="mean_std")
    fig2 = plot_grouped_dotplot(demo_df, dataset="ASD", metric="F1")
    fig3 = plot_average_rank_summary(demo_df, metric="AUC", seed_col="seed")

    # Save demo figures
    fig1.savefig("demo_heatmap_ASD_AUC_meanstd.png", dpi=150, bbox_inches='tight')
    fig2.savefig("demo_grouped_dot_ASD_F1_meanstd.png", dpi=150, bbox_inches='tight')
    fig3.savefig("demo_avg_rank_AUC_meanstd.png", dpi=150, bbox_inches='tight')
    print("Saved demo figures: demo_heatmap_ASD_AUC_meanstd.png, demo_grouped_dot_ASD_F1_meanstd.png, demo_avg_rank_AUC_meanstd.png")
