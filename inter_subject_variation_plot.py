import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from matplotlib.patches import Polygon

num_samples = 100
num_columns = 8
means = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])#np.concatenate((np.linspace(0.1, 0.4, int(num_columns/2)),
                        #np.linspace(0.6, 0.9, int(num_columns/2))), axis=0)
std = 0.15
data = np.random.normal(loc=means, scale=std, size=(num_samples, num_columns))

fig, axes = plt.subplots(nrows=num_columns, ncols=1, figsize=(6, 12))
colors = ['blue'] * 4 + ['red'] * 4

for i, ax in enumerate(axes):
    col_data = data[:, i]
    mean = np.mean(col_data)
    std = np.std(col_data)

    kde = gaussian_kde(col_data)
    x_vals = np.linspace(-0.65, 1.65, 500)
    y_vals = kde(x_vals)

    # Fill full KDE (subtle base)
    ax.fill_between(x_vals, y_vals, color=colors[i], alpha=0.2)
    
    # Highlight region within mean ± CI
    mask = (x_vals >= mean - std) & (x_vals <= mean + std)
    ax.fill_between(x_vals[mask], y_vals[mask], color=colors[i], alpha=0.3)

    # Masked CI region under curve
    mask = (x_vals >= mean - std) & (x_vals <= mean + std)
    verts = np.column_stack((x_vals[mask], y_vals[mask]))
    verts = np.concatenate([[[x_vals[mask][0], 0]], verts, [[x_vals[mask][-1], 0]]])  # close polygon

    hatch_patch = Polygon(verts, closed=True, hatch='//', edgecolor='gray',
                          facecolor='none', linewidth=0.0)
    ax.add_patch(hatch_patch)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.set_frame_on(False)
    ax.set_xlim(-0.5, 1.45)

plt.tight_layout()
plt.savefig('intra_subject_variation_hatched_std_' + str(std) + '.png', dpi=600)
plt.show()
