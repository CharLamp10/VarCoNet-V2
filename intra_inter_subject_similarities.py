#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 11:31:04 2025

@author: student1
"""

import numpy as np
import torch
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt

atlas = 'AICHA'
path_results = r'/home/student1/Desktop/Charalampos_Lamprou/VarCoNetV2/results'

if atlas == 'AICHA':
    vmax = 0.7
elif atlas == 'AAL':
    vmax = 0.77
    

with open(os.path.join(path_results,'test_results_' + atlas + '_VarCoNetV2.pkl'), 'rb') as f:
    results_VarCoNetV2 = pickle.load(f)
similarities = torch.stack(results_VarCoNetV2['test_result'][-1]).cpu()
similarities = torch.mean(similarities, dim=0).numpy()
similarities = similarities[150:250,150:250]
plt.figure(figsize=(10, 8))
ax=sns.heatmap(similarities, cmap='coolwarm', center=0, square=True, vmin=0, vmax=vmax)
plt.xticks([])
plt.yticks([])
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.savefig(os.path.join("paper_plots","VarCoNetV2_" + atlas + "_100_subjs.png"), dpi=600, bbox_inches='tight')
plt.show()


with open(os.path.join(path_results,'test_results_' + atlas + '_VAE.pkl'), 'rb') as f:
    results_VAE = pickle.load(f)
similarities = np.stack(results_VAE['result_test'][-1])
similarities = np.mean(similarities, axis=0)
similarities = similarities[150:250,150:250]
plt.figure(figsize=(10, 8))
ax=sns.heatmap(similarities, cmap='coolwarm', center=0, square=True, vmin=0, vmax=vmax)
plt.xticks([])
plt.yticks([])
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.savefig(os.path.join("paper_plots","VAE_" + atlas + "_100_subjs.png"), dpi=600, bbox_inches='tight')
plt.show()


with open(os.path.join(path_results,'test_results_' + atlas + '_AE.pkl'), 'rb') as f:
    results_AE = pickle.load(f)
similarities = np.stack(results_AE['result_test'][-1])
similarities = np.mean(similarities, axis=0)
similarities = similarities[150:250,150:250]
plt.figure(figsize=(10, 8))
ax=sns.heatmap(similarities, cmap='coolwarm', center=0, square=True, vmin=0, vmax=vmax)
plt.xticks([])
plt.yticks([])
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.savefig(os.path.join("paper_plots","AE_" + atlas + "_100_subjs.png"), dpi=600, bbox_inches='tight')
plt.show()



with open(os.path.join(path_results,'test_results_' + atlas + '_PCC.pkl'), 'rb') as f:
    results_PCC = pickle.load(f)
similarities = np.stack(results_PCC['test_result'][-1])
similarities = np.mean(similarities, axis=0)
similarities = similarities[150:250,150:250]
plt.figure(figsize=(10, 8))
ax=sns.heatmap(similarities, cmap='coolwarm', center=0, square=True, vmin=0, vmax=vmax)
plt.xticks([])
plt.yticks([])
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.savefig(os.path.join("paper_plots","PCC_" + atlas + "_100_subjs.png"), dpi=600, bbox_inches='tight')
plt.show()