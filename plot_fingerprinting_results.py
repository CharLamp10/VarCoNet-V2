import pickle
import numpy as np
import os
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


atlas = 'AAL'
path_results = r'/home/student1/Desktop/Charalampos_Lamprou/VarCoNetV2/results'

with open(os.path.join(path_results, 'test_results_' + atlas + '_VarCoNetV2.pkl'), 'rb') as f:
    test_result_VarCoNetV2 = pickle.load(f)
with open(os.path.join(path_results, 'test_results_' + atlas + '_PCC.pkl'), 'rb') as f:
    test_result_PCC = pickle.load(f)
with open(os.path.join(path_results, 'test_results_' + atlas + '_VAE.pkl'), 'rb') as f:
    test_result_VAE = pickle.load(f)
with open(os.path.join(path_results, 'test_results_' + atlas + '_AE.pkl'), 'rb') as f:
    test_result_AE = pickle.load(f)

losses = test_result_VarCoNetV2['losses']

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=np.arange(1, len(losses) + 1),
    y=losses,
    mode='lines',
    name='Training Loss',
    line=dict(color='red', width=3),
))
fig.update_layout(
    xaxis_title='Epoch',
    yaxis_title='Loss',
    font=dict(size=20),  # Controls overall font size (e.g., legend, title)
    xaxis=dict(
        tickmode='linear',
        tick0=0,
        dtick=5,
        tickfont=dict(size=18),  # X-tick font size
    ),
    yaxis=dict(
        showgrid=True,
        tickfont=dict(size=18),  # Y-tick font size
    ),
)

fig.write_image(os.path.join('paper_plots','HCP_training_loss_' + atlas + '.png'), scale=8, engine = "orca")


VarCoNetV2_0_0 = test_result_VarCoNetV2['val_result'][0][0]
VarCoNetV2_0_1 = test_result_VarCoNetV2['val_result'][0][1]
VarCoNetV2_0_2 = test_result_VarCoNetV2['val_result'][0][2]
VarCoNetV2_1_1 = test_result_VarCoNetV2['val_result'][0][3]
VarCoNetV2_1_2 = test_result_VarCoNetV2['val_result'][0][4]
VarCoNetV2_2_2 = test_result_VarCoNetV2['val_result'][0][5]
PCC_0_0 = test_result_PCC['val_result'][0][0]
PCC_0_1 = test_result_PCC['val_result'][0][1]
PCC_0_2 = test_result_PCC['val_result'][0][2]
PCC_1_1 = test_result_PCC['val_result'][0][3]
PCC_1_2 = test_result_PCC['val_result'][0][4]
PCC_2_2 = test_result_PCC['val_result'][0][5]
VAE_0_0 = test_result_VAE['result_val'][0][0]
VAE_0_1 = test_result_VAE['result_val'][0][1]
VAE_0_2 = test_result_VAE['result_val'][0][2]
VAE_1_1 = test_result_VAE['result_val'][0][3]
VAE_1_2 = test_result_VAE['result_val'][0][4]
VAE_2_2 = test_result_VAE['result_val'][0][5]
AE_0_0 = test_result_AE['result_val'][0][0]
AE_0_1 = test_result_AE['result_val'][0][1]
AE_0_2 = test_result_AE['result_val'][0][2]
AE_1_1 = test_result_AE['result_val'][0][3]
AE_1_2 = test_result_AE['result_val'][0][4]
AE_2_2 = test_result_AE['result_val'][0][5]

data = np.concatenate((VarCoNetV2_0_0, VarCoNetV2_0_1, VarCoNetV2_0_2,
                    VarCoNetV2_1_1, VarCoNetV2_1_2, VarCoNetV2_2_2,
                    PCC_0_0, PCC_0_1, PCC_0_2, PCC_1_1, PCC_1_2, PCC_2_2,
                    VAE_0_0, VAE_0_1, VAE_0_2, VAE_1_1, VAE_1_2, VAE_2_2,
                    AE_0_0, AE_0_1, AE_0_2, AE_1_1, AE_1_2, AE_2_2), axis=0)

groups = np.concatenate((np.full((10,), '2-2'),np.full((10,), '2-5'),
                    np.full((10,), '2-8'),np.full((10,), '5-5'),
                    np.full((10,), '5-8'),np.full((10,), '8-8'),
                    np.full((10,), '2-2'),np.full((10,), '2-5'),
                    np.full((10,), '2-8'),np.full((10,), '5-5'),
                    np.full((10,), '5-8'),np.full((10,), '8-8'),
                    np.full((10,), '2-2'),np.full((10,), '2-5'),
                    np.full((10,), '2-8'),np.full((10,), '5-5'),
                    np.full((10,), '5-8'),np.full((10,), '8-8'),
                    np.full((10,), '2-2'),np.full((10,), '2-5'),
                    np.full((10,), '2-8'),np.full((10,), '5-5'),
                    np.full((10,), '5-8'),np.full((10,), '8-8')),axis=0)
labels = np.concatenate((np.full((60,), 'VarCoNet'), np.full((60,), 'PCC'),
                         np.full((60,), '(Lu et al., 2024)'), np.full((60,), '(Cai et al., 2021)')),axis=0)    

df = pd.DataFrame({'Value': data, 'Method': labels, 'Group': groups})

plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

ax = sns.boxplot(x='Group', y='Value', hue='Method', data=df, palette='Set2')

plt.ylabel('Accuracy', fontsize=20)
plt.xlabel('Duration Combinations (mins)', fontsize=20)

plt.legend(loc='lower right', fontsize = 16)
plt.xticks(rotation=0, fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig(os.path.join('paper_plots','comparison_boxplot_' + atlas + '.png'), dpi=600, bbox_inches='tight')
plt.show()