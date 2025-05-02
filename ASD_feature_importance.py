import torch
from torch import nn
from torch.nn import Linear
import numpy as np
import os

class MLP(nn.Module):
    def __init__(self, num_features, num_classes):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            Linear(num_features, num_classes),
            nn.Softmax(dim=1))

    def forward(self, x):
        z = self.fc(x)
        return z
    

atlas = 'AAL'
if atlas == 'AAL':
    roi_num = 166
elif atlas == 'AICHA':
    roi_num = 384

abs_importance = []
for i in range(10):
    for j in range(10):
        model = MLP(int((roi_num*(roi_num-1))/2),2)
        state_dict = torch.load(os.path.join('models_ABIDE',atlas,'VarCoNetV2','ABIDE_min_val_loss_classifier_rs' + str(i) + '_fold' + str(j) + '.pth'))
        model.load_state_dict(state_dict)
        weights = model.fc[0].weight.detach().cpu().numpy()
        importance = weights[1] - weights[0]
        abs_importance.append(np.abs(importance))

for i in range(10):
    model = MLP(int((roi_num*(roi_num-1))/2),2)
    state_dict = torch.load(os.path.join('models_ABIDE',atlas,'VarCoNetV2','ABIDE_min_val_loss_classifier_rs' + str(i) + '.pth'))
    model.load_state_dict(state_dict)
    weights = model.fc[0].weight.detach().cpu().numpy()
    importance = weights[1] - weights[0]
    abs_importance.append(np.abs(importance))

abs_importance = np.stack(abs_importance)
abs_importance = np.mean(abs_importance, axis=0)
abs_importance = (abs_importance - np.min(abs_importance))/(np.max(abs_importance) - np.min(abs_importance))
np.save(os.path.join('results_ABIDE', atlas, atlas + '_feature_importance.npy'), abs_importance)