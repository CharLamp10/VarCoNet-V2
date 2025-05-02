import numpy as np
import os
import pickle
import pandas as pd
from torch.nn import Softmax, BCELoss
import torch.nn.functional as F
from torch import from_numpy
from sklearn.metrics import roc_auc_score

soft = Softmax(dim=-1)
loss_func = BCELoss()
path = r'/home/student1/Desktop/Charalampos_Lamprou/VarCoNetV2_extras/learning-from-brains/results/models/downstream/ABIDE'
dire = os.listdir(path)
losses = []
val_losses = []
test_losses = []
test_aucs = []
test_probs = []
y_test = []
losses_ext = []
val_losses_ext = []
ext_test_losses = []
ext_test_aucs = []
ext_test_probs = []
for folder in dire:
    path_res = os.path.join(path,folder)
    path_train_history = os.path.join(path_res,'train_history.csv')
    path_eval_history = os.path.join(path_res,'eval_history.csv')
    path_predictions = os.path.join(path_res,'test_predictions.npy')
    path_labels = os.path.join(path_res,'test_label_ids.npy')
    train_history = pd.read_csv(path_train_history)
    eval_history = pd.read_csv(path_eval_history)
    predictions = np.load(path_predictions)
    predictions = soft(from_numpy(predictions)).numpy()
    y = np.load(path_labels)
    if 'fold' in folder:
        losses.append(train_history['loss'].values)
        val_losses.append(np.min(train_history['loss'].values))
        test_losses.append(loss_func(from_numpy(predictions),F.one_hot(from_numpy(y), num_classes=2).float()).numpy())
        test_aucs.append(roc_auc_score(y, predictions[:,-1]))
        y_test.append(y)
        test_probs.append(predictions[:,-1])
    else:
        losses_ext.append(train_history['loss'].values)
        val_losses_ext.append(np.min(train_history['loss'].values))
        ext_test_losses.append(loss_func(from_numpy(predictions),F.one_hot(from_numpy(y), num_classes=2).float()).numpy())
        ext_test_aucs.append(roc_auc_score(y, predictions[:,-1]))
        y_ext_test = y
        ext_test_probs.append(predictions[:,-1])

results = {}
results['losses'] = losses
results['val_losses'] = val_losses
results['test_losses'] = test_losses
results['test_aucs'] = test_aucs
results['test_probs'] = test_probs
results['y_test'] = y_test
results['losses_ext'] = losses_ext
results['val_losses_ext'] = val_losses_ext
results['ext_test_losses'] = ext_test_losses
results['ext_test_aucs'] = ext_test_aucs
results['ext_test_probs'] = ext_test_probs
results['y_ext_test'] = y_ext_test


with open(os.path.join('results_ABIDE','ABIDE_LFB_results.pkl'), "wb") as pickle_file:
            pickle.dump(results, pickle_file)
