from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import avg_pool1d
from utils import ABIDEDataset
import os
import pickle
import math
from sklearn.model_selection import train_test_split
from utils import upper_triangular_cosine_similarity
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from nilearn.datasets import fetch_atlas_yeo_2011
from model_scripts.VarCoNet import VarCoNet


def main(config):

    path = config['path_data']
    
    data = np.load(os.path.join(path,'HCP','test_data_HCP_' + config['atlas'] + '_1_resampled.npz'))
    test_data1 = []
    for key in data:
        test_data1.append(data[key][:200,:])
    
    data = np.load(os.path.join(path,'HCP','test_data_HCP_' + config['atlas'] + '_2_resampled.npz'))
    test_data2 = []
    for key in data:
        test_data2.append(data[key][:200,:])
            
    test_data1 = test_data1[200:]
    test_data2 = test_data2[200:]      
    '''
    names = []
    with open(os.path.join(path,'ABIDE','fmriprep','ABIDE_nilearn_names.txt'), 'r') as f:
        for line in f:
            names.append(line.strip())
    
    data_list = np.load(os.path.join(path,'ABIDE','fmriprep','ABIDE_nilearn_' + config['atlas'] + '.npz'))
    data = []
    for key in data_list:
        data.append(data_list[key])
    
    names_unique, counts = np.unique(names, return_counts=True)
    names_dupl = names_unique[counts > 1]
    pos_duplicates = []
    names_duplicate = []
    for name in names_dupl:
        temp = np.where(np.array(names) == name)[0]
        for t in temp:
            pos_duplicates.append(t)
            names_duplicate.append(name)
    names_unique = names_unique[counts == 1]
    pos_unique = []
    for name in names_unique:
        pos_unique.append(np.where(np.array(names) == name)[0][0])
    train_DATA = [data[i] for i in pos_duplicates]
    data = [data[i] for i in pos_unique]
    y = np.load(os.path.join(path,'ABIDE','fmriprep','ABIDE_nilearn_classes.npy'))   
    Y_train = y[pos_duplicates]
    y = y[pos_unique]
    
    ext_test = list(range(51456,51494))
    names_ext_test = []
    for name in ext_test:      
        if 'sub-00'+str(name) in names_unique:
            names_ext_test.append('sub-00'+str(name))
    names = []
    for name in names_unique:
        if name not in names_ext_test:
            names.append(name)
            
    pos_ext_test = []
    for name in names_ext_test:
        pos_ext_test.append(np.where(np.array(names_unique) == name)[0][0])
    ext_test_data = [data[i] for i in pos_ext_test]
    y_ext_test = y[pos_ext_test]
    
    pos = []
    for name in names:
        pos.append(np.where(np.array(names_unique) == name)[0][0])
    data = [data[i] for i in pos]
    y = y[pos]
    '''
    device = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu")
    batch_size = config['batch_size']
    shuffle = config['shuffle']
    
    model_config = config['model_config'] 
    model_config['max_length'] = config['max_length']                
    
    '''---------------------------------HCP--------------------------------'''
    test_loader1 = DataLoader(test_data1, batch_size=batch_size, shuffle = shuffle)
    test_loader2 = DataLoader(test_data2, batch_size=batch_size, shuffle = shuffle)
    
    roi_num = test_data1[0].shape[1]
    VarCoNetV2 = VarCoNet(model_config, roi_num).to(device)
    random_model = VarCoNet(model_config, roi_num).to(device)
    state_dict = torch.load(os.path.join(config['path_save'],'models_HCP',config['atlas'],'VarCoNet','max_fingerprinting_acc_model.pth'))
    VarCoNetV2.load_state_dict(state_dict)
    
    outputs1 = []
    outputs2 = []
    
    with torch.no_grad():
        for batch_data in test_loader1:
            batch_data = torch.nn.functional.pad(batch_data, (0, 0, 0, 120))
            outputs1.append(VarCoNetV2(batch_data.float().to(device)))
        outputs1 = torch.cat(outputs1, dim=0)
        
        for batch_data in test_loader2:
            batch_data = torch.nn.functional.pad(batch_data, (0, 0, 0, 120))
            outputs2.append(VarCoNetV2(batch_data.float().to(device)))
        outputs2 = torch.cat(outputs2, dim=0)
    outputs = torch.cat((outputs1,outputs2), dim=0)
    
    test_data1 = np.stack(test_data1)
    test_data2 = np.stack(test_data2)
    X_orig = np.concatenate((test_data1, test_data2), axis=0)
    X = outputs.cpu()
    triu_indices = torch.triu_indices(roi_num, roi_num, offset=1)
    fc = np.zeros((X.shape[0], roi_num, roi_num))
    fc[:, triu_indices[0], triu_indices[1]] = X
    fc[:, triu_indices[1], triu_indices[0]] = X
    fc[:, np.arange(roi_num), np.arange(roi_num)] = 1.0
    
    if not os.path.exists(os.path.join("results","functional_connectomes_HCP",config['atlas'])):
        os.makedirs(os.path.join("results","functional_connectomes_HCP",config['atlas']), exist_ok=True)
    if not os.path.exists(os.path.join("results","functional_connectomes_ABIDE",config['atlas'])):
        os.makedirs(os.path.join("results","functional_connectomes_ABIDE",config['atlas']), exist_ok=True)
    for i in range(100):
        fc_examp = fc[i]
        fc_examp = fc_examp - np.eye(fc_examp.shape[0])
        plt.figure(figsize=(10, 8))
        sns.heatmap(fc_examp, cmap='coolwarm', center=0, square=True)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join("results","functional_connectomes_HCP",config['atlas'],"VarCoNetV2_sub_" + str(i) + "_rest1.png"), dpi=600, bbox_inches='tight')
        #plt.show()
        plt.close()
        
        fc_pcc_examp = np.corrcoef(X_orig[i].T)
        fc_pcc_examp = fc_pcc_examp - np.eye(fc_pcc_examp.shape[0])
        plt.figure(figsize=(10, 8))
        sns.heatmap(fc_pcc_examp, cmap='coolwarm', center=0, square=True, vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join("results","functional_connectomes_HCP",config['atlas'],"PCC_sub_" + str(i) + "_rest1.png"), dpi=600, bbox_inches='tight')
        #plt.show()
        plt.close()
        
        fc_examp = fc[fc.shape[0]//2+i]
        fc_examp = fc_examp - np.eye(fc_examp.shape[0])
        plt.figure(figsize=(10, 8))
        sns.heatmap(fc_examp, cmap='coolwarm', center=0, square=True, vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join("results","functional_connectomes_HCP",config['atlas'],"VarCoNetV2_sub_" + str(i) + "_rest2.png"), dpi=600, bbox_inches='tight')
        #plt.show()
        plt.close()
        
        fc_pcc_examp = np.corrcoef(X_orig[fc.shape[0]//2+i].T)
        fc_pcc_examp = fc_pcc_examp - np.eye(fc_pcc_examp.shape[0])
        plt.figure(figsize=(10, 8))
        sns.heatmap(fc_pcc_examp, cmap='coolwarm', center=0, square=True, vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join("results","functional_connectomes_HCP",config['atlas'],"PCC_sub_" + str(i) + "_rest2.png"), dpi=600, bbox_inches='tight')
        #plt.show()
        plt.close()
    
    '''-----------------------------------ABIDE--------------------------------'''
    for i in range(10):
        test_dataset = ABIDEDataset(ext_test_data, y_ext_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = shuffle)
        
        roi_num = ext_test_data[0].shape[1]
        VarCoNetV2 = VarCoNet(model_config, roi_num).to(device)
        state_dict = torch.load(os.path.join('models_ABIDE',config['atlas'],'VarCoNetV2','ABIDE_min_val_loss_model_rs' + str(i) + '.pth'))
        VarCoNetV2.load_state_dict(state_dict)
        
        outputs = []
        labels = []
        VarCoNetV2.eval()
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                outputs.append(VarCoNetV2(batch_data.to(device)))
                labels.append(batch_labels)
            outputs = torch.cat(outputs, dim=0)
            labels = torch.cat(labels, dim=0)
        X_ASD = outputs[labels==1].cpu()
        X_NC = outputs[labels==0].cpu()
        triu_indices = torch.triu_indices(roi_num, roi_num, offset=1)
        fc_ASD = torch.zeros((X_ASD.shape[0], roi_num, roi_num))
        fc_ASD[:, triu_indices[0], triu_indices[1]] = X_ASD
        fc_ASD[:, triu_indices[1], triu_indices[0]] = X_ASD
        fc_ASD[:, torch.arange(roi_num), torch.arange(roi_num)] = 1.0
        fc_NC = torch.zeros((X_NC.shape[0], roi_num, roi_num))
        fc_NC[:, triu_indices[0], triu_indices[1]] = X_NC
        fc_NC[:, triu_indices[1], triu_indices[0]] = X_NC
        fc_NC[:, torch.arange(roi_num), torch.arange(roi_num)] = 1.0

        fc_ASD = np.mean(fc_ASD.numpy(), axis=0)
        fc_NC = np.mean(fc_NC.numpy(), axis=0)
        plt.figure(figsize=(10, 8))
        sns.heatmap(fc_ASD, cmap='coolwarm', center=0, square=True, vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join("results_ABIDE","functional_connectomes_ABIDE",config['atlas'],"VarCoNetV2_ASD_rs_" + str(i) + ".png"), dpi=600, bbox_inches='tight')
        plt.close()
        #plt.show()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(fc_NC, cmap='coolwarm', center=0, square=True, vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join("results_ABIDE","functional_connectomes_ABIDE",config['atlas'],"VarCoNetV2_NC_rs_" + str(i) + ".png"), dpi=600, bbox_inches='tight')
        plt.close()
        #plt.show()

if __name__ == '__main__':
    config = {}
    config['path_data'] = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data_nilearn_0.11.1'
    config['path_save'] = r'/home/student1/Desktop/Charalampos_Lamprou/VarCoNet_results_revision'
    config['atlas'] = 'AAL' #AICHA, AAL
    with open(f'best_params_VarCoNet_{config["atlas"]}.pkl', 'rb') as f:
        best_params = pickle.load(f)
    config['max_length'] = 320
    config['batch_size'] = 128
    config['shuffle'] = True
    config['model_config'] = {}
    config['model_config']['layers'] = best_params['layers']
    config['model_config']['n_heads'] = best_params['n_heads']
    config['model_config']['dim_feedforward'] = best_params['dim_feedforward']    
    config['device'] = "cuda:0"
    results = main(config)



 