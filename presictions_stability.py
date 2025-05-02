parent_path = r'/home/student1/Desktop/Charalampos_Lamprou/VarCoNetV2_extras'
import sys
sys.path.append(parent_path)
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
from torch import nn
from utils import ABIDEDataset
from BolT.Models.BolT.model import Model
from BolT.utils import Option
from BolT.Models.BolT.hyperparams import getHyper_bolT
from torch.nn.functional import avg_pool1d
import math
import pickle
from utils import upper_triangular_cosine_similarity
from sklearn.metrics import roc_curve



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class PositionalEncodingTrainable(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncodingTrainable, self).__init__()

        pe = torch.zeros(1, max_seq_length, d_model)
        self.pe = nn.Parameter(pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Transformer(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, dim_feedforward, max_len):
        super(Transformer, self).__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4, stride=2)
        self.bn = nn.InstanceNorm1d(d_model)
        new_len = (max_len - 4) // 2 + 1
        self.pos_enc = PositionalEncodingTrainable(d_model,new_len)
        
        
    def forward(self, x, x_mask=None):   
        b, k, d = x.shape
        x = self.bn(x.permute(0,2,1)).permute(0,2,1)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view((b*d, 1, k))
        x = self.conv1(x)
        x = avg_pool1d(x.permute(0, 2, 1), kernel_size=16).permute(0, 2, 1)
        x = x.view((b, d, -1))
        x = torch.transpose(x, 1, 2)
        if torch.sum(x==x[0,-1,0]) >= d:
            x[x==x[0,-1,0]] = 0  
        x_mask = (x[:, :, 0] == 0).bool()    
        x = self.pos_enc(x)
        x[x_mask,:] = 0
        x = self.transformer_encoder(x, src_key_padding_mask=x_mask)     
        x[x_mask,:] = 0
        x = torch.transpose(x, 1, 2)
        return x


class SeqenceModel(nn.Module):
    def __init__(self, model_config, roi_num):
        super().__init__()

        self.extract = Transformer(
                d_model=roi_num,
                n_layers=model_config['layers'],n_heads=model_config['n_heads'],
                dim_feedforward=model_config['dim_feedforward'], 
                max_len=model_config['max_length'])

    def forward(self, x):
        x = self.extract(x)
        x = upper_triangular_cosine_similarity(x)
        return x  

class MLP(nn.Module):
    def __init__(self, num_features, num_classes):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Softmax(dim=1))

    def forward(self, x):
        z = self.fc(x)
        return z


def test_varconet(encoder_model, classifier, test_loader, device):
    encoder_model.eval()
    classifier.eval()
    with torch.no_grad():        
        outputs_test= []
        for (x,y) in test_loader:
            x = x.to(device)
            outputs_test.append(encoder_model(x))
        outputs_test = torch.cat(outputs_test, dim=0).clone().detach()
        logits = classifier(outputs_test)
    z = logits[:,-1].cpu().numpy()
    return z


def test_bolt(model, test_data_loader):
    with torch.no_grad():
        zs = []
        for (x,y) in test_data_loader:
            zero_rows = torch.all(x == 0, dim=-1)
            zero_row_indices = torch.where(zero_rows)[1]
            if len(zero_row_indices) > 0:
                x = x[:,:torch.min(zero_row_indices),:]
            x = torch.permute(x,(0,2,1))
            x = (x - torch.mean(x, dim=-1, keepdim=True))/torch.std(x, dim=-1, keepdim=True)
            _, _, test_probs, _ = model.step(x, y, train=False)
            zs.append(test_probs)
        z = torch.cat(zs,dim=0)
        z = z[:,-1].cpu().numpy()
                   
    return z

def cross_entropy(p, q):
    p = np.clip(p, 1e-9, 1)
    q = np.clip(q, 1e-9, 1)
    ce = -np.sum(p * np.log(q))
    return ce


def main(config):
    path = config['path_data']
    
    with open(os.path.join('results_ABIDE',config['atlas'],'ABIDE_VarCoNetV2_results.pkl'), 'rb') as f:
        result_varconet = pickle.load(f)
    with open(os.path.join('results_ABIDE',config['atlas'],'ABIDE_BolT_results.pkl'), 'rb') as f:
        result_bolt = pickle.load(f)
    
    if config['atlas'] == 'AAL':
        roi_num = 166
    else:
        roi_num = 384
    test_probs_bolt = []
    test_probs_varconetv2 = []
    change_BolT = 0
    change_VarCoNetV2 = 0
    for j in range(10):
        ext_test_aucs_bolt = np.array(result_bolt['ext_test_aucs'][j])
        best_epoch = np.where(ext_test_aucs_bolt == np.max(ext_test_aucs_bolt))[0][0]
        ext_test_probs_bolt = result_bolt['ext_test_probs'][j][best_epoch]
        
        ext_test_aucs_varconet = []
        for i in range(50):
            ext_test_aucs_varconet.append(np.array(result_varconet['epoch_results_ext'][j][i]['best_test_auc']))
        best_epoch = np.where(ext_test_aucs_varconet == np.max(ext_test_aucs_varconet))[0][0]
        ext_test_probs_varconet = result_varconet['epoch_results_ext'][j][best_epoch]['test_probs']
        
        data_list = np.load(os.path.join(path,'ABIDE_nilearn_' + config['atlas'] + '.npz'))
        Y = np.load(os.path.join(path,'ABIDE_nilearn_classes.npy'))   
        
        for wind_size in config['test_winds']:
            step_size = (200 - wind_size) // (config['num_winds'] - 1)
            for i in range(0, step_size*config['num_winds'],step_size):
                names = []
                with open(os.path.join(path,'ABIDE_nilearn_names.txt'), 'r') as f:
                    for line in f:
                        names.append(line.strip())
                names_unique, counts = np.unique(names, return_counts=True)
                ext_test = list(range(51456,51494))
                names_ext_test = []
                for name in ext_test:      
                    if 'sub-00'+str(name) in names_unique:
                        names_ext_test.append('sub-00'+str(name))
                test_data = []
                y_ext_test = []
                for (key,name,y) in zip(data_list,names,Y):
                    if name in names_ext_test:
                        data_temp = data_list[key]
                        zero_rows = np.all(data_temp == 0, axis=-1)
                        zero_row_indices = np.where(zero_rows)[0]
                        if len(zero_row_indices) > 0:
                            data_temp = data_temp[:np.min(zero_row_indices),:]
                        data_temp = data_temp[i:i+wind_size,:]
                        data_temp = np.pad(data_temp, ((0, config['max_length']-data_temp.shape[0]), (0, 0)), mode='constant', constant_values=0)
                        test_data.append(data_temp)
                        y_ext_test.append(y)
                y_ext_test = np.array(y_ext_test)
                device = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu")
                batch_size = config['batch_size']
                model_config = config['model_config'] 
                model_config['max_length'] = config['max_length']
                
                test_dataset = ABIDEDataset(test_data, y_ext_test)
                test_loader = DataLoader(test_dataset, batch_size=batch_size)  
                
                details = Option({
                    "device" : device,
                    "nOfTrains" : len(test_data),
                    "nOfClasses" : 2,
                    "batchSize" : batch_size,
                    "nOfEpochs" : 1
                })
                hyperParams = getHyper_bolT(rois = roi_num)
                bolt = Model(hyperParams, details)
                state_dict_bolt = torch.load(os.path.join('models_ABIDE',config['atlas'],'BolT','ABIDE_min_val_loss_model_rs' + str(j) + '.pth'))
                bolt.model.load_state_dict(state_dict_bolt)
                test_prob = test_bolt(bolt, test_loader)
                test_probs_bolt.append(test_prob)
                fpr, tpr, thresholds = roc_curve(y_ext_test, test_prob)
                youden_j = tpr - fpr
                best_index = np.argmax(youden_j)
                best_threshold = thresholds[best_index]
                below = (test_prob < best_threshold) & (ext_test_probs_bolt < best_threshold)
                above = (test_prob >= best_threshold) & (ext_test_probs_bolt >= best_threshold)
                change_BolT = change_BolT + np.sum(below | above)
                
                VarCoNetV2 = SeqenceModel(model_config, roi_num).to(device)
                classifier = MLP(int(roi_num*(roi_num-1)/2),2).to(device)
                state_dict_varconet = torch.load(os.path.join('models_ABIDE',config['atlas'],'VarCoNetV2','ABIDE_min_val_loss_model_rs' + str(j) + '.pth'))
                state_dict_cls = torch.load(os.path.join('models_ABIDE',config['atlas'],'VarCoNetV2','ABIDE_min_val_loss_classifier_rs' + str(j) + '.pth'))
                VarCoNetV2.load_state_dict(state_dict_varconet)
                classifier.load_state_dict(state_dict_cls)
                test_prob = test_varconet(VarCoNetV2, classifier, test_loader, device)
                test_probs_varconetv2.append(test_prob)
                fpr, tpr, thresholds = roc_curve(y_ext_test, test_prob)
                youden_j = tpr - fpr
                best_index = np.argmax(youden_j)
                best_threshold = thresholds[best_index]
                below = (test_prob < best_threshold) & (ext_test_probs_varconet < best_threshold)
                above = (test_prob >= best_threshold) & (ext_test_probs_varconet >= best_threshold)
                change_VarCoNetV2 = change_VarCoNetV2 + np.sum(below | above)
    
    total = y_ext_test.shape[0]*10*config['num_winds']*len(config['test_winds'])
    change_BolT = (total - change_BolT) / total
    change_VarCoNetV2 = (total - change_VarCoNetV2) / total
    
    results = {}
    results['bolt'] = {}
    results['bolt']['test_probs'] = test_probs_bolt
    results['bolt']['percent_change'] = change_BolT
    results['bolt']['base_probs'] = ext_test_probs_bolt
    results['VarCoNetV2'] = {}
    results['VarCoNetV2']['test_probs'] = test_probs_varconetv2
    results['VarCoNetV2']['percent_change'] = change_VarCoNetV2
    results['VarCoNetV2']['base_probs'] = ext_test_probs_varconet
    
    if config['save_results']:
        if not os.path.exists(os.path.join('results_ABIDE',config['atlas'])):
            os.makedirs(os.path.join('results_ABIDE',config['atlas']),exist_ok=True)
        with open(os.path.join('results_ABIDE',config['atlas'],'ABIDE_BolT_VarCoNetV2_predictions_stability.pkl'), 'wb') as f:
            pickle.dump(results,f)
    return results
        
if __name__ == '__main__':
    config = {}
    config['path_data'] = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/ABIDE/fmriprep'
    config['atlas'] = 'AICHA' #AICHA, AAL
    with open('best_params_VarCoNet_v2_final_' + config['atlas'] + '.pkl','rb') as f:
        best_params = pickle.load(f)
    config['batch_size'] = 128
    config['max_length'] = 320
    config['test_winds'] = [120,160]
    config['num_winds'] = 3
    config['model_config'] = {}
    config['model_config']['layers'] = best_params['layers']
    config['model_config']['n_heads'] = best_params['n_heads']
    config['model_config']['dim_feedforward'] = best_params['dim_feedforward']   
    config['device'] = "cuda:1"
    config['save_results'] = True
    
    results = main(config)