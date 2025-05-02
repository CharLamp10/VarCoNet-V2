parent_path = r'/home/student1/Desktop/Charalampos_Lamprou/VarCoNetV2_extras'
import sys
sys.path.append(parent_path)
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import torch.nn.functional as F
from torch import nn
from utils import ABIDEDataset
from BolT.Models.BolT.model import Model
from BolT.utils import Option
from BolT.Models.BolT.hyperparams import getHyper_bolT
from sklearn.metrics import roc_auc_score
from torch.nn.functional import avg_pool1d
import pickle
from utils import upper_triangular_cosine_similarity
    
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
        y_test = []
        for (x,y) in test_loader:
            x = x.to(device)
            y_test.append(y)
            outputs_test.append(encoder_model(x))
        outputs_test = torch.cat(outputs_test, dim=0).clone().detach()
        y_test = torch.cat(y_test,dim=0).to(device)
        logits = classifier(outputs_test)
    loss_func = nn.BCELoss()
    loss = loss_func(logits, F.one_hot(y_test, num_classes=2).float())
    z = logits[:,-1].cpu().numpy()
    y = y_test.cpu().numpy()
    auc_score = roc_auc_score(y, z)
    return loss.item(), auc_score, z, y


def test_bolt(model, test_data_loader):
    with torch.no_grad():
        zs = []
        ys = []
        for (x,y) in test_data_loader:
            zero_rows = torch.all(x == 0, dim=-1)
            zero_row_indices = torch.where(zero_rows)[1]
            if len(zero_row_indices) > 0:
                x = x[:,:torch.min(zero_row_indices),:]
            x = torch.permute(x,(0,2,1))
            x = (x - torch.mean(x, dim=-1, keepdim=True))/torch.std(x, dim=-1, keepdim=True)
            _, _, test_probs, _ = model.step(x, y, train=False)
            zs.append(test_probs)
            ys.append(y)
        z = torch.cat(zs,dim=0)
        y = torch.cat(ys,dim=0)
        loss_func = nn.BCELoss()
        loss = loss_func(z, F.one_hot(y, num_classes=2).float())
        z = z[:,-1].cpu().numpy()
        y = y.numpy()
        auc_score = roc_auc_score(y, z)
                   
    return loss.item(), auc_score, z, y


def main(config):
    path = config['path_data']
    
    data_list = np.load(os.path.join(path,'ABIDEII_nilearn_' + config['atlas'] + '.npz'))
    test_data = []
    for key in data_list:
        test_data.append(data_list[key])
    y = np.load(os.path.join(path,'ABIDEII_nilearn_classes.npy'))  
    
    roi_num = test_data[0].shape[1]
    
    device = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu")
    batch_size = config['batch_size']
    model_config = config['model_config'] 
    model_config['max_length'] = config['max_length']
    
    test_losses_bolt = []
    test_aucs_bolt = []
    test_probs_bolt = []
    test_losses_varconetv2 = []
    test_aucs_varconetv2 = []
    test_probs_varconetv2 = []
    for i in range(10):
        test_dataset = ABIDEDataset(test_data, y)
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
        state_dict_bolt = torch.load(os.path.join('models_ABIDE',config['atlas'],'BolT','ABIDE_min_val_loss_model_rs' + str(i) + '.pth'))
        bolt.model.load_state_dict(state_dict_bolt)
        test_loss,test_auc,test_prob,y_ext_test = test_bolt(bolt, test_loader)
        test_losses_bolt.append(test_loss)
        test_aucs_bolt.append(test_auc)
        test_probs_bolt.append(test_prob)
        
        VarCoNetV2 = SeqenceModel(model_config, roi_num).to(device)
        classifier = MLP(int(roi_num*(roi_num-1)/2),2).to(device)
        state_dict_varconet = torch.load(os.path.join('models_ABIDE',config['atlas'],'VarCoNetV2','ABIDE_min_val_loss_model_rs' + str(i) + '.pth'))
        state_dict_cls = torch.load(os.path.join('models_ABIDE',config['atlas'],'VarCoNetV2','ABIDE_min_val_loss_classifier_rs' + str(i) + '.pth'))
        VarCoNetV2.load_state_dict(state_dict_varconet)
        classifier.load_state_dict(state_dict_cls)
        test_loss,test_auc,test_prob,y_ext_test = test_varconet(VarCoNetV2, classifier, test_loader, device)
        test_losses_varconetv2.append(test_loss)
        test_aucs_varconetv2.append(test_auc)
        test_probs_varconetv2.append(test_prob)
        
    results = {}
    results['bolt'] = {}
    results['bolt']['test_losses'] = test_losses_bolt
    results['bolt']['test_aucs'] = test_aucs_bolt
    results['bolt']['test_probs'] = test_probs_bolt
    results['bolt']['y_test'] = y
    results['VarCoNetV2'] = {}
    results['VarCoNetV2']['test_losses'] = test_losses_varconetv2
    results['VarCoNetV2']['test_aucs'] = test_aucs_varconetv2
    results['VarCoNetV2']['test_probs'] = test_probs_varconetv2
    results['VarCoNetV2']['y_test'] = y
    
    if config['save_results']:
        if not os.path.exists(os.path.join('results_ABIDEII',config['atlas'])):
            os.makedirs(os.path.join('results_ABIDEII',config['atlas']),exist_ok=True)
        with open(os.path.join('results_ABIDEII',config['atlas'],'results.pkl'), 'wb') as f:
            pickle.dump(results,f)
    return results
        
if __name__ == '__main__':
    config = {}
    config['path_data'] = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/ABIDEII/fmriprep'
    config['atlas'] = 'AICHA' #AICHA, AAL
    with open('best_params_VarCoNet_v2_final_' + config['atlas'] + '.pkl','rb') as f:
        best_params = pickle.load(f)
    config['batch_size'] = 128
    config['max_length'] = 320
    config['model_config'] = {}
    config['model_config']['layers'] = best_params['layers']
    config['model_config']['n_heads'] = best_params['n_heads']
    config['model_config']['dim_feedforward'] = best_params['dim_feedforward']   
    config['device'] = "cuda:1"
    config['save_results'] = False
    results = main(config)