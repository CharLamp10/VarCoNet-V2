from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from utils import InfoNCE
from tqdm import tqdm
from torch.optim import Adam
from utils import DualBranchContrast
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
import os
import pickle
import math
import copy
from utils import upper_triangular_cosine_similarity, test_augment, augment_hcp_old, removeDuplicates


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
        self.bn = nn.InstanceNorm1d(d_model)
        self.pos_enc = PositionalEncodingTrainable(d_model,max_len)
        
        
    def forward(self, x, x_mask=None):   
        b, k, d = x.shape
        x = self.bn(x.permute(0,2,1)).permute(0,2,1)
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


def train(x, encoder_model, contrast_model, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z1 = encoder_model(x[0])
    z2 = encoder_model(x[1])
    loss = contrast_model(z1, z2)
    loss.backward()
    optimizer.step()
    return loss.item(), z1.shape[1]

def test(encoder_model,test_data1,test_data2,num_winds,batch_size,device):
    encoder_model.eval()
    with torch.no_grad():
        outputs1 = []
        for data in test_data1:
            data = data.to(device)
            outputs = []
            for i in range(0, data.size(0), batch_size):
                batch = data[i:i + batch_size]
                outputs.append(encoder_model(batch))#.cpu())
            outputs1.append(torch.cat(outputs, dim=0))
        outputs1 = torch.stack(outputs1)
        
        outputs2 = []
        for data in test_data2:
            data = data.to(device)
            outputs = []
            for i in range(0, data.size(0), batch_size):
                batch = data[i:i + batch_size]
                outputs.append(encoder_model(batch))#.cpu())
            outputs2.append(torch.cat(outputs, dim=0))
        outputs2 = torch.stack(outputs2)
        
        #outputs1 = outputs1.numpy()
        #outputs2 = outputs2.numpy()
        
        accuracies_all = []
        mean_accs = []
        std_accs= []
        corr_coeffs_all = []
        num_real = int(outputs1.shape[1] / num_winds)
        print('')
        for i in range (num_winds):
            for n in range(num_winds):
                if n >= i:
                    accuracies = []
                    for j in range(num_real):
                        corr_coeffs = torch.corrcoef(torch.cat([outputs1[:, i*num_real+j, :],outputs2[:, n*num_real+j, :]],dim=0))[0:outputs1.shape[0],outputs1.shape[0]:]#np.corrcoef(outputs1[:, i*num_real+j, :], outputs2[:, n*num_real+j, :])[0:outputs1.shape[0],outputs1.shape[0]:]
                        corr_coeffs_all.append(corr_coeffs)
                        lower_indices = torch.tril_indices(corr_coeffs.shape[0],corr_coeffs.shape[1], offset=-1)
                        upper_indices = torch.triu_indices(corr_coeffs.shape[0],corr_coeffs.shape[1], offset=1)
                        corr_coeffs1 = corr_coeffs.clone()
                        corr_coeffs2 = corr_coeffs.clone()
                        corr_coeffs1[lower_indices[0],lower_indices[1]] = -2
                        corr_coeffs2[upper_indices[0],upper_indices[1]] = -2
                        counter1 = 0
                        counter2 = 0
                        for j in range(corr_coeffs1.shape[0]):
                            if torch.argmax(corr_coeffs1[j, :]) == j:
                                counter1 += 1
                        for j in range(corr_coeffs2.shape[1]):
                            if torch.argmax(corr_coeffs2[:, j]) == j:
                                counter2 += 1
            
                        total_samples = outputs1.shape[0] + outputs2.shape[0]
                        accuracies.append((counter1 + counter2) / total_samples)
                    mean_acc = np.mean(accuracies)
                    std_acc = np.std(accuracies)
                    print(f'meanAcc{i,n}: {mean_acc:.2f}, stdAcc{i,n}: {std_acc:.3f}')
                    mean_accs.append(mean_acc)
                    std_accs.append(std_acc)
                    accuracies_all.append(accuracies)
        print('')
    return accuracies_all, mean_accs, std_accs, corr_coeffs_all


def main(config):

    path = config['path_data']

    names = []
    with open(os.path.join(path,'names_train.txt'), 'r') as f:
        for line in f:
            names.append(line.strip())
            
    data = np.load(os.path.join(path,'train_data_HCP_' + config['atlas'] + '_resampled.npz'))
    train_data = []
    for key in data:
        train_data.append(data[key])
    
    data = np.load(os.path.join(path,'test_data_HCP_' + config['atlas'] + '_1_resampled.npz'))
    test_data1 = []
    for key in data:
        test_data1.append(data[key])
    
    data = np.load(os.path.join(path,'test_data_HCP_' + config['atlas'] + '_2_resampled.npz'))
    test_data2 = []
    for key in data:
        test_data2.append(data[key])
            
    val_data1 = test_data1[:200]
    val_data2 = test_data2[:200]
    test_data1 = test_data1[200:]
    test_data2 = test_data2[200:]      

    roi_num = test_data1[0].shape[1]
    
    
    device = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu")
    batch_size = config['batch_size']
    shuffle = config['shuffle']
    epochs = config['epochs']
    lr = config['lr']
    tau = config['tau']
    train_length_limits = config['train_length_limits']
    max_length = train_length_limits[-1]
    test_winds = config['test_lengths']
    eval_epochs = config['eval_epochs']
    save_results = config['save_results']
    
    for i,data in enumerate(test_data1):
        data = torch.from_numpy(data.astype(np.float32))
        data = test_augment(data, config['test_lengths'], config['num_test_winds'], max_length)
        test_data1[i] = data
        
    for i,data in enumerate(test_data2):
        data = torch.from_numpy(data.astype(np.float32))
        data = test_augment(data, config['test_lengths'], config['num_test_winds'], max_length)
        test_data2[i] = data
        
    for i,data in enumerate(val_data1):
        data = torch.from_numpy(data.astype(np.float32))
        data = test_augment(data, config['test_lengths'], config['num_test_winds'], max_length)
        val_data1[i] = data
        
    for i,data in enumerate(val_data2):
        data = torch.from_numpy(data.astype(np.float32))
        data = test_augment(data, config['test_lengths'], config['num_test_winds'], max_length)
        val_data2[i] = data
        
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle = shuffle)

    
    model_config = config['model_config']
    model_config['max_length'] = max_length
  
    encoder_model = SeqenceModel(model_config, roi_num).to(device)
    contrast_model = DualBranchContrast(loss=InfoNCE(tau=tau),mode='L2L').to(device)   
    optimizer = Adam(encoder_model.parameters(), lr=lr)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_start_lr = 1e-5,
        warmup_epochs=config['warm_up_epochs'],
        max_epochs=epochs)
              
    max_val_acc = 0
    losses = []
    res_all = []
    count = 0
    with tqdm(total=epochs, desc='(T)') as pbar:
        for epoch in range(1,epochs+1):
            total_loss = 0.0
            batch_count = 0                              
            for batch_idx, sample_inds in enumerate(train_loader.batch_sampler):
                sample_inds = removeDuplicates(names,sample_inds)
                batch_list = [train_data[i] for i in sample_inds]
                batch_loader = DataLoader(batch_list, batch_size=len(batch_list))
                batch_data = next(iter(batch_loader))
                batch_data = augment_hcp_old(batch_data,train_length_limits,device)
                loss,input_dim = train(batch_data,encoder_model,contrast_model,optimizer)
                total_loss += loss
                batch_count += 1
            scheduler.step()
            average_loss = total_loss / batch_count if batch_count > 0 else float('nan')
            losses.append(average_loss)
            pbar.set_postfix({'loss': average_loss})
            pbar.update()        
            
            if epoch in eval_epochs:
                res = test(encoder_model,val_data1,val_data2,
                           len(test_winds),batch_size,device)
                res_all.append(res)
                if np.mean(res[1]) + np.min(res[1]) > max_val_acc:
                    max_val_acc = np.mean(res[1]) + np.min(res[1])
                    max_val_acc_model = copy.deepcopy(encoder_model.state_dict())
                else:
                    if epoch > 5:
                        count += 1
            if count >= 8:
                print('Early stopping')
                break
            print('')

    max_val_acc_encoder_model = SeqenceModel(model_config, roi_num).to(device)
    max_val_acc_encoder_model.load_state_dict(max_val_acc_model)
    val_result = test(max_val_acc_encoder_model, val_data1, val_data2,
                                len(test_winds), batch_size,device)
    test_result = test(max_val_acc_encoder_model, test_data1, test_data2,
                                len(test_winds), batch_size,device)
    results = {}
    results['losses'] = losses
    results['val_result'] = val_result
    results['test_result'] = test_result
    results['val_results_all'] = res_all
    
    
    if save_results:
        if not os.path.exists(os.path.join('ablations','results')):
            os.mkdir(os.path.join('ablations','results'))   
        with open(os.path.join('ablations','results','test_results_' + config['atlas'] + '_VarCoNetV2_noCNN.pkl'), 'wb') as f:
            pickle.dump(results,f)
    return results


if __name__ == '__main__':
    config = {}
    config['path_data'] = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/HCP'
    config['atlas'] = 'AAL' #AICHA, AAL
    with open('best_params_VarCoNet_v2_final_' + config['atlas'] + '.pkl','rb') as f:
        best_params = pickle.load(f)
    config['batch_size'] = best_params['batch_size'] 
    config['train_length_limits'] = [80,320] 
    config['test_lengths'] = [80,200,320]
    config['num_test_winds'] = 10
    config['shuffle'] = True
    config['epochs'] = 100
    config['tau'] = best_params['tau']
    config['lr'] = best_params['lr']
    config['warm_up_epochs'] = 10
    config['eval_epochs'] = list(range(1,config['epochs']+1))
    config['model_config'] = {}
    config['model_config']['layers'] = best_params['layers']
    config['model_config']['n_heads'] = best_params['n_heads']
    config['model_config']['dim_feedforward'] = best_params['dim_feedforward'] 
    config['device'] = "cuda:1"
    config['save_results'] = True
    results = main(config)

