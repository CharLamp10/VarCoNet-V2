from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from utils import InfoNCE
from tqdm import tqdm
from torch.optim import Adam
from utils import DualBranchContrast
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from utils import ABIDEDataset
import os
from classifier import LREvaluator
import pickle
import math
from sklearn.model_selection import train_test_split, StratifiedKFold
from utils import upper_triangular_cosine_similarity, augment, removeDuplicates, test_augment_overlap


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
        self.pos_enc = PositionalEncodingTrainable(d_model,max_len)
        self.bn = nn.InstanceNorm1d(d_model)
        
        
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


def test(encoder_model, train_loader, val_loader, test_loader, min_length, max_length, num_classes, device, num_epochs, lr):
    encoder_model.eval()
    with torch.no_grad():
        outputs_train = []
        y_train = []
        for (x,y) in train_loader:
            xs = []
            ys = []
            for (xi,yi) in zip(x,y):
                xs.append(xi.unsqueeze(0).to(device))
                ys.append(yi)
                xi = test_augment_overlap(xi,[4,3,2,1],0.5,min_length,max_length,device)
                xs.append(xi)
                for i in range(xi.shape[0]):
                    ys.append(yi)
            xs = torch.cat(xs)
            xs = xs.to(device)
            y_train.append(torch.tensor(ys))
            outputs_train.append(encoder_model(xs))
        outputs_train = torch.cat(outputs_train, dim=0).clone().detach()
        y_train = torch.cat(y_train,dim=0).to(device)
        
        outputs_val = []
        y_val = []
        for (x,y) in val_loader:
            x = x.to(device)
            y_val.append(y)
            outputs_val.append(encoder_model(x))
        outputs_val = torch.cat(outputs_val, dim=0).clone().detach()
        y_val = torch.cat(y_val,dim=0).to(device)
        
        outputs_test= []
        y_test = []
        for (x,y) in test_loader:
            x = x.to(device)
            y_test.append(y)
            outputs_test.append(encoder_model(x))
        outputs_test = torch.cat(outputs_test, dim=0).clone().detach()
        y_test = torch.cat(y_test,dim=0).to(device)
        
    result,linear_state_dict = LREvaluator(num_epochs = num_epochs,learning_rate=lr).evaluate(encoder_model, outputs_train, y_train, outputs_val, y_val, outputs_test, y_test, num_classes, device)
                
    return result,linear_state_dict


def main(config):

    path = config['path_data']
    
    names = []
    with open(os.path.join(path,'ABIDE_nilearn_names.txt'), 'r') as f:
        for line in f:
            names.append(line.strip())
    
    data_list = np.load(os.path.join(path,'ABIDE_nilearn_' + config['atlas'] + '.npz'))
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
    y = np.load(os.path.join(path,'ABIDE_nilearn_classes.npy'))   
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
    
    pos = []
    for name in names:
        pos.append(np.where(np.array(names_unique) == name)[0][0])
    data = [data[i] for i in pos]
    y = y[pos]
    
    device = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu")
    batch_size = config['batch_size']
    shuffle = config['shuffle']
    tau = config['tau']
    epochs = config['epochs']
    lr = config['lr']
    min_length = config['min_length']
    max_length = data[0].shape[0]
    train_length_limits = [min_length, max_length]
    epochs_cls = config['epochs_cls']
    lr_cls = config['lr_cls']
    num_classes = config['num_classes']
    eval_epochs = list(range(1, epochs+1)) 
    save_results = config['save_results'] 
    
    model_config = config['model_config'] 
    model_config['max_length'] = config['max_length']
    
    '''------------------------------------KFold CV------------------------------------'''
    
    losses_all = []
    test_result_all = []
    min_val_loss_epochs = []
    min_loss_epochs = []
    names_train_all = []
    names_val_all = []
    names_test_all = []
    for i in range(10):
        skf = StratifiedKFold(n_splits=10, shuffle = True, random_state=42+i)
        for j, (train_index, test_index) in enumerate(skf.split(data, y)):
            train_data = [data[i] for i in train_index]
            test_data = [data[n] for n in test_index]
            y_train = y[train_index]
            y_test = y[test_index]
            names_train = [names[n] for n in train_index]
            names_test = [names[n] for n in test_index]
            train_data, val_data, y_train, y_val, train_idx, val_idx = train_test_split(train_data, y_train, np.arange(len(train_data)), test_size=0.15, random_state=42, stratify=y_train)
            names_val = [names_train[n] for n in val_idx]
            names_train = [names_train[n] for n in train_idx]
            train_data = train_DATA + train_data
            y_train = np.concatenate((Y_train, y_train))
            names_train = names_duplicate + names_train
            train_dataset = ABIDEDataset(train_data, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = shuffle, num_workers=4)
            val_dataset = ABIDEDataset(val_data, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
            test_dataset = ABIDEDataset(test_data, y_test)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)  
            names_train_all.append(names_train)
            names_val_all.append(names_val)
            names_test_all.append(names_test)
            
            roi_num = test_data[0].shape[1]
            encoder_model = SeqenceModel(model_config, roi_num).to(device)
            contrast_model = DualBranchContrast(loss=InfoNCE(tau=tau),mode='L2L').to(device)
            optimizer = Adam(encoder_model.parameters(), lr=lr)
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer=optimizer,
                warmup_start_lr = 1e-5,
                warmup_epochs=config['warm_up_epochs'],
                max_epochs=epochs)
                  
            min_val_loss = 1000
            test_result = []
            losses = []
            with tqdm(total=epochs, desc='(T)') as pbar:
                for epoch in range(1,epochs+1):
                    total_loss = 0.0
                    batch_count = 0                
                    for batch_idx, sample_inds in enumerate(train_loader.batch_sampler):
                        sample_inds = removeDuplicates(names_train,sample_inds)
                        batch_list = [train_data[i] for i in sample_inds]
                        batch_loader = DataLoader(batch_list, batch_size=len(batch_list), num_workers=4)
                        batch_data = next(iter(batch_loader))
                        batch_data = augment(batch_data,train_length_limits,max_length,device)
                        loss,input_dim = train(batch_data,encoder_model,contrast_model,
                                               optimizer)
                        total_loss += loss
                        batch_count += 1
                    scheduler.step()
                            
                    average_loss = total_loss / batch_count if batch_count > 0 else float('nan')   
                    losses.append(average_loss)
                    pbar.set_postfix({'loss': average_loss})
                    pbar.update()        
                    
                    if epoch in eval_epochs:
                        res,linear_state_dict = test(encoder_model,train_loader,val_loader,test_loader,
                                   min_length,max_length,num_classes,device,epochs_cls,lr_cls)
                        test_result.append(res) 
                        if res['best_val_loss'] < min_val_loss:
                            min_val_loss = res['best_val_loss']
                            min_val_loss_epoch = epoch
            losses_all.append(losses)
            test_result_all.append(test_result)
            min_val_loss_epochs.append(min_val_loss_epoch)
    
    
    results = {}
    results['losses'] = losses_all
    results['epoch_results'] = test_result_all
    results['min_val_loss_epoch'] = min_val_loss_epochs
    results['min_loss_epoch'] = min_loss_epochs
    results['names_train'] = names_train_all
    results['names_val'] = names_val_all
    results['names_test'] = names_test_all
    
    if save_results:          
        if not os.path.exists(os.path.join('ablations','results_ABIDE',config['atlas'])):
            os.makedirs(os.path.join('ablations','results_ABIDE',config['atlas']),exist_ok=True)
        with open(os.path.join('ablations','results_ABIDE',config['atlas'],'ABIDE_VarCoNetV2_noCNN_results.pkl'), 'wb') as f:
            pickle.dump(results,f)
    return results


if __name__ == '__main__':
    config = {}
    config['path_data'] = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/ABIDE/fmriprep'
    config['atlas'] = 'AAL' #AICHA, AAL
    with open('best_params_VarCoNet_v2_final_' + config['atlas'] + '.pkl','rb') as f:
        best_params = pickle.load(f)
    config['min_length'] = 80
    config['max_length'] = 320
    config['batch_size'] = best_params['batch_size']
    config['shuffle'] = True
    config['tau'] = best_params['tau']
    config['epochs'] = 50                           #epochs for the contrastive learning
    config['warm_up_epochs'] = 10                   #warm-up epochs for the contrastive learning scheduler
    config['lr'] = best_params['lr']                #learning rate for the contrastive learning
    config['epochs_cls'] = 150                      #epochs for the linear layer (classification)
    config['lr_cls'] = 5e-5                         #learning rate for the linear layer (classification)
    config['num_classes'] = 2
    config['model_config'] = {}
    config['model_config']['layers'] = best_params['layers']
    config['model_config']['n_heads'] = best_params['n_heads']
    config['model_config']['dim_feedforward'] = best_params['dim_feedforward']    
    config['device'] = "cuda:0"
    config['save_results'] = True
    results = main(config)
