from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, GRU, Conv1d, MaxPool1d
from tqdm import tqdm
from torch.optim import Adam
from utils import ABIDEDataset
import os
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import copy


class GruKRegion(nn.Module):

    def __init__(self, kernel_size=128, layers=4, out_size=8, dropout=0.5):
        super().__init__()
        self.gru = GRU(1, kernel_size, layers,
                       bidirectional=True, batch_first=True)

        self.kernel_size = kernel_size

        self.linear = nn.Sequential(
            nn.Dropout(dropout),
            Linear(kernel_size*2, kernel_size),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(kernel_size, out_size)
        )

    def forward(self, raw):
        b, k, d = raw.shape
        x = raw.contiguous().view((b*k, 1, d))
        x = torch.permute(x, (0,2,1))
        x, h = self.gru(x)
        x = x[:, -1, :]
        x = x.view((b, k, -1))
        x = self.linear(x)
        return x


class ConvKRegion(nn.Module):

    def __init__(self, k=1, out_size=8, kernel_size=8, pool_size=16, time_series=512):
        super().__init__()
        self.conv1 = Conv1d(in_channels=k, out_channels=32,
                            kernel_size=kernel_size, stride=2)

        output_dim_1 = (time_series-kernel_size)//2+1

        self.conv2 = Conv1d(in_channels=32, out_channels=32,
                            kernel_size=8)
        output_dim_2 = output_dim_1 - 8 + 1
        self.conv3 = Conv1d(in_channels=32, out_channels=16,
                            kernel_size=8)
        output_dim_3 = output_dim_2 - 8 + 1
        self.max_pool1 = MaxPool1d(pool_size)
        output_dim_4 = output_dim_3 // pool_size * 16
        self.in0 = nn.InstanceNorm1d(time_series)
        self.in1 = nn.BatchNorm1d(32)
        self.in2 = nn.BatchNorm1d(32)
        self.in3 = nn.BatchNorm1d(16)

        self.linear = nn.Sequential(
            Linear(output_dim_4, 32),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(32, out_size)
        )

    def forward(self, x):

        b, k, d = x.shape

        x = torch.transpose(x, 1, 2)

        x = self.in0(x)

        x = torch.transpose(x, 1, 2)
        x = x.contiguous()

        x = x.view((b*k, 1, d))

        x = self.conv1(x)

        x = self.in1(x)
        x = self.conv2(x)

        x = self.in2(x)
        x = self.conv3(x)

        x = self.in3(x)
        x = self.max_pool1(x)

        x = x.view((b, k, -1))

        x = self.linear(x)

        return x


class SeqenceModel(nn.Module):

    def __init__(self, model_config, roi_num=384, time_series=120):
        super().__init__()

        if model_config['extractor_type'] == 'cnn':
            self.extract = ConvKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                time_series=time_series, pool_size=4, )
        elif model_config['extractor_type'] == 'gru':
            self.extract = GruKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                layers=model_config['num_gru_layers'], dropout=model_config['dropout'])
        self.linear = nn.Sequential(
            nn.Linear(roi_num*model_config['embedding_size'], 256),
            nn.ReLU(),
            nn.Linear(256,32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1))

    def forward(self, x):
        x = self.extract(x)
        x = x.view(x.shape[0],-1)
        x = self.linear(x)
        return x
    
def train(x, y, encoder_model, optimizer, loss_func):
    encoder_model.train()
    optimizer.zero_grad()
    z = encoder_model(x)
    loss = loss_func(z, F.one_hot(y, num_classes=2).float())
    loss.backward()
    optimizer.step()
    z = z[:,-1].detach().cpu().numpy()
    y = y.to(torch.device("cpu")).numpy()
    auc_score = roc_auc_score(y, z)
    return loss.item(),auc_score


def test(encoder_model, test_data_loader, batch_size, loss_func, device):
    encoder_model.eval()
    with torch.no_grad():
        zs = []
        ys = []
        for (x,y) in test_data_loader:
            zs.append(encoder_model(x.to(device)))
            ys.append(y)
        z = torch.cat(zs,dim=0)
        y = torch.cat(ys,dim=0)
        loss = loss_func(z, F.one_hot(y, num_classes=2).float().to(device))
        z = z[:,-1].cpu().numpy()
        y = y.numpy()
        auc_score = roc_auc_score(y, z)
                   
    return loss.item(), auc_score, z, y


def main(config):
    path = config['path_data']
           
    names = []
    with open(os.path.join(path,'ABIDE_nilearn_names.txt'), 'r') as f:
        for line in f:
            names.append(line.strip())
    
    data_list = np.load(os.path.join(path,'ABIDE_nilearn_' + config['atlas'] + '.npz'))
    data = []
    for key in data_list:
        data.append(data_list[key][:120,:].T)
    
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
    
     
    device = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu")
    batch_size = config['batch_size']
    shuffle = config['shuffle']
    epochs = config['epochs']
    lr = config['lr']
    save_models = config['save_models']
    save_results = config['save_results']   
    
    model_config = config['model_config'] 
    model_config['max_length'] = config['max_length']
    
    '''------------------------------------KFold CV------------------------------------'''
    test_losses_all = []
    test_aucs_all = []
    train_losses = []
    val_losses_all = []
    val_aucs_all = []
    val_probs_all = []
    y_val_all = []
    y_test_all = []
    test_probs_all = []
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
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = shuffle)
            val_dataset = ABIDEDataset(val_data, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_dataset = ABIDEDataset(test_data, y_test)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)  
            names_train_all.append(names_train)
            names_val_all.append(names_val)
            names_test_all.append(names_test)
            
            roi_num = test_data[0].shape[0]
            encoder_model = SeqenceModel(model_config, roi_num).to(device)
            loss_func = nn.BCELoss()
            optimizer = Adam(encoder_model.parameters(), lr=lr)
            
            min_val_loss = 1000
            losses = []
            val_losses = []
            test_losses = []
            aucs = []
            val_aucs = []
            test_aucs = []
            val_probs = []
            test_probs = []
            y_vals = []
            y_tests = []
            with tqdm(total=epochs, desc='(T)') as pbar:
                for epoch in range(1,epochs+1):
                    total_loss = 0.0
                    total_auc = 0.0
                    batch_count = 0                          
                    for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):            
                        loss,auc = train(batch_data.to(device), batch_labels.to(device), encoder_model, optimizer, loss_func)
                        total_loss += loss
                        total_auc += auc
                        batch_count += 1
                    val_loss,val_auc,val_prob,y_val = test(encoder_model,val_loader,batch_size,loss_func,device)
                            
                    average_loss = total_loss / batch_count if batch_count > 0 else float('nan')   
                    average_auc = total_auc / batch_count if batch_count > 0 else float('nan') 
                    losses.append(average_loss)
                    val_losses.append(val_loss)
                    aucs.append(average_auc)
                    val_aucs.append(val_auc)
                    val_probs.append(val_prob)
                    y_vals.append(y_val)
                    pbar.set_postfix({
                        'loss': average_loss, 
                        'auc': average_auc,
                        'val_loss': val_loss, 
                        'val_auc': val_auc
                    })
                    pbar.update()  
                    if val_loss < min_val_loss:
                        min_val_loss_model = copy.deepcopy(encoder_model.state_dict())
                    test_loss,test_auc,test_prob,y_test = test(encoder_model,test_loader,batch_size,loss_func,device)
                    test_losses.append(test_loss)
                    test_aucs.append(test_auc)
                    test_probs.append(test_prob)
                    y_tests.append(y_test)
            
            test_losses_all.append(test_losses)
            test_aucs_all.append(test_aucs)
            train_losses.append(losses)
            val_losses_all.append(val_losses)
            val_aucs_all.append(val_aucs)
            val_probs_all.append(val_probs)
            test_probs_all.append(test_probs)
            y_val_all.append(y_vals)
            y_test_all.append(y_tests)
            
            if save_models:
                if not os.path.exists(os.path.join('models_ABIDE',config['atlas'],'FBNET')):
                    os.makedirs(os.path.join('models_ABIDE',config['atlas'],'FBNET'), exist_ok=True)
                torch.save(min_val_loss_model, os.path.join('models_ABIDE',config['atlas'],'FBNET','ABIDE_min_val_loss_model_rs' + str(i) + '_fold' + str(j) + '.pth'))
    
    
    '''------------------------------------Ext. test------------------------------------'''
    ext_test_losses_all = []
    ext_test_aucs_all = []
    train_losses_ext = []
    val_losses_all_ext = []
    val_aucs_all_ext = []
    val_probs_all_ext = []
    y_val_all_ext = []
    ext_test_probs_all = []
    names_train_ext_all = []
    names_val_ext_all = []
    for i in range(10):
        train_data, val_data, y_train, y_val, train_idx, val_idx = train_test_split(data, y, np.arange(len(data)), test_size=0.1, random_state=42+i, stratify=y)
        names_val = [names[n] for n in val_idx]
        names_train = [names[n] for n in train_idx]
        train_data = train_DATA + train_data
        y_train = np.concatenate((Y_train, y_train))
        names_train = names_duplicate + names_train
        train_dataset = ABIDEDataset(train_data, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = shuffle)
        val_dataset = ABIDEDataset(val_data, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataset = ABIDEDataset(ext_test_data, y_ext_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)  
        names_train_ext_all.append(names_train)
        names_val_ext_all.append(names_val)
        
        roi_num = test_data[0].shape[0]
        encoder_model = SeqenceModel(model_config, roi_num).to(device)
        loss_func = nn.BCELoss()
        optimizer = Adam(encoder_model.parameters(), lr=lr)
        
        min_val_loss = 1000
        losses = []
        val_losses = []
        test_losses = []
        aucs = []
        val_aucs = []
        test_aucs = []
        val_probs = []
        test_probs = []
        y_vals = []
        y_tests = []
        with tqdm(total=epochs, desc='(T)') as pbar:
            for epoch in range(1,epochs+1):
                total_loss = 0.0
                total_auc = 0.0
                batch_count = 0                          
                for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):            
                    loss,auc = train(batch_data.to(device), batch_labels.to(device), encoder_model, optimizer, loss_func)
                    total_loss += loss
                    total_auc += auc
                    batch_count += 1
                val_loss,val_auc,val_prob,y_val = test(encoder_model,val_loader,batch_size,loss_func,device)
                        
                average_loss = total_loss / batch_count if batch_count > 0 else float('nan')   
                average_auc = total_auc / batch_count if batch_count > 0 else float('nan') 
                losses.append(average_loss)
                val_losses.append(val_loss)
                aucs.append(average_auc)
                val_aucs.append(val_auc)
                val_probs.append(val_prob)
                y_vals.append(y_val)
                pbar.set_postfix({
                    'loss': average_loss, 
                    'auc': average_auc,
                    'val_loss': val_loss, 
                    'val_auc': val_auc
                })
                pbar.update()  
                if val_loss < min_val_loss:
                    min_val_loss_model = copy.deepcopy(encoder_model.state_dict())
                test_loss,test_auc,test_prob,y_ext_test = test(encoder_model,test_loader,batch_size,loss_func,device)
                test_losses.append(test_loss)
                test_aucs.append(test_auc)
                test_probs.append(test_prob)
                y_tests.append(y_test)
        
        ext_test_losses_all.append(test_losses)
        ext_test_aucs_all.append(test_aucs)
        train_losses_ext.append(losses)
        val_losses_all_ext.append(val_losses)
        val_aucs_all_ext.append(val_aucs)
        val_probs_all_ext.append(val_probs)
        ext_test_probs_all.append(test_probs)
        y_val_all_ext.append(y_vals)
        
        if save_models:
            if not os.path.exists(os.path.join('models_ABIDE',config['atlas'],'FBNET')):
                os.makedirs(os.path.join('models_ABIDE',config['atlas'],'FBNET'), exist_ok=True)
            torch.save(min_val_loss_model, os.path.join('models_ABIDE',config['atlas'],'FBNET','ABIDE_min_val_loss_model_rs' + str(i) + '.pth'))
    
    results = {}
    results['losses'] = train_losses
    results['val_losses'] = val_losses_all
    results['test_losses'] = test_losses_all
    results['test_aucs'] = test_aucs_all
    results['val_aucs'] = val_aucs_all
    results['val_probs'] = val_probs_all
    results['test_probs'] = test_probs_all
    results['y_val'] = y_val_all
    results['y_test'] = y_test_all
    results['names_train'] = names_train_all
    results['names_val'] = names_val_all
    results['names_test'] = names_test_all
    results['losses_ext'] = train_losses_ext
    results['val_losses_ext'] = val_losses_all_ext
    results['ext_test_losses'] = ext_test_losses_all
    results['ext_test_aucs'] = ext_test_aucs_all
    results['val_aucs_ext'] = val_aucs_all_ext
    results['val_probs_ext'] = val_probs_all_ext
    results['ext_test_probs'] = ext_test_probs_all
    results['y_val_ext'] = y_val_all_ext
    results['y_ext_test'] = y_ext_test
    results['names_val_ext'] = names_val_ext_all
    results['names_ext_test'] = names_ext_test
    if save_results:            
        if not os.path.exists(os.path.join('results_ABIDE',config['atlas'])):
            os.makedirs(os.path.join('results_ABIDE',config['atlas']),exist_ok=True)
        
        with open(os.path.join('results_ABIDE',config['atlas'],'ABIDE_FBNET_results.pkl'), "wb") as pickle_file:
            pickle.dump(results, pickle_file)
    return results

if __name__ == '__main__':
    config = {}
    config['path_data'] = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/ABIDE/fmriprep'
    config['atlas'] = 'AAL' #AICHA, AAL
    config['max_length'] = 120
    config['batch_size'] = 128
    config['shuffle'] = True
    config['epochs'] = 250
    config['model_config'] = {}
    config['model_config']['extractor_type'] = 'cnn'
    if config['model_config']['extractor_type'] == 'cnn':
        config['model_config']['embedding_size'] = 8
        config['model_config']['window_size'] = 4
        config['lr'] = 2e-5
    elif config['model_config']['extractor_type'] == 'gru':
        config['model_config']['embedding_size'] = 8
        config['model_config']['window_size'] = 4
        config['model_config']['num_gru_layers'] = 4
        config['model_config']['dropout'] = 0.5 
        config['lr'] = 1e-4
    config['device'] = "cuda:2"
    config['save_models'] = True
    config['save_results'] = True
    results = main(config)

