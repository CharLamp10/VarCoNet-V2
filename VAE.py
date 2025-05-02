parent_path = r'/home/student1/Desktop/Charalampos_Lamprou/VarCoNetV2_extras/ksvd-sparse-dictionary'
import sys
sys.path.append(parent_path)
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import os
import pickle
from utils import test_augment, augment_VAE, PCC
from ksvd import ksvd
from sparseRep import random_sensor, sense, reconstruct
import copy

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD


class VAE(nn.Module):
    def __init__(self, input_dim=64620, hidden_dims=[3000, 2000, 1000, 500, 100]):
        super(VAE, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.Tanh(),
            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.BatchNorm1d(hidden_dims[3]),
            nn.Tanh(),
        )

        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[3], hidden_dims[4])
        self.fc_logvar = nn.Linear(hidden_dims[3], hidden_dims[4])

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[4], hidden_dims[3]),
            nn.BatchNorm1d(hidden_dims[3]),
            nn.Tanh(),
            nn.Linear(hidden_dims[3], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.Tanh(),
            nn.Linear(hidden_dims[2], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], input_dim),
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)  # Scaled variance initialization
                torch.nn.init.zeros_(m.bias)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar    


class Model(nn.Module):

    def __init__(self, roi_num):
        super().__init__()

        self.vae = VAE(input_dim=int((roi_num*(roi_num-1))/2))
        
    def forward(self, x, train):
        y,mu,logvar = self.vae(x)
        if train:
            return y,mu,logvar 
        else:
            return y


def train(x, model, optimizer):
    model.train()
    optimizer.zero_grad()
    z,mu,logvar = model(x,True)
    loss = loss_function(x,z,mu,logvar)
    loss.backward()
    optimizer.step()
    return z,loss.item()

def test(model,test_data1,test_data2,D,num_winds,batch_size,device):
    model.eval()
    with torch.no_grad():
        outputs1 = []
        for data in test_data1:
            data = data.to(device)
            outputs = []
            for i in range(0, data.size(0), batch_size):
                batch = data[i:i + batch_size]
                outputs.append(model(batch,False))
            outputs1.append(torch.cat(outputs, dim=0))
        outputs1 = torch.stack(outputs1)
        
        outputs2 = []
        for data in test_data2:
            data = data.to(device)
            outputs = []
            for i in range(0, data.size(0), batch_size):
                batch = data[i:i + batch_size]
                outputs.append(model(batch,False))
            outputs2.append(torch.cat(outputs, dim=0))
        outputs2 = torch.stack(outputs2)
        
        test_data1 = torch.stack(test_data1)
        test_data2 = torch.stack(test_data2)
        
        outputs1 = outputs1 - test_data1.to(device)
        outputs2 = outputs2 - test_data2.to(device)
        N,M,K = outputs1.shape
        
        outputs = torch.cat((outputs1, outputs2), dim=0)
        outputs = outputs.reshape(-1, K)
        sensor = random_sensor((K, int(0.4*K)), device=device)
        representation = sense(outputs, sensor, device)
        r = reconstruct(representation.T, sensor, D, 25, device).T
        outputs = outputs.cpu().numpy() - r.cpu().numpy()
        del test_data1, test_data2, outputs1, outputs2, sensor, representation, r
        torch.cuda.empty_cache()
        outputs = outputs.reshape(2*N, M, K)
        outputs1 = outputs[:int(outputs.shape[0]/2)]
        outputs2 = outputs[int(outputs.shape[0]/2):]
        
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
                        corr_coeffs = np.corrcoef(outputs1[:, i*num_real+j, :], outputs2[:, n*num_real+j, :])[0:outputs1.shape[0],outputs1.shape[0]:]
                        corr_coeffs_all.append(corr_coeffs)
                        lower_indices = np.tril_indices(corr_coeffs.shape[0], k=-1)
                        upper_indices = np.triu_indices(corr_coeffs.shape[0], k=1)
                        corr_coeffs1 = corr_coeffs.copy()
                        corr_coeffs2 = corr_coeffs.copy()
                        corr_coeffs1[lower_indices] = -2
                        corr_coeffs2[upper_indices] = -2
                        counter1 = 0
                        counter2 = 0
                        for j in range(corr_coeffs1.shape[0]):
                            if np.argmax(corr_coeffs1[j, :]) == j:
                                counter1 += 1
                        for j in range(corr_coeffs2.shape[1]):
                            if np.argmax(corr_coeffs2[:, j]) == j:
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
    train_length_limits = config['train_length_limits']
    max_length = train_length_limits[-1]
    test_winds = config['test_lengths']
    save_models = config['save_models']
    save_results = config['save_results']
    load_prev_model = config['load_prev_model']
    
    for i,data in enumerate(test_data1):
        data = torch.from_numpy(data.astype(np.float32))
        data = test_augment(data, config['test_lengths'], config['num_test_winds'], max_length)
        corrs = []
        for dat in data:
            zero_rows = torch.all(dat == 0, axis=1)
            zero_row_indices = torch.where(zero_rows)[0]
            if len(zero_row_indices) > 0:
                dat = dat[:torch.min(zero_row_indices),:]
            corr = PCC(dat.T)
            triu_indices = torch.triu_indices(corr.shape[0], corr.shape[0], offset=1)
            corrs.append(corr[triu_indices[0], triu_indices[1]])
        test_data1[i] = torch.stack(corrs)
        
    for i,data in enumerate(test_data2):
        data = torch.from_numpy(data.astype(np.float32))
        data = test_augment(data, config['test_lengths'], config['num_test_winds'], max_length)
        corrs = []
        for dat in data:
            zero_rows = torch.all(dat == 0, axis=1)
            zero_row_indices = torch.where(zero_rows)[0]
            if len(zero_row_indices) > 0:
                dat = dat[:torch.min(zero_row_indices),:]
            corr = PCC(dat.T)
            triu_indices = torch.triu_indices(corr.shape[0], corr.shape[0], offset=1)
            corrs.append(corr[triu_indices[0], triu_indices[1]])
        test_data2[i] = torch.stack(corrs)
        
    for i,data in enumerate(val_data1):
        data = torch.from_numpy(data.astype(np.float32))
        data = test_augment(data, config['test_lengths'], config['num_test_winds'], max_length)
        corrs = []
        for dat in data:
            zero_rows = torch.all(dat == 0, axis=1)
            zero_row_indices = torch.where(zero_rows)[0]
            if len(zero_row_indices) > 0:
                dat = dat[:torch.min(zero_row_indices),:]
            corr = PCC(dat.T)
            triu_indices = torch.triu_indices(corr.shape[0], corr.shape[0], offset=1)
            corrs.append(corr[triu_indices[0], triu_indices[1]])
        val_data1[i] = torch.stack(corrs)
        
    for i,data in enumerate(val_data2):
        data = torch.from_numpy(data.astype(np.float32))
        data = test_augment(data, config['test_lengths'], config['num_test_winds'], max_length)
        corrs = []
        for dat in data:
            zero_rows = torch.all(dat == 0, axis=1)
            zero_row_indices = torch.where(zero_rows)[0]
            if len(zero_row_indices) > 0:
                dat = dat[:torch.min(zero_row_indices),:]
            corr = PCC(dat.T)
            triu_indices = torch.triu_indices(corr.shape[0], corr.shape[0], offset=1)
            corrs.append(corr[triu_indices[0], triu_indices[1]])
        val_data2[i] = torch.stack(corrs)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle = shuffle)
   
    model = Model(roi_num).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ExponentialLR(optimizer, gamma=0.9991)
              
    test_result = []
    all_batch_data = []
    all_z = []
    if not load_prev_model:
        with tqdm(total=epochs, desc='(T)') as pbar:
            for epoch in range(1,epochs+1):
                total_loss = 0.0
                batch_count = 0                              
                for batch_data in train_loader:
                    batch_data = augment_VAE(batch_data,train_length_limits[0],train_length_limits[1],device)
                    if epoch == epochs:
                        z,loss = train(batch_data,model,optimizer)
                        all_batch_data.append(batch_data)
                        all_z.append(z)
                    else:
                        _,loss = train(batch_data,model,optimizer)
                    total_loss += loss
                    batch_count += 1
                average_loss = total_loss / batch_count if batch_count > 0 else float('nan')
                pbar.set_postfix({'loss': average_loss})
                pbar.update()   
                scheduler.step()
        model_state_dict = copy.deepcopy(model.state_dict())
        if save_models:        
            if not os.path.exists(os.path.join('models','VAE')):
                os.makedirs(os.path.join('models','VAE'), exist_ok=True)        
            torch.save(model_state_dict, os.path.join('models','VAE','min_loss_model_' + config['atlas'] + '.pth'))
    else:
        model_state_dict = torch.load(os.path.join('models','VAE','min_loss_model_' + config['atlas'] + '.pth'))
        model = Model(roi_num).to(device)
        model.load_state_dict(model_state_dict)
        model.eval()
        for batch_data in train_loader:
            batch_data = augment_VAE(batch_data,train_length_limits[0],train_length_limits[1],device)
            z = model(batch_data, False)
            all_batch_data.append(batch_data)
            all_z.append(z)
    if not load_prev_model:
        r = torch.cat(all_z) - torch.cat(all_batch_data)
        D,x,_ = ksvd(r.detach().cpu().numpy(), 100, 25, maxiter=50,device=config['device'])
        if save_models:        
            if not os.path.exists(os.path.join('models','VAE')):
                os.makedirs(os.path.join('models','VAE'), exist_ok=True)        
            torch.save(D, os.path.join('models','VAE','dictionary_' + config['atlas'] + '.pt'))
    else:
        D = torch.load(os.path.join('models','VAE','dictionary_' + config['atlas'] + '.pt'))
    
    val_result = test(model, val_data1, val_data2, D, len(test_winds), batch_size, device)
    test_result = test(model, test_data1, test_data2, D, len(test_winds), batch_size, device)
    results = {}
    results['result_val'] = val_result
    results['result_test'] = test_result
    
    if save_results:
        if not os.path.exists('results'):
            os.mkdir('results')   
        with open(os.path.join('results','test_results_' + config['atlas'] + '_VAE.pkl'), 'wb') as f:
            pickle.dump(results,f)
    return results


if __name__ == '__main__':
    config = {}
    config['path_data'] = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/HCP'
    config['atlas'] = 'AICHA' #AICHA, AAL
    config['batch_size'] = 128 
    config['train_length_limits'] = [80,320] 
    config['test_lengths'] = [80,200,320]
    config['num_test_winds'] = 10
    config['shuffle'] = True
    config['epochs'] = 300
    config['lr'] = 1e-4
    config['device'] = "cuda:1"
    config['save_models'] = False
    config['save_results'] = True
    config['load_prev_model'] = True
    results = main(config)

