parent_path = r'/home/student1/Desktop/Charalampos_Lamprou/VarCoNetV2_extras/ksvd-sparse-dictionary'
import sys
sys.path.append(parent_path)
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
import os
import pickle
from utils import test_augment_AE, augment_hcp, PCC
from ksvd import ksvd
from sparseRep import random_sensor, sense, reconstruct
import copy

def loss_function(x, x_hat):
    reproduction_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='mean')
    return reproduction_loss


class AE(nn.Module):
    def __init__(self, input_dim=80, hidden_dims=[500, 500, 2000], bottleneck=10):
        super(AE, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], bottleneck),
            nn.ReLU(),
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim)
        )  
        
    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
        
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class Model(nn.Module):

    def __init__(self, length):
        super().__init__()
        self.ae = AE(input_dim=length)
        self.length = length
        
    def forward(self, x):
        y = self.ae(x.permute(0,2,1).contiguous().view(-1,self.length))
        y = y.view(x.shape[0],x.shape[2],x.shape[1]).permute(0,2,1)
        return y


def train(x, model, optimizer):
    model.train()
    optimizer.zero_grad()
    z = model(x)
    loss = loss_function(x,z)
    loss.backward()
    optimizer.step()
    return z,loss.item()

def test(models,test_data1,test_data2,D,num_winds,batch_size,device):
    C = test_data1[0].shape[3]
    corrs = torch.zeros(2*test_data1[0].shape[0],len(test_data1)*test_data1[0].shape[1],int(C*(C-1)/2),dtype=torch.float32,device=device)
    for it,(model,test_dat1,test_dat2) in enumerate(zip(models,test_data1,test_data2)):
        model.eval()
        with torch.no_grad():
            outputs1 = []
            for data in test_dat1:
                data = data.to(device)
                outputs = []
                for i in range(0, data.size(0), batch_size):
                    batch = data[i:i + batch_size]
                    outputs.append(model(batch))
                outputs1.append(torch.cat(outputs, dim=0))
            
            outputs2 = []
            for data in test_dat2:
                data = data.to(device)
                outputs = []
                for i in range(0, data.size(0), batch_size):
                    batch = data[i:i + batch_size]
                    outputs.append(model(batch))
                outputs2.append(torch.cat(outputs, dim=0))
        outputs1 = torch.stack(outputs1)
        outputs2 = torch.stack(outputs2)
        
        del data, outputs
        torch.cuda.empty_cache()
        outputs1 = outputs1 - test_dat1.to(device)
        outputs2 = outputs2 - test_dat2.to(device)
        outputs = torch.cat((outputs1, outputs2), dim=0)
        del outputs2, outputs1
        torch.cuda.empty_cache()
        for i,output in enumerate(outputs):
            for j,out in enumerate(output):
                corr = PCC(out.T)
                triu_indices = torch.triu_indices(corr.shape[0], corr.shape[0], offset=1)
                corrs[i,j+it*test_data1[0].shape[1],:] = corr[triu_indices[0], triu_indices[1]]
    del test_data1, test_data2, outputs, test_dat1, test_dat2, corr, out, output
    torch.cuda.empty_cache()
    N,M,K = corrs.shape
    corrs = corrs.reshape(-1, K)
    sensor = random_sensor((K, int(0.4*K)), device=device)
    representation = sense(corrs, sensor, device)
    r = reconstruct(representation.T, sensor, D, 25, device).T
    corrs = corrs.cpu().numpy() - r.cpu().numpy()
    del sensor, representation, r
    torch.cuda.empty_cache()
    outputs = corrs.reshape(N, M, K)
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
            
    
    val_dat1 = test_data1[:200]
    val_dat2 = test_data2[:200]  
    test_dat1 = test_data1[200:]
    test_dat2 = test_data2[200:]      

    device = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu")
    batch_size = config['batch_size']
    shuffle = config['shuffle']
    epochs = config['epochs']
    lr = config['lr']
    weight_decay = config['weight_decay']
    train_test_lengths = config['train_test_lengths']
    test_winds = config['test_lengths']
    save_models = config['save_models']
    save_results = config['save_results']
    load_prev_model = config['load_prev_model']
    
    val_data1 = []
    val_data2 = []
    test_data1 = []
    test_data2 = []
    for length in train_test_lengths:
        val_d1 = []
        for i,data in enumerate(val_dat1):
            data = torch.from_numpy(data.astype(np.float32))
            data = test_augment_AE(data, [length], config['num_test_winds'], length)
            val_d1.append(data)
        val_data1.append(torch.stack(val_d1))
        
        val_d2 = []
        for i,data in enumerate(val_dat2):
            data = torch.from_numpy(data.astype(np.float32))
            data = test_augment_AE(data, [length], config['num_test_winds'], length)
            val_d2.append(data)
        val_data2.append(torch.stack(val_d2))
        
        test_d1 = []
        for i,data in enumerate(test_dat1):
            data = torch.from_numpy(data.astype(np.float32))
            data = test_augment_AE(data, [length], config['num_test_winds'], length)
            test_d1.append(data)
        test_data1.append(torch.stack(test_d1))
        
        test_d2 = []
        for i,data in enumerate(test_dat2):
            data = torch.from_numpy(data.astype(np.float32))
            data = test_augment_AE(data, [length], config['num_test_winds'], length)
            test_d2.append(data)
        test_data2.append(torch.stack(test_d2))
    
    if not load_prev_model:
        models = [Model(length).to(device) for length in train_test_lengths]
        optimizers = [SGD(models[i].parameters(), lr=lr, weight_decay=weight_decay, momentum=0.5) for i in range(len(train_test_lengths))]
        schedulers = [ExponentialLR(optimizers[i], gamma=0.9991) for i in range(len(train_test_lengths))]
    else:
        models = []
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle = shuffle) 
    corrs = []
    for i,length in enumerate(train_test_lengths):                  
        test_result = []
        all_batch_data = []
        all_z = []
        if not load_prev_model:
            with tqdm(total=epochs, desc='(T)') as pbar:
                for epoch in range(1,epochs+1):
                    total_loss = 0.0
                    batch_count = 0                              
                    for batch_data in train_loader:
                        batch_data = augment_hcp(batch_data,[length,length+1],device)
                        batch_data = batch_data[0][:,:-1,:]
                        if epoch == epochs:
                            z,loss = train(batch_data,models[i],optimizers[i])
                            all_batch_data.append(batch_data)
                            all_z.append(z)
                        else:
                            _,loss = train(batch_data,models[i],optimizers[i])
                        total_loss += loss
                        batch_count += 1
                    average_loss = total_loss / batch_count if batch_count > 0 else float('nan')
                    pbar.set_postfix({'loss': average_loss})
                    pbar.update()   
                    schedulers[i].step()
                    if epoch == 5:
                        for param_group in optimizers[i].param_groups:
                            param_group['momentum'] = 0.9
            model_state_dict = copy.deepcopy(models[i].state_dict())
        else:
            model_state_dict = torch.load(os.path.join('models','AE','min_loss_model_' + config['atlas'] + '_length' + str(length) + '.pth'))
            model = Model(length).to(device)
            model.load_state_dict(model_state_dict)
            models.append(model)
    if not load_prev_model:
        r = torch.cat(all_z) - torch.cat(all_batch_data)
        for dat in r:
            corr = PCC(dat.T)
            triu_indices = torch.triu_indices(corr.shape[0], corr.shape[0], offset=1)
            corrs.append(corr[triu_indices[0], triu_indices[1]])
        if save_models:        
            if not os.path.exists(os.path.join('models','AE')):
                os.makedirs(os.path.join('models','AE'), exist_ok=True)        
            torch.save(model_state_dict, os.path.join('models','AE','min_loss_model_' + config['atlas'] + '_length' + str(length) + '.pth'))
        corrs = torch.stack(corrs)
        first_section = corrs[:int(corrs.shape[0]/3)]
        mid_section = corrs[int(corrs.shape[0]/3):2*int(corrs.shape[0]/3)]
        last_section = corrs[2*int(corrs.shape[0]/3):]
        first_samples = first_section[torch.randperm(int(corrs.shape[0]/3))[:int(corrs.shape[0]/9)]]
        mid_samples = mid_section[torch.randperm(int(corrs.shape[0]/3))[:int(corrs.shape[0]/9)]]
        last_samples = last_section[torch.randperm(int(corrs.shape[0]/3))[:int(corrs.shape[0]/9)]]
        corrs = torch.cat([first_samples, mid_samples, last_samples])
        D,x,_ = ksvd(corrs.detach().cpu().numpy(), 100, 25, maxiter=50, device=config['device'])
        if save_models:        
            if not os.path.exists(os.path.join('models','AE')):
                os.makedirs(os.path.join('models','AE'), exist_ok=True)        
            torch.save(D, os.path.join('models','AE','dictionary_' + config['atlas'] + '.pt'))
    else:
        D = torch.load(os.path.join('models','AE','dictionary_' + config['atlas'] + '.pt'))
    
    val_result = test(models, val_data1, val_data2, D, len(test_winds), batch_size, device)
    test_result = test(models, test_data1, test_data2, D, len(test_winds), batch_size, device)
    results = {}
    results['result_val'] = val_result
    results['result_test'] = test_result
    
    if save_results:
        if not os.path.exists('results'):
            os.mkdir('results')   
        with open(os.path.join('results','test_results_' + config['atlas'] + '_AE.pkl'), 'wb') as f:
            pickle.dump(results,f)
    return results


if __name__ == '__main__':
    config = {}
    config['path_data'] = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/HCP'
    config['atlas'] = 'AAL' #AICHA, AAL
    config['batch_size'] = 128 
    config['train_test_lengths'] = [80,200,320] 
    config['test_lengths'] = [80,200,320]
    config['num_test_winds'] = 10
    config['shuffle'] = True
    config['epochs'] = 50
    config['lr'] = 0.0001
    config['weight_decay'] = 0.0002
    config['device'] = "cuda:1"
    config['save_models'] = False
    config['save_results'] = True
    config['load_prev_model'] = True
    results = main(config)

