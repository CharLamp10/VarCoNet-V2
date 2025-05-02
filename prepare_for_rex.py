parent_path = r'/home/student1/Desktop/Charalampos_Lamprou/VarCoNetV2_extras/ksvd-sparse-dictionary'
import sys
sys.path.append(parent_path)
from utils import PCC
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Conv3d
from torch.nn.functional import avg_pool1d
import os
import pickle
import math
from torch.nn import Conv1d
from utils import upper_triangular_cosine_similarity
import pandas as pd
from sparseRep import random_sensor, sense, reconstruct


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
        self.conv1 = Conv1d(in_channels=1, out_channels=16, kernel_size=4, stride=2)
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


class Model_VAE(nn.Module):

    def __init__(self, roi_num):
        super().__init__()

        self.vae = VAE(input_dim=int((roi_num*(roi_num-1))/2))
        
    def forward(self, x, train):
        y,mu,logvar = self.vae(x)
        if train:
            return y,mu,logvar 
        else:
            return y
        
        
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


class Model_AE(nn.Module):

    def __init__(self, length):
        super().__init__()
        self.ae = AE(input_dim=length)
        self.length = length
        
    def forward(self, x):
        y = self.ae(x.permute(0,2,1).contiguous().view(-1,self.length))
        y = y.view(x.shape[0],x.shape[2],x.shape[1]).permute(0,2,1)
        return y
    
    
class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(16, 16, 1)  # No downsampling
        self.enc2 = self.conv_block(16, 32, 2)  # Downsampling
        self.enc3 = self.conv_block(32, 32, 2)  # Downsampling

        # Bottleneck
        self.bottleneck = self.conv_block(32, 32, 2)  # Downsampling

        # Decoder
        self.dec3 = self.deconv_block(32, 32, 2, (1, 0, 1))  # Upsampling
        self.dec2 = self.deconv_block(64, 32, 2, (0, 0, 0))  # Upsampling
        self.dec1 = self.deconv_block(64, 16, 2, (0, 0, 0))  # Upsampling

    def conv_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def deconv_block(self, in_channels, out_channels, stride, output_padding):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=output_padding),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        
        e3 = self.enc3(e2)
        
        # Bottleneck
        b = self.bottleneck(e3)

        # Decoder
        d3 = self.dec3(b)
        d3 = torch.cat([d3, e3], dim=1)

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)

        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)

        return d1,b


class Model_PFN(nn.Module):
    def __init__(self):
        super(Model_PFN, self).__init__()
        self.conv1 = Conv3d(in_channels=1, out_channels=16,kernel_size=3, stride=1,padding=1)
        self.unet = UNet3D()
        self.conv2 = Conv3d(in_channels=32, out_channels=16,kernel_size=3, stride=1, padding=1)
        self.conv3 = Conv3d(in_channels=16, out_channels=16,kernel_size=3, stride=1, padding=1)
        self.pred = Conv3d(in_channels=16, out_channels=17,kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):   
        x = self.conv1(torch.permute(x,(4,0,1,2,3)))
        x = torch.mean(x, dim=0)
        x,b = self.unet(x.unsqueeze(0))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.relu(self.pred(x))
        x = x / x.max(dim=1, keepdim=True).values
        return x,b
    
    
def test_varconet(encoder_model,test_data1,test_data2,device):
    encoder_model.eval()
    with torch.no_grad():
        outputs = []
        for data in test_data1:
            data = torch.from_numpy(data).to(device)
            outputs.append(encoder_model(data[:200,:].unsqueeze(0).float()))
        outputs1 = torch.stack(outputs).cpu().numpy()
        
        outputs = []
        for data in test_data2:
            data = torch.from_numpy(data).to(device)
            outputs.append(encoder_model(data[:200,:].unsqueeze(0).float()))
        outputs2 = torch.stack(outputs).cpu().numpy()
        
    return outputs1, outputs2
    

def test_vae(model,test_data1,test_data2,D,device):
    model.eval()
    with torch.no_grad():
        outputs = []
        for data in test_data1:
            data = data.to(device)
            outputs.append(model(data.unsqueeze(0).float(),False))
        outputs1 = torch.stack(outputs)
        
        outputs = []
        for data in test_data2:
            data = data.to(device)
            outputs.append(model(data.unsqueeze(0).float(),False))
        outputs2 = torch.stack(outputs)
        
    test_data1 = torch.stack(test_data1)
    test_data2 = torch.stack(test_data2)
    
    outputs1 = outputs1.squeeze()
    outputs2 = outputs2.squeeze()
    outputs1 = outputs1 - test_data1.to(device)
    outputs2 = outputs2 - test_data2.to(device)
    N,K = outputs1.shape
    
    outputs = torch.cat((outputs1, outputs2), dim=0)
    sensor = random_sensor((K, int(0.4*K)), device=device)
    representation = sense(outputs, sensor, device)
    r = reconstruct(representation.T, sensor, D, 25, device).T
    outputs = outputs.cpu().numpy() - r.cpu().numpy()
    del test_data1, test_data2, outputs1, outputs2, sensor, representation, r
    torch.cuda.empty_cache()
    outputs1 = outputs[:int(outputs.shape[0]/2)]
    outputs2 = outputs[int(outputs.shape[0]/2):]
    return outputs1, outputs2


def test_ae(model,test_data1,test_data2,D,device):
    model.eval()
    with torch.no_grad():
        outputs = []
        for data in test_data1:
            data = torch.from_numpy(data[:200,:]).to(device)
            outputs.append(model(data.unsqueeze(0).float()))
        outputs1 = torch.stack(outputs)
        
        outputs = []
        for data in test_data2:
            data = torch.from_numpy(data[:200,:]).to(device)
            outputs.append(model(data.unsqueeze(0).float()))
        outputs2 = torch.stack(outputs)
    
    outputs1 = outputs1.squeeze()
    outputs2 = outputs2.squeeze()
    outputs1 = outputs1 - torch.from_numpy(np.stack(test_data1)[:,:200,:]).to(device)
    outputs2 = outputs2 - torch.from_numpy(np.stack(test_data2)[:,:200,:]).to(device)
    outputs = torch.cat((outputs1, outputs2), dim=0)
    corrs = []
    for output in outputs:
        corr = PCC(output.T)
        triu_indices = torch.triu_indices(corr.shape[0], corr.shape[0], offset=1)
        corrs.append(corr[triu_indices[0], triu_indices[1]])
    corrs = torch.stack(corrs)
    N,K = corrs.shape
    sensor = random_sensor((K, int(0.4*K)), device=device)
    representation = sense(corrs.float(), sensor, device)
    r = reconstruct(representation.T, sensor, D, 25, device).T
    outputs = corrs.cpu().numpy() - r.cpu().numpy()
    del test_data1, test_data2, outputs1, outputs2, sensor, representation, r
    torch.cuda.empty_cache()
    outputs1 = outputs[:int(outputs.shape[0]/2)]
    outputs2 = outputs[int(outputs.shape[0]/2):]
    return outputs1, outputs2


def test_pfn(encoder_model,test_data1,test_data2,device):
    encoder_model.eval()
    with torch.no_grad():
        print('')
        outputs1 = []
        for i,data in enumerate(test_data1):
            data = torch.load(data)[:,:,:,:416].to(device)
            data = data/torch.max(data)
            _,out = encoder_model(data.unsqueeze(0))
            out = out.contiguous().view(1,-1).cpu()
            outputs1.append(out)
            if (i+1) % 10 == 0:
                print(i+1)
        outputs1 = torch.stack(outputs1)
        
        print('')
        outputs2 = []
        for i,data in enumerate(test_data2):
            data = torch.load(data)[:,:,:,:416].to(device)
            data = data/torch.max(data)
            _,out = encoder_model(data.unsqueeze(0))
            out = out.contiguous().view(1,-1).cpu()
            outputs2.append(out)
            if (i+1) % 10 == 0:
                print(i+1)
        outputs2 = torch.stack(outputs2)
        
        outputs1 = outputs1.numpy()
        outputs2 = outputs2.numpy()
    return outputs1, outputs2


path = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/HCP'
path_PFN = r'/mnt/harddrive/HCP_PFN'
path_test = os.path.join(path_PFN,'test')

atlas = 'AAL'
length = 80
with open('best_params_VarCoNet_v2_final_' + atlas + '.pkl','rb') as f:
    best_params = pickle.load(f)  
batch_size = 128
model_config = {}
model_config['layers'] = best_params['layers']
model_config['n_heads'] = best_params['n_heads']
model_config['dim_feedforward'] = best_params['dim_feedforward'] 
model_config['max_length'] = 320
device = 'cuda:1'
device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
   

data = np.load(os.path.join(path,'test_data_HCP_' + atlas + '_1_resampled.npz'))
test_data1 = []
for key in data:
    test_data1.append(data[key])

data = np.load(os.path.join(path,'test_data_HCP_' + atlas + '_2_resampled.npz'))
test_data2 = []
for key in data:
    test_data2.append(data[key])

test_data1 = test_data1[200:]
test_data2 = test_data2[200:] 
        
test_data1_varconet = test_data1
test_data2_varconet = test_data2  

test_data1_ae = test_data1
test_data2_ae = test_data2  

test_data1_vae = []
test_data2_vae = []
for i,data in enumerate(test_data1):
    data = torch.from_numpy(data.astype(np.float32))
    data = data[:length,:]
    corr = PCC(data.T)
    triu_indices = torch.triu_indices(corr.shape[0], corr.shape[0], offset=1)
    test_data1_vae.append(corr[triu_indices[0], triu_indices[1]])
    
for i,data in enumerate(test_data2):
    data = torch.from_numpy(data.astype(np.float32))
    data = data[:length,:]
    corr = PCC(data.T)
    triu_indices = torch.triu_indices(corr.shape[0], corr.shape[0], offset=1)
    test_data2_vae.append(corr[triu_indices[0], triu_indices[1]])
   
roi_num = test_data1[0].shape[1]

test_data_PFN = []
data_dir = os.listdir(path_test)
for file in data_dir:
    test_data_PFN.append(os.path.join(path_test,file))
    
test_data1_pfn = test_data_PFN[200:int(len(test_data_PFN)/2)]
test_data2_pfn = test_data_PFN[int(len(test_data_PFN)/2)+200:]


'''------------------------------------------VarCoNet-V2---------------------------------------'''  
VarCoNetV2 = SeqenceModel(model_config, roi_num).to(device)
state_dict_varconet = torch.load(os.path.join('models','VarCoNetV2','max_acc_model_' + atlas + '.pth'))
VarCoNetV2.load_state_dict(state_dict_varconet)
out1_varconet, out2_varconet = test_varconet(VarCoNetV2,test_data1_varconet,test_data2_varconet,device)
out1_varconet = np.squeeze(out1_varconet)
out2_varconet = np.squeeze(out2_varconet)
final_varconet = np.zeros((2*out1_varconet.shape[0], out1_varconet.shape[1]))
subIDs = []
visits = []
count = 0
for i in range(0,out1_varconet.shape[0]):
    final_varconet[count] = out1_varconet[i,:]
    count += 1
    final_varconet[count] = out2_varconet[i,:]
    count += 1
    if i < 10:
        subIDs.append('sub00' + str(i))
        subIDs.append('sub00' + str(i))
    elif i >= 10 and i < 100:
        subIDs.append('sub0' + str(i))
        subIDs.append('sub0' + str(i))
    else:
        subIDs.append('sub' + str(i))
        subIDs.append('sub' + str(i))
    visits.append('time1')
    visits.append('time2')

columns = []
for i in range(out1_varconet.shape[1]):
    columns.append('ROI.' + str(i))
    
subIDs = np.expand_dims(np.array(subIDs), axis=1)
visits = np.expand_dims(np.array(visits), axis=1)
subIDs = pd.DataFrame(subIDs, columns=['subID'])
visits = pd.DataFrame(subIDs, columns=['visit'])
final_varconet = pd.DataFrame(final_varconet, columns=columns)
final_varconet = pd.concat((subIDs,visits,final_varconet), axis=1)
final_varconet.to_csv(os.path.join('results','rex_' + atlas + '_VarCoNetV2_' + str(length) + '_samples.csv'))

"""
'''------------------------------------------VAE-K-SVD---------------------------------------'''  
vae = Model_VAE(roi_num).to(device)
state_dict_vae = torch.load(os.path.join('models','VAE','min_loss_model_' + atlas + '.pth'))
D = torch.load(os.path.join('models','VAE','dictionary_' + atlas + '.pt'))
vae.load_state_dict(state_dict_vae)
out1_vae, out2_vae = test_vae(vae,test_data1_vae,test_data2_vae,D,device)
out1_vae = np.squeeze(out1_vae)
out2_vae = np.squeeze(out2_vae)
final_vae = np.zeros((2*out1_vae.shape[0], out1_vae.shape[1]))
count = 0
for i in range(0,out1_vae.shape[0]):
    final_vae[count] = out1_vae[i,:]
    count += 1
    final_vae[count] = out2_vae[i,:]
    count += 1
    
final_vae = pd.DataFrame(final_vae, columns=columns)
final_vae = pd.concat((subIDs,visits,final_vae), axis=1)
final_vae.to_csv(os.path.join('results','rex_' + atlas + '_VAE_' + str(length) + '_samples.csv'))    


'''------------------------------------------AE-K-SVD---------------------------------------'''    
ae = Model_AE(length).to(device)
state_dict_ae = torch.load(os.path.join('models','AE','min_loss_model_' + atlas + '_length' + str(length) + '.pth'))
D = torch.load(os.path.join('models','AE','dictionary_' + atlas + '.pt'))
ae.load_state_dict(state_dict_ae)
out1_ae, out2_ae = test_ae(ae,test_data1_ae,test_data2_ae,D,device)
out1_ae = np.squeeze(out1_ae)
out2_ae = np.squeeze(out2_ae)
final_ae = np.zeros((2*out1_ae.shape[0], out1_ae.shape[1]))
count = 0
for i in range(0,out1_ae.shape[0]):
    final_ae[count] = out1_ae[i,:]
    count += 1
    final_ae[count] = out2_ae[i,:]
    count += 1
    
final_ae = pd.DataFrame(final_ae, columns=columns)
final_ae = pd.concat((subIDs,visits,final_ae), axis=1)
final_ae.to_csv(os.path.join('results','rex_' + atlas + '_AE_' + str(length) + '_samples.csv'))  


'''------------------------------------------PCC---------------------------------------''' 
out1_pcc = []
out2_pcc = []
for data in test_data1:
    corr = np.corrcoef(data[:length,:].T)
    triu_indices = np.triu_indices(corr.shape[0], k=1)
    upper_triangular_values = corr[triu_indices[0], triu_indices[1]]
    out1_pcc.append(upper_triangular_values)
for data in test_data2:
    corr = np.corrcoef(data[:length,:].T)
    triu_indices = np.triu_indices(corr.shape[0], k=1)
    upper_triangular_values = corr[triu_indices[0], triu_indices[1]]
    out2_pcc.append(upper_triangular_values)
    
out1_pcc = np.stack(out1_pcc)
out2_pcc = np.stack(out2_pcc)
final_pcc = np.zeros((2*out1_pcc.shape[0], out1_pcc.shape[1]))
count = 0
for i in range(0,out1_pcc.shape[0]):
    final_pcc[count] = out1_pcc[i,:]
    count += 1
    final_pcc[count] = out2_pcc[i,:]
    count += 1
    
final_pcc = pd.DataFrame(final_pcc, columns=columns)
final_pcc = pd.concat((subIDs,visits,final_pcc), axis=1)
final_pcc.to_csv(os.path.join('results','rex_' + atlas + '_PCC_' + str(length) + '_samples.csv'))  

'''------------------------------------------PFN-SSL---------------------------------------''' 
if atlas == 'AICHA':
    pfn = Model_PFN().to(device)
    state_dict_pfn = torch.load(os.path.join('models','PFN','min_loss_model.pth'))
    pfn.load_state_dict(state_dict_pfn)
    out1_pfn, out2_pfn = test_pfn(pfn,test_data1_pfn,test_data2_pfn,device)
    out1_pfn = np.squeeze(out1_pfn)
    out2_pfn = np.squeeze(out2_pfn)
    final_pfn = np.zeros((2*out1_pfn.shape[0], out1_pfn.shape[1]))
    subIDs = []
    visits = []
    count = 0
    for i in range(0,out1_pfn.shape[0]):
        final_pfn[count] = out1_pfn[i,:]
        count += 1
        final_pfn[count] = out2_pfn[i,:]
        count += 1
        if i < 10:
            subIDs.append('sub00' + str(i))
            subIDs.append('sub00' + str(i))
        elif i >= 10 and i < 100:
            subIDs.append('sub0' + str(i))
            subIDs.append('sub0' + str(i))
        else:
            subIDs.append('sub' + str(i))
            subIDs.append('sub' + str(i))
        visits.append('time1')
        visits.append('time2')
    
    
    columns = []
    for i in range(out1_pfn.shape[1]):
        columns.append('ROI.' + str(i))
    subIDs = np.expand_dims(np.array(subIDs), axis=1)
    visits = np.expand_dims(np.array(visits), axis=1)
    subIDs = pd.DataFrame(subIDs, columns=['subID'])
    visits = pd.DataFrame(subIDs, columns=['visit'])
    final_pfn = pd.DataFrame(final_pfn, columns=columns)
    final_pfn = pd.concat((subIDs,visits,final_pfn), axis=1)
    final_pfn.to_csv(os.path.join('results','rex_' + atlas + '_PFN_' + str(length) + '_samples.csv'))  
    
"""