import torch.nn as nn
from torch.nn.functional import avg_pool1d
import torch.nn.functional as F
from torch.nn import Conv1d
import torch

def upper_triangular_cosine_similarity(x):
    N, M, D = x.shape
    x_norm = F.normalize(x, p=2, dim=-1)
    cosine_similarity = torch.matmul(x_norm, x_norm.transpose(1, 2))
    triu_indices = torch.triu_indices(M, M, offset=1)
    upper_triangular_values = cosine_similarity[:, triu_indices[0], triu_indices[1]]
    return upper_triangular_values 


class PositionalEncodingTrainable(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncodingTrainable, self).__init__()
        pe = torch.zeros(1, max_seq_length, d_model)
        self.pe = nn.Parameter(pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CNN_Transformer(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, dim_feedforward, max_len):
        super(CNN_Transformer, self).__init__()
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

class VarCoNet(nn.Module):

    def __init__(self, model_config, roi_num):
        super().__init__()

        self.extract = CNN_Transformer(
                d_model=roi_num,
                n_layers=model_config['layers'],n_heads=model_config['n_heads'],
                dim_feedforward=model_config['dim_feedforward'],
                max_len=model_config['max_length'])
    def forward(self, x):
        x = self.extract(x)
        x = upper_triangular_cosine_similarity(x)
        return x 
    
class VarCoNet_noCNN(nn.Module):

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