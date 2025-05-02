# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:17:28 2024

@author: 100063082
"""

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import numpy as np
import random
import os


class Loss(ABC):
    @abstractmethod
    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs) -> torch.FloatTensor:
        pass

    def __call__(self, anchor, sample, pos_mask=None, neg_mask=None, *args, **kwargs) -> torch.FloatTensor:
        loss = self.compute(anchor, sample, pos_mask, neg_mask, *args, **kwargs)
        return loss

def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()

class InfoNCE(Loss):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss.mean()

class Sampler(ABC):
    def __init__(self, intraview_negs=False):
        self.intraview_negs = intraview_negs

    def __call__(self, anchor, sample, *args, **kwargs):
        ret = self.sample(anchor, sample, *args, **kwargs)
        if self.intraview_negs:
            ret = self.add_intraview_negs(*ret)
        return ret

def get_sampler(mode: str, intraview_negs: bool) -> Sampler:
    if mode in {'L2L', 'G2G'}:
        return SameScaleSampler(intraview_negs=intraview_negs)
    else:
        raise RuntimeError(f'unsupported mode: {mode}')

class SameScaleSampler(Sampler):
    def __init__(self, *args, **kwargs):
        super(SameScaleSampler, self).__init__(*args, **kwargs)

    def sample(self, anchor, sample, *args, **kwargs):
        assert anchor.size(0) == sample.size(0)
        num_nodes = anchor.size(0)
        device = anchor.device
        pos_mask = torch.eye(num_nodes, dtype=torch.float32, device=device)
        neg_mask = 1. - pos_mask
        return anchor, sample, pos_mask, neg_mask


def add_extra_mask(pos_mask, neg_mask=None, extra_pos_mask=None, extra_neg_mask=None):
    if extra_pos_mask is not None:
        pos_mask = torch.bitwise_or(pos_mask.bool(), extra_pos_mask.bool()).float()
    if extra_neg_mask is not None:
        neg_mask = torch.bitwise_and(neg_mask.bool(), extra_neg_mask.bool()).float()
    else:
        neg_mask = 1. - pos_mask
    return pos_mask, neg_mask

    
class DualBranchContrast(torch.nn.Module):
    def __init__(self, loss: Loss, mode: str, intraview_negs: bool = False, **kwargs):
        super(DualBranchContrast, self).__init__()
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=intraview_negs)
        self.kwargs = kwargs

    def forward(self, h1=None, h2=None, g1=None, g2=None, batch=None, h3=None, h4=None,
                extra_pos_mask=None, extra_neg_mask=None):
        if self.mode == 'L2L':
            assert h1 is not None and h2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=h1, sample=h2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=h2, sample=h1)
        elif self.mode == 'G2G':
            assert g1 is not None and g2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=g2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=g1)

        pos_mask1, neg_mask1 = add_extra_mask(pos_mask1, neg_mask1, extra_pos_mask, extra_neg_mask)
        pos_mask2, neg_mask2 = add_extra_mask(pos_mask2, neg_mask2, extra_pos_mask, extra_neg_mask)
        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1, neg_mask=neg_mask1, **self.kwargs)
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2, neg_mask=neg_mask2, **self.kwargs)

        return (l1 + l2) * 0.5
    
class ABIDEDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
    
    
class ABIDEDataset_BAnD(Dataset):
    def __init__(self, data, labels, path, max_length, mode):
        self.data = data
        self.labels = labels
        self.path = path
        self.max_length = max_length
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = np.load(os.path.join(self.path,self.data[idx]))
        if self.mode == 'train':
            if x.shape[-1] < self.max_length:
                repeats = (self.max_length // x.shape[-1]) + 1
                x = np.tile(x, repeats)
                x = x[:,:,:,:self.max_length]
            else:
                start_idx = np.random.randint(0, x.shape[-1] - self.max_length + 1)
                x = x[:, :, :, start_idx:start_idx + self.max_length]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
    
    
def upper_triangular_cosine_similarity(x):
    N, M, D = x.shape
    x_norm = F.normalize(x, p=2, dim=-1)
    cosine_similarity = torch.matmul(x_norm, x_norm.transpose(1, 2))
    triu_indices = torch.triu_indices(M, M, offset=1)
    upper_triangular_values = cosine_similarity[:, triu_indices[0], triu_indices[1]]
    return upper_triangular_values    

def removeDuplicates(names,inds):
    names_batch = []
    for ind in inds:
        names_batch.append(names[ind])
    names_unique,counts = np.unique(names_batch,return_counts=True)
    if len(names_unique) == len(names_batch):
        return inds
    else:
        non_common = list(set(names).symmetric_difference(set(names_batch)))
        positions = np.where(counts>1)[0]
        for pos in positions:
            name_dupl = names_unique[pos]
            pos_name = np.where(np.array(names_batch) == name_dupl)[0][1]
            names_batch[pos_name] = non_common[random.randint(0, len(non_common)-1)]
            possible_pos = np.where(np.array(names) == names_batch[pos_name])[0]
            inds[pos_name] = possible_pos[random.randint(0,len(possible_pos)-1)]
            non_common = list(set(non_common).symmetric_difference(set(names_batch)))
        return inds

def test_augment(data,wind_sizes,num_winds,max_length):
    windows_all = []
    for wind_size in wind_sizes:
        windows = []
        step_size = (data.shape[0] - wind_size) // (num_winds - 1)
        for i in range(0, step_size*num_winds,step_size):
            temp = torch.zeros(max_length,data.shape[1])
            temp[:wind_size] = data[i:i+wind_size]
            windows.append(temp)
        windows_all.append(torch.stack(windows))
    return torch.cat(windows_all,dim=0)



def test_augment_AE(data,wind_sizes,num_winds,max_length):
    for wind_size in wind_sizes:
        windows = []
        step_size = (data.shape[0] - wind_size) // (num_winds - 1)
        for i in range(0, step_size*num_winds,step_size):
            temp = torch.zeros(max_length,data.shape[1])
            temp[:wind_size] = data[i:i+wind_size]
            windows.append(temp)
    return torch.stack(windows)


def augment_hcp(data, train_length_limits, device):
    max_length = train_length_limits[-1]   
    data1 = []
    data2 = []
    for dat in data:
        length1 = random.randint(train_length_limits[0],train_length_limits[-1]-1)
        length2 = random.randint(train_length_limits[0],train_length_limits[-1]-1)

        a11 = int(length1/2+1)
        a12 = dat.shape[0]-int(length1/2+2)
        a21 = int(length2/2+1)
        a22 = dat.shape[0]-int(length2/2+2)
        if a12 > a11:
            c1 = random.randint(a11,a12)
        else:
            c1 = a11
        if a22 > a21:
            c2 = random.randint(a21,a22)   
        else:
            c2 = a21  
        counter = 0            
        c1s = [c1]
        c2s = [c2]
        dists = [abs(c1 - c2)]
        while counter < 100:
            if a12 > a11:
                c1 = random.randint(a11,a12)
            else:
                c1 = a11
            if a22 > a21:
                c2 = random.randint(a21,a22)   
            else:
                c2 = a21
            c1s.append(c1)
            c2s.append(c2)
            dists.append(abs(c1 - c2))
            counter = counter + 1
        max_dist = max(dists)
        max_pos = dists.index(max_dist)
        c1 = c1s[max_pos]
        c2 = c2s[max_pos]
        temp = torch.zeros(max_length,dat.shape[1])
        if length1 % 2 == 0:
            temp[:length1,:] = dat[c1-int(length1/2):c1+int(length1/2),:]
        else:
            temp[:length1,:] = dat[c1-int(length1/2):c1+int(length1/2)+1,:]
        data1.append(temp) 
        temp = torch.zeros(max_length,dat.shape[1])
        if length2 % 2 == 0:
            temp[:length2,:] = dat[c2-int(length2/2):c2+int(length2/2),:]
        else:
            temp[:length2,:] = dat[c2-int(length2/2):c2+int(length2/2)+1,:]
        data2.append(temp)
    data1 = torch.stack(data1).to(device)
    data2 = torch.stack(data2).to(device)
    return [data1,data2]


def augment(data, train_length_limits, max_length, device):
    data = data.to(device)  

    batch_size, _, feat_dim = data.shape  # Extract batch size & feature dimension
    data1 = torch.zeros((batch_size, max_length, feat_dim), device=device)
    data2 = torch.zeros((batch_size, max_length, feat_dim), device=device)

    for i, dat in enumerate(data):
        zero_rows = torch.all(dat == 0, dim=1)
        zero_row_indices = torch.where(zero_rows)[0]
        if len(zero_row_indices) > 0:
            dat = dat[:torch.min(zero_row_indices), :]

        low_limit = train_length_limits[0]
        up_limit = min(dat.shape[0], train_length_limits[1])

        length1 = torch.randint(low_limit, up_limit, (1,), device=device).item()
        length2 = torch.randint(low_limit, up_limit, (1,), device=device).item()

        a11, a12 = length1 // 2 + 1, dat.shape[0] - length1 // 2 - 2
        a21, a22 = length2 // 2 + 1, dat.shape[0] - length2 // 2 - 2

        # Compute 100 alternative distances on GPU
        c1_candidates = torch.randint(a11, max(a12, a11 + 1), (100,), device=device)
        c2_candidates = torch.randint(a21, max(a22, a21 + 1), (100,), device=device)
        dists = torch.abs(c1_candidates - c2_candidates)

        # Select the maximum distance
        max_pos = torch.argmax(dists).item()
        c1, c2 = c1_candidates[max_pos].item(), c2_candidates[max_pos].item()

        # Slice & store in preallocated tensors
        if length1 % 2 == 0:
            data1[i, :length1, :] = dat[c1 - length1 // 2 : c1 + length1 // 2, :]
        else:
            data1[i, :length1, :] = dat[c1 - length1 // 2 : c1 + length1 // 2 + 1, :]

        if length2 % 2 == 0:
            data2[i, :length2, :] = dat[c2 - length2 // 2 : c2 + length2 // 2, :]
        else:
            data2[i, :length2, :] = dat[c2 - length2 // 2 : c2 + length2 // 2 + 1, :]

    return [data1, data2]


def test_augment_overlap(data, wind_sizes, overlap, min_length, max_length, device):
    data = data.to(device)  

    zero_rows = torch.all(data == 0, dim=1)
    zero_row_indices = torch.where(zero_rows)[0]
    if len(zero_row_indices) > 0:
        data = data[:torch.min(zero_row_indices), :]

    windows_all = []
    data_len = data.shape[0]

    for i in range(len(wind_sizes) - 1):
        if data_len // wind_sizes[i+1] >= min_length:
            wind_size_min = max(min_length, wind_sizes[i])
            wind_size_max = min(data_len // wind_sizes[i+1], data_len)
            wind_size = torch.randint(wind_size_min, wind_size_max + 1, (1,), device=device).item()

            step_size = int(wind_size * (1 - overlap))
            num_windows = max(1, (data_len - wind_size) // step_size + 1)

            indices = torch.arange(0, num_windows * step_size, step_size, device=device).unsqueeze(1)
            range_indices = torch.arange(wind_size, device=device).unsqueeze(0)
            window_indices = indices + range_indices  # Shape: (num_windows, wind_size)

            windows = data[window_indices]  # Gather windows using tensor indexing

            # Pad windows to max_length (if needed)
            pad_size = max_length - wind_size
            if pad_size > 0:
                pad_tensor = torch.zeros((windows.shape[0], pad_size, data.shape[1]), device=device)
                windows = torch.cat((windows, pad_tensor), dim=1)

            windows_all.append(windows)

    return torch.cat(windows_all, dim=0) if windows_all else torch.empty(0, max_length, data.shape[1], device=device)


def repeat_to_fill(x):
    batch_size, seq_len, num_channels = x.shape
    new_x = torch.zeros_like(x)
    for i in range(batch_size):
        nonzero_rows = (x[i] != 0).any(dim=1)
        N = nonzero_rows.sum().item()
        if N == 0:
            continue
        signal = x[i, :N,:]
        if seq_len // N > 1:
            for i in range(seq_len // N):
                if i == 0:
                    new_signal = torch.concat((signal,signal),dim=0)
                elif i < seq_len // N - 1:
                    new_signal = torch.concat((new_signal,signal),dim=0)
                else:
                    new_signal = torch.concat((new_signal,signal[:seq_len-new_signal.shape[0],:]),dim=0)
        elif seq_len // N == 1: 
            new_signal = torch.concat((signal,signal[:seq_len-signal.shape[0],:]),dim=0)
        else:
            new_signal = signal
        new_x[i] = new_signal
    return new_x


def PCC(X):
    M, N = X.shape
    mean = X.mean(dim=1, keepdim=True)
    X_centered = X - mean
    std = X_centered.std(dim=1, unbiased=True, keepdim=True)
    std[std == 0] = 1
    Z = X_centered / std
    corr_matrix = torch.matmul(Z, Z.T) / (N - 1)
    return corr_matrix

def augment_VAE(data,a,b,device):
    x = []
    for dat in data:
        length = random.randint(a, b)
        max_start = dat.shape[0] - length
        start = random.randint(0, max_start)
        dat = dat[start:start + length,:]
        corr = PCC(dat.T)
        triu_indices = torch.triu_indices(corr.shape[0], corr.shape[0], offset=1)
        upper_triangular_values = corr[triu_indices[0], triu_indices[1]]
        x.append(upper_triangular_values)
    return torch.stack(x).float().to(device)


def test_augment_PCC(data,wind_sizes,num_winds):
    new_data = []
    for wind_size in wind_sizes:
        step_size = (data.shape[0] - wind_size) // (num_winds - 1)
        for i in range(0, step_size*num_winds,step_size):
            segment = data[i:i+wind_size]
            corr = np.corrcoef(segment.T)
            triu_indices = np.triu_indices(corr.shape[0], k=1)
            upper_triangular_values = corr[triu_indices[0], triu_indices[1]]
            new_data.append(upper_triangular_values)
    return np.stack(new_data)
