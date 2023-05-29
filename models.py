# models
import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataset import KuaishouDataset
from mestric import xauc_score

class BackboneNet(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int):
        super(BackboneNet, self).__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.fc1        = nn.Linear(input_dim, hidden_dim)
        self.fc2        = nn.Linear(hidden_dim, output_dim)

        self.act        = nn.LeakyReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = x / (torch.norm(x, p=2.0, dim=1, keepdim=True) + 1e-8)
        return x
    

class DurationFusionModule(nn.Module): 
    def __init__(self,
                 input_dim   : int,
                 emb_dim     : int,
                 attem_dim   : int,
                 head_num    : int,
                 split_num   : int,
                 duration_num: int): 
        super(DurationFusionModule, self).__init__()
        self.input_dim = input_dim
        self.attem_dim = attem_dim
        self.head_num  = head_num
        self.split_num = split_num
        self._d1_embs  = torch.nn.Embedding(num_embeddings=duration_num, embedding_dim=emb_dim)
        self._d2_embs  = torch.nn.Embedding(num_embeddings=duration_num, embedding_dim=emb_dim)
        self.fc_q      = nn.Linear(emb_dim, split_num * attem_dim * head_num)
        self.fc_k      = nn.Linear(attem_dim, attem_dim * head_num)
        self.fc_v      = nn.Linear(attem_dim, attem_dim * head_num)
        self.fc_b      = nn.Linear(emb_dim, 1)
        self.fc_w      = nn.Linear(emb_dim, attem_dim * head_num)
        self.act       = nn.LeakyReLU()

    def multi_head_atten(self, q, k, v): 
        q = torch.reshape(self.fc_q(q), shape=[-1, self.split_num, self.head_num, self.attem_dim])
        k = torch.reshape(self.fc_k(k), shape=[-1, self.split_num, self.head_num, self.attem_dim])
        v = torch.reshape(self.fc_v(v), shape=[-1, self.split_num, self.head_num, self.attem_dim])
        q = torch.transpose(q, 2, 1)
        k = torch.transpose(k, 2, 1)
        v = torch.transpose(v, 2, 1)

        scores = torch.matmul(q, torch.transpose(k, 3, 2)) / math.sqrt(self.attem_dim)
        scores = torch.nn.Softmax(dim=-1)(scores)

        output = torch.sum(torch.matmul(scores, v), dim=-2, keepdim=False)
        return output
    
    def forward(self, r, d):
        d1     = self._d1_embs(d.to(torch.int64))
        d2     = self._d2_embs(d.to(torch.int64))
        r      = torch.reshape(r, shape=[-1, self.split_num, self.attem_dim])
        atten  = torch.reshape(self.multi_head_atten(d1, r, r), shape=[-1, self.attem_dim * self.head_num])
        w      = self.fc_w(d2)
        b      = self.fc_b(d2)
        output = torch.sum(self.act(atten) * w) + b
        return output 
    

class MultiClassDiscriminator(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_class: int):
        super(MultiClassDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.num_class = num_class
        self.fc1       = nn.Linear(input_dim, 128)
        self.fc2       = nn.Linear(128, num_class)
        self.act1      = nn.LeakyReLU()
        self.act2      = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return x
    

class ADN(nn.Module):
    def __init__(self, cfgs):
        super(ADN, self).__init__()
        self.backbone      = BackboneNet(**cfgs['backbone'])
        self.discriminator = MultiClassDiscriminator(**cfgs['m_d'])
        self.dfm           = DurationFusionModule(**cfgs['dfm'])
        self._init_embedings(cfgs)
        
    def _init_embedings(self, cfgs): 
        self._user_embs  = torch.nn.Embedding(num_embeddings=cfgs["user_num"], embedding_dim=cfgs["user_emb_dim"])
        self._video_embs = torch.nn.Embedding(num_embeddings=cfgs["video_num"], embedding_dim=cfgs["video_emb_dim"])
        self._utype_embs = torch.nn.Embedding(num_embeddings=cfgs["uType_num"], embedding_dim=cfgs["uType_emb_dim"])
        self._vtype_embs = torch.nn.Embedding(num_embeddings=cfgs["vType_num"], embedding_dim=cfgs["vType_emb_dim"])

    def get_interest_representation(self, x):
        uid = self._user_embs(x[0])
        vid = self._video_embs(x[1])
        utp = self._utype_embs(x[2])
        vtp = self._vtype_embs(x[3])
        inputs = torch.cat((uid, vid, utp, vtp), dim=-1)
        r = self.backbone(inputs)
        return r

    def forward(self, x):
        r = self.get_interest_representation(x)
        w = self.dfm(r, x[4])
        return w