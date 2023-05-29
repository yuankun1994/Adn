# dataset
import torch
import pickle
from torch.utils.data import Dataset

class KuaishouDataset(Dataset): 
    def __init__(self, file_name): 
        super(KuaishouDataset, self).__init__()
        self.file_name = file_name
        self._load_data()

    def _load_data(self): 
        with open(self.file_name, "rb") as f: 
            self.files = pickle.load(f)
            self.uids  = torch.IntTensor(pickle.load(f))
            self.vids  = torch.IntTensor(pickle.load(f))
            self.ts    = pickle.load(f)
            self.plays = torch.FloatTensor(pickle.load(f))
            self.duras = torch.IntTensor(pickle.load(f))
            self.utype = torch.IntTensor(pickle.load(f))
            self.vtype = torch.IntTensor(pickle.load(f))
            self.gid   = torch.IntTensor(pickle.load(f))

    def __len__(self):
        return len(self.ts)
    
    def __getitem__(self, i): 
        idx = i % self.__len__()
        return [self.uids[idx], self.vids[idx], self.utype[idx], self.vtype[idx], self.duras[idx], self.gid[idx], self.plays[idx]]