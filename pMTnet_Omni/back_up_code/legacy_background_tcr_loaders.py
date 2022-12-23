# THESE DATALOADERS LOADS EMBEDDINGS 
# FASTER PREDICTION
import os 

import pandas as pd 

from pMTnet_Omni.utilities import read_file

import torch 
from torch.utils.data import Dataset, DataLoader, RandomSampler

from typing import Union, Optional 


class background_tcr_datatset_class(Dataset):
    def __init__(self, data_dir: str,\
                 species: str="human",\
                 chain: str="alpha",
                 load_all: bool=False) -> None:
        super().__init__()
        file_name = species + "_" + chain + ".txt"
        self.data_path = os.path.join(data_dir, file_name)

        if chain == "alpha":
            self.chain_abbreviation = "a"
        else:
            self.chain_abbreviation = "b"

        with open(self.data_path, mode='r') as f:
            for count, line in enumerate(f):
                pass 
        self.data_len = count

        self.load_all = load_all
        self.tcrs = None
        if self.load_all:
            self.tcrs = read_file(file_path=self.data_path)
            self.tcrs = torch.tensor(self.tcrs.values, dtype=torch.float32)

    def __len__(self):
        return self.data_len 
    
    def __getitem__(self, idx):
        if self.load_all:
            return self.tcrs[idx, :]
        else: 
            self.tcrs = read_file(file_path=self.data_path,\
                                 sep="\t",\
                                 header=0,\
                                 idx=idx)
            self.tcrs = torch.tensor(self.tcrs.values, dtype=torch.float32)
            print(idx)
            print(type(idx))
            return self.tcrs


def generate_background_tcr_dataloader(background_tcr_dataset: background_tcr_datatset_class,\
                                       batch_size: int,\
                                       replacement: bool=True):
    sampler = RandomSampler(background_tcr_dataset,\
                            replacement = replacement,\
                            num_samples=len(background_tcr_dataset))
    return DataLoader(background_tcr_dataset, batch_size=batch_size, sampler=sampler)


