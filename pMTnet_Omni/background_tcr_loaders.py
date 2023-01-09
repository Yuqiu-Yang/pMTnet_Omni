# Data IO
import os
import pandas as pd 

# Numeric manipulation 
import numpy as np 

# Typing 
from typing import Union, Optional


class background_tcr_dataset_class:
    def __init__(self, data_dir: str='./data/background_tcrs/',\
                 species: str="human",\
                 chain: str="alpha",\
                 load_embedding: bool=False,\
                 load_size: int=10000) -> None:
        """Dataset class for one chain and one species 
        
        Parameters
        ----------
        data_dir: str
            The directory containing background TCRs
        species: str
            Species to load 
        chain: str
            Chain to load 
        load_embedding: bool
            Whether or not to load embedding file or string file 
        load_size: int 
            The number of samples to load from the background TCRs file
        
        """
        # Get file path 
        if not load_embedding:
            file_name = species + "_" + chain + ".txt"
        else:
            file_name = species + "_" + chain + "_embedding.txt"
        
        self.data_path = os.path.join(data_dir, file_name)
        # Check number of observations 
        with open(self.data_path, mode='r') as f:
            for count, _ in enumerate(f):
                pass 
        self.data_len = load_size
        # 
        if load_size > count:
            print("Load size greater than the dataset. Setting the load size to the size of the dataset\n")
            self.data_len = count
            self.tcrs = pd.read_csv(self.data_path,\
                                    sep="\t",\
                                    header=0)
        else:
            # 
            skip_idx = np.random.choice(count,\
                                        size=count - load_size,\
                                        replace=False)
            skip_idx += 1
            self.tcrs = pd.read_csv(self.data_path,\
                                    sep="\t",\
                                    header=0,\
                                    skiprows=skip_idx)

    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx: list):
        return self.tcrs.iloc[idx,:]
        

class background_tcr_dataloader_class:
    def __init__(self, background_tcr_dataset: background_tcr_dataset_class,\
                 batch_size: Union[int, list]=10, replacement: bool=True) -> None:
        """Dataloader class for a single chain on a single species 

        Parameters
        ----------
        background_tcr_dataset: background_tcr_dataset_class
            The dataset class 
        batch_size: Union[int, list]
            Batch size to load the data. If it is a list, each iteration will load a size found in the list
        replacement: bool
            Whether or not sample with replacement 

        """
        self.background_tcr_dataset = background_tcr_dataset
        self.batch_sizes = batch_size
        if isinstance(batch_size, int):
            self.batch_size = batch_size    
        else:
            self.batch_size = 0
        self.replacement = replacement
        
    def __iter__(self):
        return background_tcr_dataset_iterator(self)


class background_tcr_dataset_iterator:
    def __init__(self, background_tcr_dataloader: background_tcr_dataloader_class) -> None:
        """Iterator class for a tcr dataset

        Parameters
        ----------
        background_tcr_dataloader: background_tcr_dataloader_class
            Dataloader for the dataset

        """
        self.background_tcr_dataloader = background_tcr_dataloader
        self.batch_size_ind = -1

    def __next__(self):
        if isinstance(self.background_tcr_dataloader.batch_sizes, list):
            self.batch_size_ind += 1
            if self.batch_size_ind >= len(self.background_tcr_dataloader.batch_sizes):
                raise StopIteration
            self.background_tcr_dataloader.batch_size = self.background_tcr_dataloader.batch_sizes[self.batch_size_ind]
        
        idx = np.random.choice(len(self.background_tcr_dataloader.background_tcr_dataset),\
                               size=self.background_tcr_dataloader.batch_size,\
                               replace=self.background_tcr_dataloader.replacement)
        return self.background_tcr_dataloader.background_tcr_dataset[idx.tolist()]


class background_species_tcr_dataset_class:
    def __init__(self, data_dir: str='./data/background_tcrs/',\
                 species: str="human",\
                 load_embedding: bool=False,\
                 load_size: int=10000) -> None:
        """Dataset class for one species 
        
        Parameters
        ----------
        data_dir: str
            The directory containing background TCRs
        species: str
            Species to load 
        load_embedding: bool
            Whether or not to load embedding file or string file 
        load_size: int 
            The number of samples to load from the background TCRs file
        """
        self.background_species_alpha_tcr_dataset = background_tcr_dataset_class(data_dir=data_dir,\
                                                                                 species=species,\
                                                                                 chain="alpha",\
                                                                                 load_embedding=load_embedding,\
                                                                                 load_size=load_size)
        self.background_species_beta_tcr_dataset = background_tcr_dataset_class(data_dir=data_dir,\
                                                                                 species=species,\
                                                                                 chain="beta",\
                                                                                 load_embedding=load_embedding,\
                                                                                 load_size=load_size)

    def __len__(self):
        return len(self.background_species_alpha_tcr_dataset) + len(self.background_species_beta_tcr_dataset)
                             
    def __getitem__(self, idx_tuple):
        idx_alpha, idx_beta = idx_tuple
        alpha_df = self.background_species_alpha_tcr_dataset[idx_alpha]
        beta_df = self.background_species_beta_tcr_dataset[idx_beta]
        return pd.concat([alpha_df, beta_df.set_index(alpha_df.index)], axis=1)


class background_species_tcr_dataloader_class:
    def __init__(self, background_species_tcr_dataset: background_species_tcr_dataset_class,\
                 batch_size: Union[int, list]=10, replacement: bool=True) -> None:
        """Dataloader class for a single species 

        Parameters
        ----------
        background_species_tcr_dataset: background_species_tcr_dataset_class
            The dataset class 
        batch_size: Union[int, list]
            Batch size to load the data. If it is a list, each iteration will load a size found in the list
        replacement: bool
            Whether or not sample with replacement 

        """
        self.background_species_alpha_tcr_dataloader = background_tcr_dataloader_class(background_species_tcr_dataset.background_species_alpha_tcr_dataset,\
                                                                                       batch_size=batch_size,
                                                                                       replacement=replacement)
        self.background_species_beta_tcr_dataloader = background_tcr_dataloader_class(background_species_tcr_dataset.background_species_beta_tcr_dataset,\
                                                                                       batch_size=batch_size,
                                                                                       replacement=replacement)

    def __iter__(self):
        return  background_species_tcr_dataset_iterator(self)

class background_species_tcr_dataset_iterator:
    def __init__(self, background_species_tcr_dataloader: background_species_tcr_dataloader_class) -> None:
        """Iterator class for a tcr dataset

        Parameters
        ----------
        background_species_tcr_dataloader: background_species_tcr_dataloader_class
            Dataloader for the dataset

        """
        self.alpha_iterator = iter(background_species_tcr_dataloader.background_species_alpha_tcr_dataloader)
        self.beta_iterator = iter(background_species_tcr_dataloader.background_species_beta_tcr_dataloader)
    
    def __next__(self):
        alpha_df = next(self.alpha_iterator)
        beta_df = next(self.beta_iterator)
        return pd.concat([alpha_df, beta_df.set_index(alpha_df.index)], axis=1)
        
        

        