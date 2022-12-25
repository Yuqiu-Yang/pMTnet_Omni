# Data IO
import pandas as pd 

# Numeric manipulation 
import numpy as np 

# To entertain users 
from tqdm import tqdm 

# DL model 
import torch

# Typing 
from typing import Optional

# pMTnet_Omni modules 
from pMTnet_Omni.encoders.encoder_class import encoder_class
from pMTnet_Omni.classifier import pMHCTCR
from pMTnet_Omni.background_tcr_loaders import background_species_tcr_dataset_class, background_species_tcr_dataloader_class
from pMTnet_Omni.utilities import read_file, get_auroc, plot_roc_curve, batchify_check_size


def predict(user_data_path: str='./test_data/test_df.csv',
            sep: str=",",
            header: int=0,
            B: int = 2,
            rank_threshold: float=0.03,
            model_device: str='cpu',
            load_embedding: bool=False,
            background_data_dir: str="./data/background_tcrs",
            mhc_path: str='./data/data_for_encoders/valid_mhc.txt',
            encoder_data_dir='./data/data_for_encoders/',
            vGdVAEacheckpoint_path: Optional[str]=None,\
            vGdVAEbcheckpoint_path: Optional[str]=None,\
            cdr3VAEacheckpoint_path: Optional[str]=None,\
            cdr3VAEbcheckpoint_path: Optional[str]=None,\
            pMHCcheckpoint_path: Optional[str]=None,
            replacement: bool=True,
            check_size: list=[10, 20],
            load_size: int=10) -> dict:
    
    # Read in user dataframe 
    user_df = read_file(file_path=user_data_path,
                        sep=sep,
                        header=header,
                        background_tcr_dir=background_data_dir,
                        mhc_path=mhc_path)
    # Calculate the batch sizes for the dataloader class 
    batch_size, batch_finished_indicator = batchify_check_size(check_size=check_size,
                                                              load_size=load_size)
    
    # Creat the encoder
    encoder = encoder_class(encoder_data_dir=encoder_data_dir,
                            model_device=model_device,
                            vGdVAEacheckpoint_path=vGdVAEacheckpoint_path,
                            vGdVAEbcheckpoint_path=vGdVAEbcheckpoint_path,
                            cdr3VAEacheckpoint_path=cdr3VAEacheckpoint_path,
                            cdr3VAEbcheckpoint_path=cdr3VAEbcheckpoint_path,
                            pMHCcheckpoint_path=pMHCcheckpoint_path)
    user_tcr_embedding, user_pmhc_embedding = encoder.encode(df=user_df,
                                                             is_embedding=load_embedding)

    ####
    ## SET MODEL HERE 
    pMTnet_Omni_model = pMHCTCR().to(model_device)
    # LOAD MODE LHERE 

    user_result_tensor = pMTnet_Omni_model.predict(user_tcr_embedding, user_pmhc_embedding)
    
    user_df_output = user_result_tensor.numpy()
    
    n_rows = user_df.shape[0]

    tcr_species_df = user_df["tcr_species"].to_frame()

    user_df_rank = {}

    for b in range(B):
        # For each b, we reload the dataset 
        # This is potentially the slowest part of the algorithm 
        user_df_rank[b] = np.array([])
        background_human_dataset = background_species_tcr_dataset_class(data_dir=background_data_dir,\
                                                                        species="human",\
                                                                        load_embedding=load_embedding,\
                                                                        load_size=load_size)
        background_mouse_dataset = background_species_tcr_dataset_class(data_dir=background_data_dir,\
                                                                        species="mouse",\
                                                                        load_embedding=load_embedding,\
                                                                        load_size=load_size) 
        background_human_dataloader = background_species_tcr_dataloader_class(background_human_dataset,\
                                                                            batch_size=batch_size,\
                                                                            replacement=replacement)
        background_mouse_dataloader = background_species_tcr_dataloader_class(background_mouse_dataset,\
                                                                            batch_size=batch_size,\
                                                                            replacement=replacement)                                                                    
        for row in tqdm(range(n_rows)):
            if tcr_species_df.iloc[row,0] == "human":
                data_loader = background_human_dataloader
            else:
                data_loader = background_mouse_dataloader
            
            batch_results = np.array([])
            for batch_size_ind, tcrs in enumerate(data_loader):
                b_size = batch_size[batch_size_ind]
                #####################################
                # DEPENDING ON WHETHER THE USER WANTS TO LOAD EMBEDDING OR NOT 
                # THIS PART OF THE CODE WILL CHANGE 
                background_tcr_embedding, _ = encoder.encode(tcrs, False)
                duplicated_pmhc_embedding = user_pmhc_embedding[row, :].repeat(b_size, 1)
                #####################################
                background_result = pMTnet_Omni_model.predict(background_tcr_embedding, duplicated_pmhc_embedding)
                background_result = background_result.numpy()
                batch_results = np.append(batch_results, background_result)
                if batch_finished_indicator[batch_size_ind]:
                    # COMPUT RANK HERE 
                    batch_results = np.append(batch_results, user_df_output[row])
                    temp_ranks = batch_results.argsort().argsort()
                    temp_rank = temp_ranks[b_size]/(b_size+1)
                    if (temp_rank > rank_threshold) or (b_size == np.max(batch_size)):
                        user_df_rank[b] = np.append(user_df_rank[b], values=temp_rank)
                        break
                    batch_results = np.array([])
    temp = np.array([user_df_rank[0][0] for i in range(10)]) + np.random.standard_normal(10)
    auc = get_auroc(np.random.binomial(1, 0.5, 10), temp)
    plot_roc_curve(np.random.binomial(1, 0.5, 10), temp)
    return user_df_rank             

