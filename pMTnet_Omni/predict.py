# Data IO
import pandas as pd 
import numpy as np 
import math
# To entertain users 
from tqdm import tqdm 
# DL model 
import torch

from pMTnet_Omni.utilities import read_file, get_auroc, plot_roc_curve
from pMTnet_Omni.encoders.encoder_class import encoder_class
from pMTnet_Omni.classifier import pMHCTCR
from pMTnet_Omni.background_tcr_loaders import background_species_tcr_dataset_class, background_species_tcr_dataloader_class

# 
# SOMETHING RELATED TO THE MODEL HERE

# common_column_names = ["va", "cdr3a", "vaseq", "vb", "cdr3b", "vbseq",\
#                         "peptide", "mhca", "mhcb", "tcr_species", "pmhc_species"]
background_data_dir = "./data/background_tcrs"
B = 2
check_size = [10, 20]
load_size = 10

batch_size = []
batch_finished_indicator = []
for size in check_size:
    if size <= load_size:
        batch_size.append(size)
        batch_finished_indicator.append(True)
    else:
        n_batch = math.ceil(size / load_size)
        i=1
        while i < n_batch:
            batch_size.append(load_size)
            batch_finished_indicator.append(False)
            i += 1
        batch_finished_indicator.append(True)

load_embedding = False
replacement = True

model_device = 'cpu'
rank_threshold = 0.03

user_data_path = './test_data/test_df.csv'
# user_df = read_file(file_path = user_data_path, sep="\t", header=0)
user_df = read_file(file_path=user_data_path, sep=",", header=0,\
             background_tcr_dir='./data/background_tcrs',\
             mhc_path='./data/data_for_encoders/valid_mhc.txt')


encoder = encoder_class(encoder_data_dir='./data/data_for_encoders/')
user_tcr_embedding, user_pmhc_embedding = encoder.encode(user_df, False)

# Regardless of the background tcrs,
# we generate the model output 

####
## SET MODEL HERE 
pMTnet_Omni_model = pMHCTCR().to(model_device)
# LOAD MODE LHERE 

pMTnet_Omni_model.eval()
with torch.no_grad():
    user_result_tensor = pMTnet_Omni_model(user_tcr_embedding, user_pmhc_embedding)
 

user_df_output = user_result_tensor.numpy()

n_rows = user_df.shape[0]

tcr_species_df = user_df["tcr_species"].to_frame()

user_df_rank = {}

pMTnet_Omni_model.eval()
with torch.no_grad():
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
                background_result = pMTnet_Omni_model(background_tcr_embedding, duplicated_pmhc_embedding)
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