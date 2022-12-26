# Data IO
import os 
import pandas as pd 

# Numeric manipulation 
import numpy as np 

# To entertain users 
from tqdm import tqdm 

# Typing 
from typing import Optional

# pMTnet_Omni modules 
from pMTnet_Omni.encoders.encoder_class import encoder_class
from pMTnet_Omni.classifier import pMHCTCR
from pMTnet_Omni.background_tcr_loaders import background_species_tcr_dataset_class, background_species_tcr_dataloader_class
from pMTnet_Omni.utilities import read_file, get_auroc, plot_roc_curve, batchify_check_size


class pMTnet_Omni_class:
    def __init__(self,
                 model_device: str = 'cpu',
                 background_data_dir: str = "./data/background_tcrs",
                 encoder_data_dir='./data/data_for_encoders/',
                 classifier_model_weights_dir='./data/classifier_model_weights/',
                 encoder_model_weights_dir='./data/encoder_model_weights/',
                 vGdVAEacheckpoint_path: Optional[str] = None,
                 vGdVAEbcheckpoint_path: Optional[str] = None,
                 cdr3VAEacheckpoint_path: Optional[str] = None,
                 cdr3VAEbcheckpoint_path: Optional[str] = None,
                 pMHCcheckpoint_path: Optional[str] = None) -> None:

        self.model_device = model_device
        self.background_data_dir = background_data_dir
        self.encoder_data_dir = encoder_data_dir
        self.classifier_model_weights_dir = classifier_model_weights_dir
        self.encoder_model_weights_dir = encoder_model_weights_dir
        # User df & embeddings
        self.user_df = None
        self.user_tcr_embedding = None
        self.user_pmhc_embedding = None
        self.user_df_output = None
        self.user_df_rank = None
        # Creat the encoder
        self.encoder = encoder_class(encoder_data_dir=encoder_data_dir,
                                     model_device=model_device,
                                     vGdVAEacheckpoint_path=vGdVAEacheckpoint_path,
                                     vGdVAEbcheckpoint_path=vGdVAEbcheckpoint_path,
                                     cdr3VAEacheckpoint_path=cdr3VAEacheckpoint_path,
                                     cdr3VAEbcheckpoint_path=cdr3VAEbcheckpoint_path,
                                     pMHCcheckpoint_path=pMHCcheckpoint_path)
        self.classifier_model = pMHCTCR().to(model_device)
        # LOAD MODE LHERE

    def read_user_df(self,
                     user_data_path: str = './test_data/test_df.csv',
                     sep: str = ",",
                     header: int = 0) -> None:
        self.user_df = read_file(file_path=user_data_path,
                                 sep=sep,
                                 header=header,
                                 background_tcr_dir=self.encoder_data_dir,
                                 mhc_path=os.path.join(self.encoder_data_dir, "valid_mhc.txt"))

    def encode_user_df(self) -> None:
        self.user_tcr_embedding, self.user_pmhc_embedding = self.encoder.encode(df=self.user_df,
                                                                                is_embedding=False)

    def get_user_df_output(self) -> None:
        self.user_result_output = self.classifier_model.predict(
            self.user_tcr_embedding, self.user_pmhc_embedding)

    def _generate_background_dataloader(self,
                                        load_embedding: bool = False,
                                        replacement: bool = True,
                                        check_size: list = [10, 20],
                                        load_size: int = 10):
        batch_size, batch_finished_indicator = batchify_check_size(check_size=check_size,
                                                                   load_size=load_size)
        background_human_dataset = background_species_tcr_dataset_class(data_dir=self.background_data_dir,
                                                                        species="human",
                                                                        load_embedding=load_embedding,
                                                                        load_size=load_size)
        background_mouse_dataset = background_species_tcr_dataset_class(data_dir=self.background_data_dir,
                                                                        species="mouse",
                                                                        load_embedding=load_embedding,
                                                                        load_size=load_size)
        background_human_dataloader = background_species_tcr_dataloader_class(background_human_dataset,
                                                                              batch_size=batch_size,
                                                                              replacement=replacement)
        background_mouse_dataloader = background_species_tcr_dataloader_class(background_mouse_dataset,
                                                                              batch_size=batch_size,
                                                                              replacement=replacement)
        return batch_size, batch_finished_indicator, background_human_dataloader, background_mouse_dataloader

    def predict(self,
                rank_threshold: float = 0.03,
                B: int = 2,
                load_embedding: bool = False,
                replacement: bool = True,
                check_size: list = [10, 20],
                load_size: int = 10):
        n_rows = self.user_df.shape[0]

        tcr_species_df = self.user_df["tcr_species"].to_frame()

        self.user_df_rank = {}
        for b in range(B):
            self.user_df_rank[b] = np.array([])
            batch_size, batch_finished_indicator, background_human_dataloader, background_mouse_dataloader = self._generate_background_dataloader(load_embedding=load_embedding,
                                                                                                                                                  replacement=replacement,
                                                                                                                                                  check_size=check_size,
                                                                                                                                                  load_size=load_size)
            for row in tqdm(range(n_rows)):
                if tcr_species_df.iloc[row, 0] == "human":
                    data_loader = background_human_dataloader
                else:
                    data_loader = background_mouse_dataloader

                batch_results = np.array([])
                for batch_size_ind, tcrs in enumerate(data_loader):
                    b_size = batch_size[batch_size_ind]
                    #####################################
                    # DEPENDING ON WHETHER THE USER WANTS TO LOAD EMBEDDING OR NOT
                    # THIS PART OF THE CODE WILL CHANGE
                    background_tcr_embedding, _ = self.encoder.encode(
                        df=tcrs, is_embedding=load_embedding)
                    duplicated_pmhc_embedding = self.user_pmhc_embedding[row, :].repeat(
                        b_size, 1)
                    #####################################
                    background_result = self.classifier_model.predict(
                        background_tcr_embedding, duplicated_pmhc_embedding)
                    batch_results = np.append(batch_results, background_result)
                    if batch_finished_indicator[batch_size_ind]:
                        # COMPUT RANK HERE
                        batch_results = np.append(
                            batch_results, self.user_df_output[row])
                        temp_ranks = batch_results.argsort().argsort()
                        temp_rank = temp_ranks[b_size]/(b_size+1)
                        if (temp_rank > rank_threshold) or (b_size == np.max(batch_size)):
                            self.user_df_rank[b] = np.append(
                                self.user_df_rank[b], values=temp_rank)
                            break
                        batch_results = np.array([])
    def validate(self, true_labels):
        temp = np.array([self.user_df_rank[0][0] for i in range(10)]) + np.random.standard_normal(10)
        auc = get_auroc(np.random.binomial(1, 0.5, 10), temp)
        plot_roc_curve(np.random.binomial(1, 0.5, 10), temp)
        return auc



# def predict(user_data_path: str='./test_data/test_df.csv',
#             sep: str=",",
#             header: int=0,
#             B: int = 2,
#             rank_threshold: float=0.03,
#             model_device: str='cpu',
#             load_embedding: bool=False,
#             background_data_dir: str="./data/background_tcrs",
#             mhc_path: str='./data/data_for_encoders/valid_mhc.txt',
#             encoder_data_dir='./data/data_for_encoders/',
#             vGdVAEacheckpoint_path: Optional[str]=None,\
#             vGdVAEbcheckpoint_path: Optional[str]=None,\
#             cdr3VAEacheckpoint_path: Optional[str]=None,\
#             cdr3VAEbcheckpoint_path: Optional[str]=None,\
#             pMHCcheckpoint_path: Optional[str]=None,
#             replacement: bool=True,
#             check_size: list=[10, 20],
#             load_size: int=10) -> dict:
    
#     # Read in user dataframe 
#     user_df = read_file(file_path=user_data_path,
#                         sep=sep,
#                         header=header,
#                         background_tcr_dir=background_data_dir,
#                         mhc_path=mhc_path)
#     # Calculate the batch sizes for the dataloader class 
#     batch_size, batch_finished_indicator = batchify_check_size(check_size=check_size,
#                                                               load_size=load_size)
    
#     # Creat the encoder
#     encoder = encoder_class(encoder_data_dir=encoder_data_dir,
#                             model_device=model_device,
#                             vGdVAEacheckpoint_path=vGdVAEacheckpoint_path,
#                             vGdVAEbcheckpoint_path=vGdVAEbcheckpoint_path,
#                             cdr3VAEacheckpoint_path=cdr3VAEacheckpoint_path,
#                             cdr3VAEbcheckpoint_path=cdr3VAEbcheckpoint_path,
#                             pMHCcheckpoint_path=pMHCcheckpoint_path)
#     user_tcr_embedding, user_pmhc_embedding = encoder.encode(df=user_df,
#                                                              is_embedding=load_embedding)

#     ####
#     ## SET MODEL HERE 
#     classifier_model = pMHCTCR().to(model_device)
#     # LOAD MODE LHERE 

#     user_result_tensor = classifier_model.predict(user_tcr_embedding, user_pmhc_embedding)
    
#     user_df_output = user_result_tensor.numpy()
    
#     n_rows = user_df.shape[0]

#     tcr_species_df = user_df["tcr_species"].to_frame()

#     user_df_rank = {}

#     for b in range(B):
#         # For each b, we reload the dataset 
#         # This is potentially the slowest part of the algorithm 
#         user_df_rank[b] = np.array([])
#         background_human_dataset = background_species_tcr_dataset_class(data_dir=background_data_dir,\
#                                                                         species="human",\
#                                                                         load_embedding=load_embedding,\
#                                                                         load_size=load_size)
#         background_mouse_dataset = background_species_tcr_dataset_class(data_dir=background_data_dir,\
#                                                                         species="mouse",\
#                                                                         load_embedding=load_embedding,\
#                                                                         load_size=load_size) 
#         background_human_dataloader = background_species_tcr_dataloader_class(background_human_dataset,\
#                                                                             batch_size=batch_size,\
#                                                                             replacement=replacement)
#         background_mouse_dataloader = background_species_tcr_dataloader_class(background_mouse_dataset,\
#                                                                             batch_size=batch_size,\
#                                                                             replacement=replacement)                                                                    
#         for row in tqdm(range(n_rows)):
#             if tcr_species_df.iloc[row,0] == "human":
#                 data_loader = background_human_dataloader
#             else:
#                 data_loader = background_mouse_dataloader
            
#             batch_results = np.array([])
#             for batch_size_ind, tcrs in enumerate(data_loader):
#                 b_size = batch_size[batch_size_ind]
#                 #####################################
#                 # DEPENDING ON WHETHER THE USER WANTS TO LOAD EMBEDDING OR NOT 
#                 # THIS PART OF THE CODE WILL CHANGE 
#                 background_tcr_embedding, _ = encoder.encode(tcrs, False)
#                 duplicated_pmhc_embedding = user_pmhc_embedding[row, :].repeat(b_size, 1)
#                 #####################################
#                 background_result = classifier_model.predict(background_tcr_embedding, duplicated_pmhc_embedding)
#                 background_result = background_result.numpy()
#                 batch_results = np.append(batch_results, background_result)
#                 if batch_finished_indicator[batch_size_ind]:
#                     # COMPUT RANK HERE 
#                     batch_results = np.append(batch_results, user_df_output[row])
#                     temp_ranks = batch_results.argsort().argsort()
#                     temp_rank = temp_ranks[b_size]/(b_size+1)
#                     if (temp_rank > rank_threshold) or (b_size == np.max(batch_size)):
#                         user_df_rank[b] = np.append(user_df_rank[b], values=temp_rank)
#                         break
#                     batch_results = np.array([])
#     temp = np.array([user_df_rank[0][0] for i in range(10)]) + np.random.standard_normal(10)
#     auc = get_auroc(np.random.binomial(1, 0.5, 10), temp)
#     plot_roc_curve(np.random.binomial(1, 0.5, 10), temp)
#     return user_df_rank             

