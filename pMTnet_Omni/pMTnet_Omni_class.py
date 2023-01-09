# Data IO
import os 
import pandas as pd 
from copy import deepcopy

# Numeric manipulation 
import numpy as np 

# PyTorch
import torch

# To entertain users 
from tqdm import tqdm 
from contextlib import redirect_stdout
from matplotlib.backends.backend_pdf import PdfPages

# Typing 
from typing import Optional

# pMTnet_Omni modules 
from pMTnet_Omni.encoders.encoder_class import encoder_class
from pMTnet_Omni.classifier import pMHCTCR
from pMTnet_Omni.background_tcr_loaders import background_species_tcr_dataset_class, background_species_tcr_dataloader_class
from pMTnet_Omni.utilities import read_file, get_auroc, plot_roc_curve, batchify_check_size, setup_seed, get_mhc_class


class pMTnet_Omni_class:
    def __init__(self,
                 model_device: str = 'cpu',
                 background_data_dir: str = "./data/background_tcrs",
                 encoder_data_dir='./data/data_for_encoders/',
                 classifier_checkpoint_path: Optional[str]=None,
                 vGdVAEacheckpoint_path: Optional[str]=None,
                 vGdVAEbcheckpoint_path: Optional[str]=None,
                 cdr3VAEacheckpoint_path: Optional[str]=None,
                 cdr3VAEbcheckpoint_path: Optional[str]=None,
                 pMHCcheckpoint_path: Optional[str]=None,
                 seed: Optional[int]=None) -> None:

        self.model_device = model_device
        self.background_data_dir = background_data_dir
        self.encoder_data_dir = encoder_data_dir
        # Set seed
        if seed is not None:
            setup_seed(seed=seed)

        # User df & embeddings
        self.user_df = None
        self.user_tcr_embedding = None
        self.user_pmhc_embedding = None
        self.user_df_output = None
        self.user_df_rank = None
        # Create the encoder
        self.encoder = encoder_class(encoder_data_dir=encoder_data_dir,
                                     model_device=model_device,
                                     vGdVAEacheckpoint_path=vGdVAEacheckpoint_path,
                                     vGdVAEbcheckpoint_path=vGdVAEbcheckpoint_path,
                                     cdr3VAEacheckpoint_path=cdr3VAEacheckpoint_path,
                                     cdr3VAEbcheckpoint_path=cdr3VAEbcheckpoint_path,
                                     pMHCcheckpoint_path=pMHCcheckpoint_path)
        
        # Initialize th clssifier model 
        self.classifier_model = pMHCTCR().to(model_device)
        # LOAD MODE LHERE
        print("Attempt to load the classifier model\n")
        if classifier_checkpoint_path is not None:
            classifier_checkpoint = torch.load(
                classifier_checkpoint_path, map_location=model_device)
            self.classifier_model.load_state_dict(classifier_checkpoint['net'])
            print("Success\n")
        else:
            print("Check point is not provided. Use random weights\n")


    def read_user_df(self,
                     user_data_path: str,
                     sep: str = ",",
                     header: int = 0) -> None:
        self.user_df = read_file(file_path=user_data_path,
                                 sep=sep,
                                 header=header,
                                 background_tcr_dir=self.encoder_data_dir,
                                 mhc_path=os.path.join(self.encoder_data_dir, "valid_mhc.txt"))

    def encode_user_df(self,
                       verbose: bool=True,
                       create_incomplete_data: bool=False) -> None:
        self.user_tcr_embedding, self.user_pmhc_embedding = self.encoder.encode(df=self.user_df,
                                                                                is_embedding=False,
                                                                                verbose=verbose)
        self.user_df['is_complete'] = "complete"

        if create_incomplete_data:
            # Zero out TCR Alpha chain 
            # Manipulate the dataframe 
            # not necessary but might be useful when debugging
            df_alpha_missing = deepcopy(self.user_df)
            df_alpha_missing['is_complete'] = "alpha missing"
            df_alpha_missing['va'] = ""
            df_alpha_missing['vaseq'] = ""
            df_alpha_missing['cdr3a'] = ""
            # Manipulate tcr embedding 
            tcr_embedding_alpha_missing = self.user_tcr_embedding.detach().clone()
            # The embedding order is va (5) vb (5) cdr3a (30) cdr3b (30)
            tcr_embedding_alpha_missing[:, 0:5] = 0
            tcr_embedding_alpha_missing[:, 10:40]=0
            # Simple clone pmhc embedding 
            pmhc_embedding_alpha_missing = self.user_pmhc_embedding.detach().clone()
            
            # Zero out TCR Beta chain 
            # Manipulate the dataframe 
            # not necessary but might be useful when debugging
            df_beta_missing = deepcopy(self.user_df)
            df_beta_missing['is_complete'] = "beta missing"
            df_beta_missing['vb'] = ""
            df_beta_missing['vbseq'] = ""
            df_beta_missing['cdr3b'] = ""
            # Manipulate tcr embedding 
            tcr_embedding_beta_missing = self.user_tcr_embedding.detach().clone()
            # The embedding order is va (5) vb (5) cdr3a (30) cdr3b (30)
            tcr_embedding_beta_missing[:, 5:10] = 0
            tcr_embedding_beta_missing[:, 40:70]=0
            # Simple clone pmhc embedding 
            pmhc_embedding_beta_missing = self.user_pmhc_embedding.detach().clone()

            # Stack all three of them 
            self.user_df = pd.concat([self.user_df, df_alpha_missing, df_beta_missing], axis=0, ignore_index=True) 
            self.user_tcr_embedding = torch.concat([self.user_tcr_embedding, tcr_embedding_alpha_missing, tcr_embedding_beta_missing], dim=0)
            self.user_pmhc_embedding = torch.concat([self.user_pmhc_embedding, pmhc_embedding_alpha_missing, pmhc_embedding_beta_missing], dim=0)

    def get_user_df_output(self) -> None:
        self.user_df_output = self.classifier_model.predict(
            torch.cat((self.user_pmhc_embedding, self.user_tcr_embedding), dim=1))

    def _generate_background_dataloader(self,
                                        load_embedding: bool,
                                        replacement: bool,
                                        check_size: list,
                                        load_size: int,
                                        minibatch_size: int):
        batch_size, batch_finished_indicator = batchify_check_size(check_size=check_size,
                                                                  minibatch_size=minibatch_size)
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
                check_size: list = [1000, 10000, 100000],
                load_size: int = 1000000,
                minibatch_size: int = 50000,
                log_file_path: str="./pmtnet_omni_predict_log.txt"):
        with open(log_file_path, 'w') as f:
            with redirect_stdout(f):
                # Make sure check size is in an ascending order 
                check_size = np.sort(check_size)
                
                n_rows = self.user_df.shape[0]

                tcr_species_df = self.user_df["tcr_species"].to_frame()

                self.user_df_rank = {}
                for b in range(B):
                    print("Starting trial "+str(b+1)+" out of "+str(B)+" total tries\n")
                    self.user_df_rank[b] = np.array([])
                    print("Loading background TCR datasets\n")
                    batch_size, batch_finished_indicator, background_human_dataloader, background_mouse_dataloader = self._generate_background_dataloader(load_embedding=load_embedding,
                                                                                                                                                        replacement=replacement,
                                                                                                                                                        check_size=check_size,
                                                                                                                                                        load_size=load_size,
                                                                                                                                                        minibatch_size=minibatch_size)
                    for row in tqdm(range(n_rows), position=0, leave=True):
                        if tcr_species_df.iloc[row, 0] == "human":
                            data_loader = background_human_dataloader
                        else:
                            data_loader = background_mouse_dataloader
                        print(self.user_df.iloc[[row],:])
                        check_size_ind = 0
                        user_output_position = 0
                        b_size_so_far = 0
                        for batch_size_ind, tcrs in enumerate(data_loader):
                            b_size = batch_size[batch_size_ind]
                            b_size_so_far += b_size
                            print("Verifying using "+str(check_size[check_size_ind])+ " background tcrs\n")
                            print("\tLoading minibatch of "+str(tcrs.shape[0])+ " tcrs\n")
                            #####################################
                            # DEPENDING ON WHETHER THE USER WANTS TO LOAD EMBEDDING OR NOT
                            # THIS PART OF THE CODE WILL CHANGE
                            print("\tEncoding background tcrs\n")
                            background_tcr_embedding, _ = self.encoder.encode(
                                df=tcrs, is_embedding=load_embedding)
                            duplicated_pmhc_embedding = self.user_pmhc_embedding[row, :].repeat(
                                b_size, 1)
                            #####################################
                            print("\tPredicting background tcrs\n")
                            background_result = self.classifier_model.predict(
                                torch.cat((duplicated_pmhc_embedding, background_tcr_embedding), dim=1))
                            # Compute how many backgrounds are "better" than user output
                            current_position = np.sum(background_result > self.user_df_output[row])
                            print("\tRank in the current minibatch is "+str(current_position+1)+"th out of "+str(b_size+1)+"\n")
                            user_output_position += current_position
                            print("\tRank in the current batch so far is "+str(user_output_position+1)+"th out of "+str(b_size_so_far+1)+"\n")
                            if batch_finished_indicator[batch_size_ind]:
                                rank_percentile = user_output_position/(check_size[check_size_ind]+1)
                                print("\tCurrent batch is done. The final position is "+str(user_output_position+1)+" out of "+str(check_size[check_size_ind]+1)+"\n") 
                                if (rank_percentile > rank_threshold) or (b_size == np.max(batch_size)):
                                    self.user_df_rank[b] = np.append(
                                        self.user_df_rank[b], values=rank_percentile)
                                    break
                                user_output_position = 0
                                b_size_so_far = 0
                                check_size_ind += 1

    def validate(self, 
                 true_labels: pd.DataFrame,
                 roc_plot_path: str="./roc_plots.pdf") -> float:
        
        # Divide the datasets based on missingness 
        missingness = ['complete', 'alpha missing', 'beta missing']
        result_dfs = {key: None for key in missingness}

        # For each missingness type, we also compute different auroc types 
        auroc_to_compute = ["overall", "human overall", "mouse overall",
                            "human class i", "human class ii", 
                            "mouse class i", "mouse class ii"]
        
        # We assume that true_labels is a one-column dataframe
        truth = pd.DataFrame({'true_label': true_labels.iloc[:,0]})
        
        pp = PdfPages(roc_plot_path)
        for m in missingness:
            ind = (self.user_df['is_complete'] == m)
            
            if np.sum(ind) < 1:
                continue

            # First get mhc classes 
            mhc_class = get_mhc_class(self.user_df[ind].reset_index())
            pred = pd.DataFrame({'pred': self.user_df_rank[0][ind]})

            # Then we concatenate truth, predicted and mhc_class into one dataframe 
            df = pd.concat([truth, pred.set_index(truth.index), mhc_class.set_index(truth.index)], axis=1)
            
            auc_dict = {key: [None] for key in auroc_to_compute}
        
            for auc_type in auroc_to_compute:
                if auc_type == "overall":
                    # If the column contains anyting
                    ind = df['mhc_class'].str.contains('.')
                elif auc_type == "human overall":
                    ind = df['mhc_class'].str.contains('human')
                elif auc_type == "mouse overall":
                    ind = df['mhc_class'].str.contains('mouse')
                else:
                    ind = df['mhc_class'].str.contains(auc_type)

                if np.sum(ind) < 1:
                    continue
                
                auc = get_auroc(true_labels=df[ind]['true_label'], predicted_labels=df[ind]['pred'])
                auc_dict[auc_type][0] = auc
                pp.savefig(plot_roc_curve(true_labels=df[ind]['true_label'], 
                                         predicted_labels=df[ind]['pred'],
                                         label=m+" "+auc_type+": "+str(auc)))
            
            # Convert the results to dataframe    
            result_dfs[m] = pd.DataFrame(auc_dict)
        pp.close()
        return result_dfs
