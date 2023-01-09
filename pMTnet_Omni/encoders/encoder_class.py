# Data IO
import os
import pandas as pd
import csv

# Numeric manipulation
import numpy as np

# PyTorch
import torch

# User entertainment
from tqdm import tqdm

# Typing
from typing import Optional, Tuple

# Subclasses
from pMTnet_Omni.encoders.tcr_encoder_class import tcr_encoder_class
from pMTnet_Omni.encoders.pmhc_encoder_class import pmhc_encoder_class


class encoder_class:
    def __init__(self,
                 encoder_data_dir: str,
                 model_device: str = 'cpu',
                 vGdVAEacheckpoint_path: Optional[str] = None,
                 vGdVAEbcheckpoint_path: Optional[str] = None,
                 cdr3VAEacheckpoint_path: Optional[str] = None,
                 cdr3VAEbcheckpoint_path: Optional[str] = None,
                 pMHCcheckpoint_path: Optional[str] = None) -> None:
        """The main encoder class 

        Parameters
        ----------
        encoder_data_dir: str
            The directory containing data for encoders 
        model_device: str
            cpu or gpu
        vGdVAEacheckpoint_path: Optional[str]
            The path to VA the encoder
        vGdVAEbcheckpoint_path: Optional[str]
            The path to VB the encoder
        cdr3VAEacheckpoint_path: Optional[str]
            The path to CDR3 Alpha encoder
        cdr3VAEbcheckpoint_path: Optional[str]
            The path to CDR3 Beta encoder
        pMHCcheckpoint_path: Optional[str]
            The path to pMHC the encoder
        
        """
        # Build an amino acid dictionary
        # to convert strings to numeric vectors
        print("Building Atchley Factors dictionary\n")
        self.aa_dict_atchley = dict()
        with open(encoder_data_dir + 'Atchley_factors.csv', 'r') as aa:
            aa_reader = csv.reader(aa)
            next(aa_reader, None)
            for rows in aa_reader:
                aa_name = rows[0]
                aa_factor = rows[1:len(rows)]
                self.aa_dict_atchley[aa_name] = np.asarray(
                    aa_factor, dtype='float')
        # Build mhc dictionary
        # to convert strings to numeric vectors
        # Since mhc dictionary is huge
        # we will only read in the file names
        # each file should be an esm tensor
        print("Building MHC dictionary... This might take a few seconds\n")
        mhc_files = os.listdir(encoder_data_dir + 'name_dict')
        self.mhc_dict = {}
        for file_name in tqdm(mhc_files):
            with open(encoder_data_dir + "name_dict/"+file_name) as f:
                line = f.readline()
            self.mhc_dict[line] = encoder_data_dir + "mhc_dict/" + file_name

        # Load models
        self.model_device = model_device

        # Initialize two encoders
        print("Initializing TCR+CDR3 encoders\n")
        self.tcr_encoder = tcr_encoder_class(model_device=self.model_device,
                                             vGdVAEacheckpoint_path=vGdVAEacheckpoint_path,
                                             vGdVAEbcheckpoint_path=vGdVAEbcheckpoint_path,
                                             cdr3VAEacheckpoint_path=cdr3VAEacheckpoint_path,
                                             cdr3VAEbcheckpoint_path=cdr3VAEbcheckpoint_path)
        print("Initializing pMHC encoders\n")
        self.pmhc_encoder = pmhc_encoder_class(model_device=self.model_device,
                                               pMHCcheckpoint_path=pMHCcheckpoint_path)

    def encode(self,
               df: pd.DataFrame,
               is_embedding: bool,
               verbose: bool=False) -> Tuple[Optional[torch.tensor], Optional[torch.tensor]]:
        """Encode TCRs and pMHCs
        
        This function encodes TCRs and pMHCs
        If only TCRs are present in the dataframe, which can happen when comparing with background
        then only TCRs will be encoded

        Parameters
        ------------
        df: pd.DataFrame
            A dataframe containing data to be encoded 
        is_embedding: bool
            Whether or not the TCRs are already embeddings 
        
        Returns
        ----------
        Tuple[Optional[torch.tensor], Optional[torch.tensor]]
            Embeddings of TCRs and pMHCs
        
        """
        tcr_embedding = None
        pmhc_embedding = None
        if is_embedding:
            # if TCRs are embeddings
            # This only happens when comparing with background
            # In this case, the dataframe would only contain
            # data related to TCRs
            tcr_embedding = self.tcr_encoder.encode(df=df,
                                                    aa_dict_atchley=self.aa_dict_atchley,
                                                    is_embedding=True,
                                                    verbose=False)
        else:
            # Otherwise, we check if required columns are
            # present in the dataframe
            # If so, we encode
            # If not, we return None
            tcr_columns = ["vaseq", "vbseq", "cdr3a", "cdr3b"]
            if all([name in df.columns for name in tcr_columns]):
                tcr_embedding = self.tcr_encoder.encode(df=df[tcr_columns],
                                                        aa_dict_atchley=self.aa_dict_atchley,
                                                        is_embedding=False,
                                                        verbose=verbose)
            pmhc_columns = ["peptide", "mhca", "mhcb"]
            if all([name in df.columns for name in pmhc_columns]):
                pmhc_embedding = self.pmhc_encoder.encode(df=df[pmhc_columns],
                                                          aa_dict_atchley=self.aa_dict_atchley,
                                                          mhc_dict=self.mhc_dict,
                                                          verbose=verbose)
        return tcr_embedding, pmhc_embedding
