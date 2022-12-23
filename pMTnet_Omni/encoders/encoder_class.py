import os 
import csv 
import pandas as pd 
import numpy as np 

import torch 

from typing import Optional

from pMTnet_Omni.encoders.tcr_encoder_class import tcr_encoder_class
from pMTnet_Omni.encoders.pmhc_encoder_class import pmhc_encoder_class

class encoder_class:
    def __init__(self,\
                 encoder_data_dir: str,\
                 model_device: str='cpu',\
                 vGdVAEacheckpoint_path: Optional[str]=None,\
                 vGdVAEbcheckpoint_path: Optional[str]=None,\
                 cdr3VAEacheckpoint_path: Optional[str]=None,\
                 cdr3VAEbcheckpoint_path: Optional[str]=None,\
                 pMHCcheckpoint_path: Optional[str]=None):
        # aa_dict_dir='/work/DPDS/s213303/pmtnetv2/test_data/pmtnetv1/pMTnet-master/library/Atchley_factors.csv'
        # Build an amino acid dictionary 
        # to convert strings to numeric vectors 
        self.aa_dict_atchley=dict()
        with open(encoder_data_dir + 'Atchley_factors.csv', 'r') as aa:
            aa_reader=csv.reader(aa)      
            next(aa_reader, None)
            for rows in aa_reader:
                aa_name=rows[0]
                aa_factor=rows[1:len(rows)]
                self.aa_dict_atchley[aa_name]=np.asarray(aa_factor,dtype='float') 
        # Build mhc dictionary 
        # to convert strings to numeric vectors 
        # Since mhc dictionary is huge
        # we will only read in the file names 
        # each file should be an esm tensor
        mhc_files = os.listdir(encoder_data_dir + 'name_dict')
        self.mhc_dict = {}
        for file_name in mhc_files:
            with open(encoder_data_dir + "name_dict/"+file_name) as f:
                line = f.readline()
            self.mhc_dict[line] = encoder_data_dir + "mhc_dict/" + file_name

        
        # Load models 
        self.model_device = model_device

        # Initialize two encoders 
        self.tcr_encoder = tcr_encoder_class(model_device=self.model_device,\
                                             vGdVAEacheckpoint_path=vGdVAEacheckpoint_path,\
                                             vGdVAEbcheckpoint_path=vGdVAEbcheckpoint_path,\
                                             cdr3VAEacheckpoint_path=cdr3VAEacheckpoint_path,\
                                             cdr3VAEbcheckpoint_path=cdr3VAEbcheckpoint_path)
        self.pmhc_encoder = pmhc_encoder_class(model_device=self.model_device,\
                                               pMHCcheckpoint_path=pMHCcheckpoint_path)   

    def encode(self, source_dataset, is_embedding):
        tcr_embedding = None
        pmhc_embedding = None
        tcr_columns = ["vaseq", "vbseq", "cdr3a", "cdr3b"]
        if is_embedding:
            tcr_embedding = torch.tensor(source_dataset, dtype=torch.float32)
        else:
            if all([name in source_dataset.columns for name in tcr_columns]):
                tcr_embedding = self.tcr_encoder.encode(source_dataset[tcr_columns],\
                                                        self.aa_dict_atchley)    
            pmhc_columns = ["peptide", "mhca", "mhcb"]
            if all([name in source_dataset.columns for name in pmhc_columns]):
                pmhc_embedding = self.pmhc_encoder.encode(source_dataset[pmhc_columns],\
                                                        self.aa_dict_atchley,\
                                                        self.mhc_dict)
        return tcr_embedding, pmhc_embedding
 
