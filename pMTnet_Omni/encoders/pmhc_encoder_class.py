import os 
import pandas as pd 
import numpy as np
import csv 

import torch

from typing import Optional 

from pMTnet_Omni.encoders.pmhc_encoder_model import pMHC
from pMTnet_Omni.encoders.utilities import peptide_map
from pMTnet_Omni.encoders.utilities import mhc_map

class pmhc_encoder_class:
    def __init__(self,\
                 model_device: str,\
                 pMHCcheckpoint_path: Optional[str]=None):
        self.model_device = model_device
        self.pmhc_model = pMHC().to(model_device)
        if pMHCcheckpoint_path is not None:
            pMHCcheckpoint = torch.load(pMHCcheckpoint_path,map_location=model_device)
            self.pmhc_model.load_state_dict(pMHCcheckpoint['net'])
        self.pmhc_model.eval()
    def encode(self, df, aa_dict_atchley, mhc_dict):
        with torch.no_grad():
            x_p = torch.Tensor(peptide_map(df, aa_dict_atchley, "peptide", 30)).to(self.model_device)
            x_a = torch.Tensor(mhc_map(df, "mhca", mhc_dict)).to(self.model_device)
            x_b = torch.Tensor(mhc_map(df, "mhcb", mhc_dict)).to(self.model_device)
            pmhc_embedding, _ = self.pmhc_model(x_p, x_a, x_b)
            return pmhc_embedding
    
