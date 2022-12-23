import os 
import pandas as pd 
import numpy as np
import csv 

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from pMTnet_Omni.encoders.v_alpha_encoder_model import vGdVAEa
from pMTnet_Omni.encoders.v_beta_encoder_model import vGdVAEb
from pMTnet_Omni.encoders.cdr3_alpha_encoder_model import cdr3VAEa
from pMTnet_Omni.encoders.cdr3_beta_encoder_model import cdr3VAEb
from pMTnet_Omni.encoders.utilities import peptide_map


class tcr_encoder_class:
    def __init__(self,\
                 model_device: str,\
                 vGdVAEacheckpoint_path: Optional[str]=None,\
                 vGdVAEbcheckpoint_path: Optional[str]=None,\
                 cdr3VAEacheckpoint_path: Optional[str]=None,\
                 cdr3VAEbcheckpoint_path: Optional[str]=None):
        self.model_device = model_device
        self.va_model = vGdVAEa().to(model_device)
        self.vb_model = vGdVAEb().to(model_device)
        self.cdr3a_model = cdr3VAEa().to(model_device)
        self.cdr3b_model = cdr3VAEb().to(model_device)

        # Load models 
        # va
        if vGdVAEacheckpoint_path is not None:
            vGdVAEacheckpoint = torch.load(vGdVAEacheckpoint_path, map_location=model_device)
            self.va_model.load_state_dict(vGdVAEacheckpoint['net'])
        # vb
        if vGdVAEbcheckpoint_path is not None:
            vGdVAEbcheckpoint = torch.load(vGdVAEbcheckpoint_path, map_location=model_device)
            self.vb_model.load_state_dict(vGdVAEbcheckpoint['net'])
        # cdr3a
        if cdr3VAEacheckpoint_path is not None:
            cdr3VAEacheckpoint = torch.load(cdr3VAEacheckpoint_path, map_location=model_device)
            self.cdr3a_model.load_state_dict(cdr3VAEacheckpoint['net'])
        # cdr3b
        if cdr3VAEbcheckpoint_path is not None:
            cdr3VAEbcheckpoint = torch.load(cdr3VAEbcheckpoint_path, map_location=model_device)
            self.cdr3b_model.load_state_dict(cdr3VAEbcheckpoint['net'])

        self.va_model.eval()
        self.vb_model.eval()
        self.cdr3a_model.eval()
        self.cdr3b_model.eval()


    def _encode(self,\
                model,\
                df,\
                aa_dict_atchley,\
                column,\
                padding):
        seq = torch.Tensor(peptide_map(df, aa_dict_atchley, column, padding)).to(self.model_device)
        encoded, _, _, _ = model(seq)
        encoded[torch.isnan(encoded).all(dim=1)] = 0
        return encoded

    def encode(self, df, aa_dict_atchley):
        with torch.no_grad():
            va_embedding = F.normalize(self._encode(model=self.va_model,\
                                                    df=df,\
                                                    aa_dict_atchley=aa_dict_atchley,\
                                                    column="vaseq", padding=100))
            
            vb_embedding = F.normalize(self._encode(model=self.vb_model,\
                                                    df=df,\
                                                    aa_dict_atchley=aa_dict_atchley,\
                                                    column="vbseq", padding=100))
            
            cdr3a_embedding = F.normalize(self._encode(model=self.cdr3a_model,\
                                                       df=df,\
                                                       aa_dict_atchley=aa_dict_atchley,\
                                                       column="cdr3a", padding=25))
            
            cdr3b_embedding = F.normalize(self._encode(model=self.cdr3b_model,\
                                                       df=df,\
                                                       aa_dict_atchley=aa_dict_atchley,\
                                                       column="cdr3b", padding=25))
            
            tcr_embedding = torch.cat((va_embedding,vb_embedding,cdr3a_embedding,cdr3b_embedding),dim=1)
            return tcr_embedding