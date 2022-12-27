# Data IO
import pandas as pd

# PyTorch modules
import torch

# Typing
from typing import Optional

# Utilities and model
from pMTnet_Omni.encoders.pmhc_encoder_model import pMHC
from pMTnet_Omni.encoders.utilities import peptide_map, mhc_map


class pmhc_encoder_class:
    def __init__(self,
                 model_device: str,
                 pMHCcheckpoint_path: Optional[str] = None) -> None:
        """The pMHC encoder

        Parameters
        ---------
        model_device: str
            cpu or gpu
        pMHCcheckpoint_path: Optional[str]
            The path to pMHC the encoder
        
        """
        self.model_device = model_device
        self.pmhc_model = pMHC().to(model_device)
        if pMHCcheckpoint_path is not None:
            pMHCcheckpoint = torch.load(
                pMHCcheckpoint_path, map_location=model_device)
            self.pmhc_model.load_state_dict(pMHCcheckpoint['net'])
        self.pmhc_model.eval()

    def encode(self,
               df: pd.DataFrame,
               aa_dict_atchley: dict,
               mhc_dict: dict) -> torch.tensor:
        """Encodes all pMHCs in a dataframe 

        Parameters
        ----------
        df: pd.DataFrame 
            A user dataframe containing pairing data 
        aa_dict_atchley: dict
            A dictionary whose keys are amino acids and values are the
            corresponding Atchley Factors 
        mhc_dict: dict 
            A dictionary whose keys are MHC sequences and values are the file paths 
            to the corresponding ESM embeddings 

        Returns
        ---------
        torch.tensor
            A tensor of the pMHCs
        
        """
        with torch.no_grad():
            x_p = torch.Tensor(peptide_map(df=df,
                                           column_name="peptide",
                                           aa_dict_atchley=aa_dict_atchley,
                                           padding=30)).to(self.model_device)
            x_a = torch.Tensor(mhc_map(df=df,
                                       column_name="mhca",
                                       mhc_dict=mhc_dict)).to(self.model_device)
            x_b = torch.Tensor(mhc_map(df=df,
                                       column_name="mhcb",
                                       mhc_dict=mhc_dict)).to(self.model_device)
            pmhc_embedding, _ = self.pmhc_model(x_p, x_a, x_b)
            return pmhc_embedding
