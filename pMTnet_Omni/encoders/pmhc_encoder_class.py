# Data IO
import pandas as pd

# PyTorch modules
import torch
import torch.nn.functional as F

# User entertainment
from tqdm import tqdm

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
        
        print("Attempt to load the pMHC encoder\n")
        if pMHCcheckpoint_path is not None:
            pMHCcheckpoint = torch.load(
                pMHCcheckpoint_path, map_location=model_device)
            self.pmhc_model.load_state_dict(pMHCcheckpoint['net'])
            print("Success\n")
        else:
            print("Check point is not provided. Use random weights\n")

        self.pmhc_model.eval()

    def encode(self,
               df: pd.DataFrame,
               aa_dict_atchley: dict,
               mhc_dict: dict,
               verbose: bool=False) -> torch.tensor:
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
                                           padding=30,
                                           verbose=verbose)).to(self.model_device)
            x_a = torch.Tensor(mhc_map(df=df,
                                       column_name="mhca",
                                       mhc_dict=mhc_dict,
                                       verbose=verbose)).to(self.model_device)
            x_b = torch.Tensor(mhc_map(df=df,
                                       column_name="mhcb",
                                       mhc_dict=mhc_dict,
                                       verbose=verbose)).to(self.model_device)
            if self.model_device == "cpu":
                print("As the device is cpu, we will encode pMHCs sequentially.\n")
                pmhc_embedding = torch.zeros(df.shape[0], 30)
                for i in tqdm(range(df.shape[0])):
                    pmhc_embedding[i,:], _ = self.pmhc_model(x_p[[i],:], 
                                                             x_a[[i],:],
                                                             x_b[[i],:])
            else:
                print("As the device is not cpu, we will encode pMHCs in bulk.\n")
                pmhc_embedding, _ = self.pmhc_model(x_p, x_a, x_b)
            return F.normalize(pmhc_embedding)
