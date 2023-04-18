# Data IO
import pandas as pd

# PyTorch modules
import torch
import torch.nn.functional as F

# User entertainment
from tqdm import tqdm

# Typing
from typing import Optional, Union

# Utilities and model
try:
    from pMTnet_Omni.models.pmhc_encoder_model import pMHC
except ImportError:
    from pMTnet_Omni.models.load_dummies import pMHC

from pMTnet_Omni.encoders.utilities import peptide_map, mhc_map


class pmhc_encoder_class:
    def __init__(self,
                 model_device: Union[str, torch.device],
                 pMHCcheckpoint_path: Optional[str] = None,
                 mutation_mode: str="zero") -> None:
        """The pMHC encoder

        Parameters
        ---------
        model_device: torch.device
            cpu or gpu
        pMHCcheckpoint_path: Optional[str]
            The path to pMHC the encoder
        mutation_mode: str
            How unknown AAs should be interpolated
            zero: 5 zeros 
            alanine: atchley factor for A
            average: average atchley factor 

        """
        if isinstance(model_device, str):
            model_device = torch.device(model_device)
        self.model_device = model_device
        self.mutation_mode = mutation_mode
        
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
        
        # Seen peptide mhcs dictionaries 
        # Key would be the name(seq), value would be the embedding
        ##  This would be only used when device is cpu 
        self.seen_pmhc = {}
        

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
                                           verbose=verbose,
                                           mutation_mode=self.mutation_mode)).to(self.model_device)
            x_a = torch.Tensor(mhc_map(df=df,
                                       column_name="mhca",
                                       mhc_dict=mhc_dict,
                                       verbose=verbose)).to(self.model_device)
            x_b = torch.Tensor(mhc_map(df=df,
                                       column_name="mhcb",
                                       mhc_dict=mhc_dict,
                                       verbose=verbose)).to(self.model_device)
            if self.model_device.type == "cpu":
                print("As the device is cpu, we will encode pMHCs sequentially.\n")
                pmhc_embedding = torch.zeros(df.shape[0], 30)
                for i in tqdm(range(df.shape[0])):
                    peptide = df.at[i, "peptide"]
                    if df.at[i, "mhca_use_seq"]:
                        mhca = df.at[i, "mhcaseq"]
                    else:
                        mhca = df.at[i, "mhca"]
                    if df.at[i, "mhcb_use_seq"]:
                        mhcb = df.at[i, "mhcbseq"]
                    else:
                        mhcb = df.at[i, "mhcb"]
                    pmhc = peptide + "_" + mhca + "_" + mhcb
                    if pmhc not in self.seen_pmhc:
                        self.seen_pmhc[pmhc], _ = self.pmhc_model(x_p[[i],:], 
                                                                x_a[[i],:],
                                                                x_b[[i],:])
                    pmhc_embedding[i,:] = self.seen_pmhc[pmhc]
            else:
                print("As the device is not cpu, we will encode pMHCs in bulk.\n")
                pmhc_embedding, _ = self.pmhc_model(x_p, x_a, x_b)
            return F.normalize(pmhc_embedding)
