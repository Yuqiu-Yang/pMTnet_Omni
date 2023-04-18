# Data IO
import pandas as pd

# PyTorch modules
import torch
import torch.nn.functional as F

# Typing
from typing import Optional, Union

# Utilities and models
try:
    from pMTnet_Omni.models.v_alpha_encoder_model import vGdVAEa
except ImportError:
    from pMTnet_Omni.models.load_dummies import vGdVAEa

try:
    from pMTnet_Omni.models.v_beta_encoder_model import vGdVAEb
except ImportError:
    from pMTnet_Omni.models.load_dummies import vGdVAEb
    
try:
    from pMTnet_Omni.models.cdr3_alpha_encoder_model import cdr3VAEa
except ImportError:
    from pMTnet_Omni.models.load_dummies import cdr3VAEa

try:
    from pMTnet_Omni.models.cdr3_beta_encoder_model import cdr3VAEb
except ImportError:
    from pMTnet_Omni.models.load_dummies import cdr3VAEb

from pMTnet_Omni.encoders.utilities import peptide_map


class tcr_encoder_class:
    def __init__(self,
                 model_device: Union[str, torch.device],
                 vGdVAEacheckpoint_path: Optional[str] = None,
                 vGdVAEbcheckpoint_path: Optional[str] = None,
                 cdr3VAEacheckpoint_path: Optional[str] = None,
                 cdr3VAEbcheckpoint_path: Optional[str] = None,
                 mutation_mode: str="zero") -> None:
        """The TCR encoder

        Parameters
        ---------
        model_device: Union[str, torch.device]
            cpu or gpu
        vGdVAEacheckpoint_path: Optional[str]
            The path to VA the encoder
        vGdVAEbcheckpoint_path: Optional[str]
            The path to VB the encoder
        cdr3VAEacheckpoint_path: Optional[str]
            The path to CDR3 Alpha encoder
        cdr3VAEbcheckpoint_path: Optional[str]
            The path to CDR3 Beta encoder
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
        # Initialize the models
        self.va_model = vGdVAEa().to(model_device)
        self.vb_model = vGdVAEb().to(model_device)
        self.cdr3a_model = cdr3VAEa().to(model_device)
        self.cdr3b_model = cdr3VAEb().to(model_device)

        # Load models if paths are provided
        # va
        print("Attempt to load the VA encoder\n")
        if vGdVAEacheckpoint_path is not None:
            vGdVAEacheckpoint = torch.load(
                vGdVAEacheckpoint_path, map_location=model_device)
            self.va_model.load_state_dict(vGdVAEacheckpoint['net'])
            print("Success\n")
        else:
            print("Check point is not provided. Use random weights\n")
        # vb
        print("Attempt to load the VB encoder\n")
        if vGdVAEbcheckpoint_path is not None:
            vGdVAEbcheckpoint = torch.load(
                vGdVAEbcheckpoint_path, map_location=model_device)
            self.vb_model.load_state_dict(vGdVAEbcheckpoint['net'])
            print("Success\n")
        else:
            print("Check point is not provided. Use random weights\n")
        # cdr3a
        print("Attempt to load the CDR3A encoder\n")
        if cdr3VAEacheckpoint_path is not None:
            cdr3VAEacheckpoint = torch.load(
                cdr3VAEacheckpoint_path, map_location=model_device)
            self.cdr3a_model.load_state_dict(cdr3VAEacheckpoint['net'])
            print("Success\n")
        else:
            print("Check point is not provided. Use random weights\n")
        # cdr3b
        print("Attempt to load the CDR3B encoder\n")
        if cdr3VAEbcheckpoint_path is not None:
            cdr3VAEbcheckpoint = torch.load(
                cdr3VAEbcheckpoint_path, map_location=model_device)
            self.cdr3b_model.load_state_dict(cdr3VAEbcheckpoint['net'])
            print("Success\n")
        else:
            print("Check point is not provided. Use random weights\n")

        self.va_model.eval()
        self.vb_model.eval()
        self.cdr3a_model.eval()
        self.cdr3b_model.eval()

    def _encode(self,
                model,
                df: pd.DataFrame,
                column_name: str,
                aa_dict_atchley: dict,
                padding: int,
                verbose: bool=False) -> torch.tensor:
        """Encode an array of Atchley Factors 

        Parameters
        ---------
        model
            The PyTorch model used to encode the sequences 
        df: pd.DataFrame 
            A user dataframe containing pairing data 
        column_name: str
            The name of the column corresponding to the amino acid sequences 
        aa_dict_atchley: dict
            A dictionary whose keys are amino acids and values are the
            corresponding Atchley Factors 
        padding: int 
            The maximun length of the amino acid squence
        
        Returns
        --------
        torch.tensor
            A tensor of the encoded amino acids 
        """
        seq = torch.Tensor(peptide_map(df=df,
                                       column_name=column_name,
                                       aa_dict_atchley=aa_dict_atchley,
                                       padding=padding,
                                       verbose=verbose,
                                       mutation_mode=self.mutation_mode)).to(self.model_device)
        _, _, encoded, _ = model(seq)
        encoded[torch.isnan(encoded).all(dim=1)] = 0
        return encoded

    def encode(self,
               df: pd.DataFrame,
               aa_dict_atchley: dict,
               is_embedding: bool=False,
               verbose: bool=False) -> torch.tensor:
        """Encodes all TCRs in a dataframe 

        Parameters
        ----------
        df: pd.DataFrame 
            A user dataframe containing pairing data 
        aa_dict_atchley: dict
            A dictionary whose keys are amino acids and values are the
            corresponding Atchley Factors 
        Returns
        ---------
        torch.tensor
            A tensor of the encoded amino acids 
        
        """
        if is_embedding:
            # VA
            ind = ["va" in n for n in df.columns]
            va_embedding = torch.tensor(df.loc[:, ind].values, dtype=torch.float32, device=self.model_device)
            # VB
            ind = ["vb" in n for n in df.columns]
            vb_embedding = torch.tensor(df.loc[:, ind].values, dtype=torch.float32, device=self.model_device)
            # CDR3A
            ind = ["cdr3a" in n for n in df.columns]
            cdr3a_embedding = torch.tensor(df.loc[:, ind].values, dtype=torch.float32, device=self.model_device)
            # CDR3B
            ind = ["cdr3b" in n for n in df.columns]
            cdr3b_embedding = torch.tensor(df.loc[:, ind].values, dtype=torch.float32, device=self.model_device)
        else:
            with torch.no_grad():
                va_embedding = F.normalize(self._encode(model=self.va_model,
                                                        df=df,
                                                        column_name="vaseq",
                                                        aa_dict_atchley=aa_dict_atchley,
                                                        padding=100,
                                                        verbose=verbose))

                vb_embedding = F.normalize(self._encode(model=self.vb_model,
                                                        df=df,
                                                        column_name="vbseq",
                                                        aa_dict_atchley=aa_dict_atchley,
                                                        padding=100,
                                                        verbose=verbose))

                cdr3a_embedding = F.normalize(self._encode(model=self.cdr3a_model,
                                                        df=df,
                                                        column_name="cdr3a",
                                                        aa_dict_atchley=aa_dict_atchley,
                                                        padding=25,
                                                        verbose=verbose))

                cdr3b_embedding = F.normalize(self._encode(model=self.cdr3b_model,
                                                        df=df,
                                                        column_name="cdr3b",
                                                        aa_dict_atchley=aa_dict_atchley,
                                                        padding=25,
                                                        verbose=verbose))

        tcr_embedding = torch.cat(
            (va_embedding, cdr3a_embedding, vb_embedding, cdr3b_embedding), dim=1)
        return tcr_embedding
