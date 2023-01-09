# Data IO
import pandas as pd

# Numeric manipulation
import numpy as np

# PyTorch module
import torch

# User entertain 
from tqdm import tqdm 

def esm_mapping(mhc: str,
                mhc_dict: dict) -> np.ndarray:
    """Maps an MHC sequence to its ESM embedding

    Parameters
    -----------
    mhc: str
        An MHC sequence 
    mhc_dic: dict
        A dictionary whose keys are MHC sequences and values are the file paths 
        to the corresponding ESM embeddings 
    
    Returns
    ---------
    np.ndarray
        The ESM embedding of the given MHC sequence padded with 0s

    """
    mhc_encoding = torch.load(mhc_dict[mhc]).numpy()
    num_padding = 380-mhc_encoding.shape[0]
    return np.concatenate((mhc_encoding, np.zeros((num_padding, 1280), dtype='float32')), axis=0)


def mhc_map(df: pd.DataFrame,
            column_name: str,
            mhc_dict: dict,
            verbose: bool=False) -> np.ndarray:
    """Maps a column of MHC sequences to their ESM embeddings

    Parameters
    ----------
    df: pd.DataFrame 
        The dataframe containing pairing data 
    column_name: str
        The name of the column corresponding to the MHC sequences 
    mhc_dict: dict 
        A dictionary whose keys are MHC sequences and values are the file paths 
        to the corresponding ESM embeddings 

    Returns
    ----------
    np.ndarray
        The ESM embeddings of the given MHC sequences padded with 0s

    """
    mhc_array = np.zeros((len(df), 1, 380, 1280), dtype=np.float32)
    mhc_seen = dict()
    verboseprint = print if verbose else lambda *a, **k: None
    verboseprint("Mapping "+column_name+"\n")
    for pos, mhc in tqdm(enumerate(df[column_name]), disable=(not verbose)):
        try:
            mhc_array[pos, 0] = mhc_seen[mhc]
        except:
            mhc_array[pos, 0] = esm_mapping(mhc=mhc,
                                            mhc_dict=mhc_dict)
            mhc_seen[mhc] = mhc_array[pos, 0]
    return mhc_array


def aa_mapping(peptide_seq: str,
               aa_dict_atchley: dict,
               padding: int) -> np.ndarray:
    """Converts a sequence of amino acids to their Atchley Factors

    Parameters
    ----------
    peptide_seq: str
        A sequence of amino acids 
    aa_dict_atchley: dict
        A dictionary whose keys are amino acids and values are the
        corresponding Atchley Factors 
    padding: int 
        The maximun length of the amino acid squence 
    
    Returns
    ---------
    np.array
        An array of the corresponding Atchley Factors 

    """
    peptide_array = []
    if len(peptide_seq) > padding:
        peptide_seq = peptide_seq[0:padding]
    for aa_single in peptide_seq:
        try:
            # If the aa is a legit aa
            peptide_array.append(aa_dict_atchley[aa_single])
        except:
            # Otherwise, we replace with 0s
            peptide_array.append(np.zeros(5, dtype='float32'))
    return np.concatenate((np.asarray(peptide_array), np.zeros((padding - len(peptide_seq), 5), dtype='float32')), axis=0)


def peptide_map(df: pd.DataFrame,
                column_name: str,
                aa_dict_atchley: dict,
                padding: int,
                verbose: bool=False) -> np.ndarray:
    """Converts a column of amino acid sequences to their Atchley Factors 

    Parameters
    ---------
    df: pd.DataFrame
        A pandas dataframe containing pairing data 
    column_name: str
        The name of the column corresponding to the amino acid sequences 
    aa_dict_atchley: dict
        A dictionary whose keys are amino acids and values are the
        corresponding Atchley Factors 
    padding: int 
        The maximun length of the amino acid squence 

    Returns
    -------
    np.ndarray
        An array of the corresponding Atchley Factors 
    
    """
    verboseprint = print if verbose else lambda *a, **k: None
    peptide_array = np.zeros((len(df), 1, padding, 5), dtype=np.float32)
    verboseprint("Mapping "+column_name+"\n")
    for pos, seq in tqdm(enumerate(df[column_name]), disable=(not verbose)):
        peptide_array[pos, 0] = aa_mapping(peptide_seq=seq,
                                           aa_dict_atchley=aa_dict_atchley,
                                           padding=padding)
    return peptide_array
