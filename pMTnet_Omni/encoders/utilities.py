# Data IO
import pandas as pd

# Numeric manipulation
import numpy as np

# PyTorch module
import torch

# User entertain 
from tqdm import tqdm 

def esm_mapping(mhc: str,
                mhc_dict: dict,
                use_seq: bool,
                mhcseq: str) -> np.ndarray:
    """Maps an MHC sequence to its ESM embedding

    Parameters
    -----------
    mhc: str
        An MHC sequence 
    mhc_dic: dict
        A dictionary whose keys are MHC sequences and values are the file paths 
        to the corresponding ESM embeddings 
    use_seq: bool
        Whether or not to use the MHC sequence. This will use ESM2 to produce 
        the corresponding encoding 
    mhcseq: str
        The MHC sequence 
    
    Returns
    ---------
    np.ndarray
        The ESM embedding of the given MHC sequence padded with 0s

    """
    
    if use_seq:
        mhc_encoding = mhc_dict[mhcseq]
    else:
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
    verbose: bool
        Whether or not to print out the process

    Returns
    ----------
    np.ndarray
        The ESM embeddings of the given MHC sequences padded with 0s

    """
    
    mhc_array = np.zeros((len(df), 1, 380, 1280), dtype=np.float32)
    mhc_seen = dict()
    verboseprint = print if verbose else lambda *a, **k: None
    verboseprint("Mapping "+column_name+"\n")
    
    nrows = df.shape[0]
    
    for pos, i in tqdm(enumerate(range(nrows)), disable=(not verbose)):
        mhc = df.at[i, column_name]
        try:
            mhc_array[pos, 0] = mhc_seen[mhc]
        except:
            use_seq = df.at[i, column_name+"_use_seq"]
            mhcseq = df.at[i, column_name+"seq"]
            mhc_array[pos, 0] = esm_mapping(mhc=mhc,
                                            mhc_dict=mhc_dict,
                                            use_seq=use_seq,
                                            mhcseq=mhcseq)
            mhc_seen[mhc] = mhc_array[pos, 0]
    return mhc_array


def aa_mapping(peptide_seq: str,
               aa_dict_atchley: dict,
               padding: int,
               mutation_value: np.ndarray=np.zeros(5, dtype='float32')) -> np.ndarray:
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
    mutation_mode: np.ndarray
        The embedding for unknown AAs 

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
            peptide_array.append(mutation_value)
    return np.concatenate((np.asarray(peptide_array).reshape(-1,5), 
                           np.zeros((padding - len(peptide_seq), 5), 
                                    dtype='float32')), axis=0)


def peptide_map(df: pd.DataFrame,
                column_name: str,
                aa_dict_atchley: dict,
                padding: int,
                verbose: bool=False,
                mutation_mode: str="zero") -> np.ndarray:
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
    mutation_mode: str
        How unknown AAs should be interpolated
        zero: 5 zeros 
        alanine: atchley factor for A
        average: average atchley factor 

    Returns
    -------
    np.ndarray
        An array of the corresponding Atchley Factors 
    
    """
    verboseprint = print if verbose else lambda *a, **k: None
    peptide_array = np.zeros((len(df), 1, padding, 5), dtype=np.float32)
    verboseprint("Mapping "+column_name+"\n")
    
    atchley_ave = np.mean(np.array(list(aa_dict_atchley.values())), axis=0)
    
    mutation_dict = {"zero": np.zeros(5, dtype='float32'),
                     "alanine": aa_dict_atchley['A'].astype('float32'),
                     "average": atchley_ave.astype('float32')}
    
    for pos, seq in tqdm(enumerate(df[column_name]), disable=(not verbose)):
        peptide_array[pos, 0] = aa_mapping(peptide_seq=seq,
                                           aa_dict_atchley=aa_dict_atchley,
                                           padding=padding,
                                           mutation_value=mutation_dict[mutation_mode])
    return peptide_array
