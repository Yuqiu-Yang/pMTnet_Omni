
import numpy as np 
import pandas as pd 
import torch

ESM_DIM = 1280
ESM_LAYER = 33


def esm_mapping(mhc: str,\
                mhc_dic: dict) -> np.ndarray:
    mhc_encoding = torch.load(mhc_dic[mhc]).numpy()
    num_padding = 380-mhc_encoding.shape[0]
    return np.concatenate((mhc_encoding, np.zeros((num_padding, 1280),dtype='float32')), axis=0)

def mhc_map(df: pd.DataFrame,\
            allele: str,\
            mhc_dic: dict) -> np.ndarray:
    mhc_array = np.zeros((len(df), 1, 380, 1280), dtype=np.float32)
    mhc_seen = dict()
    for pos, mhc in enumerate(df[allele]):
        try:
            mhc_array[pos, 0] = mhc_seen[mhc]
        except:
            mhc_array[pos, 0] = esm_mapping(mhc, mhc_dic)
            mhc_seen[mhc] = mhc_array[pos, 0]
    return mhc_array


def aa_mapping(peptideSeq, aa_dict_atchley, padding):
        peptideArray = []
        if len(peptideSeq)>padding:
            #print('Length: '+str(len(peptideSeq))+'is over bound'+ ' (' +str(padding)+ ')' +'!')
            peptideSeq=peptideSeq[0:padding]
        for aa_single in peptideSeq:
            try:
                peptideArray.append(aa_dict_atchley[aa_single])
            except:
    #            print('Inproper aa: ' + aa_single + ', in seq: ' + peptideSeq + '. 0 was applied for encoding.')
                peptideArray.append(np.zeros(5, dtype='float32'))
        return np.concatenate((np.asarray(peptideArray), np.zeros((padding - len(peptideSeq), 5), dtype='float32')), axis=0)
    

def peptide_map(df, aa_dict_atchley, column, padding):
    peptideArray = np.zeros((len(df), 1, padding, 5), dtype=np.float32)
    for pos, seq in enumerate(df[column]):
        peptideArray[pos, 0] = aa_mapping(seq, aa_dict_atchley, padding)
    return peptideArray