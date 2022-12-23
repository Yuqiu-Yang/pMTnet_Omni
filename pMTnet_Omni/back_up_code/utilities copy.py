# Data IO
import os
import pandas as pd 
import csv
import torch

# Numeric manipulation 
import numpy as np 

# Strig operation 
import re

# ROC curve
from sklearn import metrics 
from matplotlib import pyplot as plt

# Typing 
from typing import Union, Optional

def read_file(file_path: Optional[str]=None,
             sep: str=",",
             header: Optional[int]=0,
             idx: Optional[list]=None) -> pd.DataFrame:
    """This function reads in .csv or .txt files 

    The function provides a simple wrapper of pd.read_csv 
    and performs some rudementary sanity check

    Parameters
    ----------
    file_path: str
        The location of the file containing TCR-pMHC pair(s)
    sep: str
        Seperation used in the file 
    header: bool
        Whether or not the file contains a header

    Returns
    ------
    pairing_data: np.ndarray
        An numpy ndarray containing curated pairs 

    """
    if file_path is None:
        pairing_data = pd.DataFrame(None)
    else:
        if idx is not None:
            idx.append(header)
            pairing_data = pd.read_csv(file_path, sep=sep, header=header, skiprows=lambda x: x not in idx)
        else:
            pairing_data = pd.read_csv(file_path, sep=sep, header=header)

    return pairing_data


def get_auroc(true_labels: np.ndarray,\
              predicted_labels: np.ndarray):
    """Compute and plot AUROC
    """
    auc = metrics.roc_auc_score(true_labels, predicted_labels)
    return auc

def plot_roc_curve(true_labels: np.ndarray,\
                   predicted_labels: np.ndarray,
                   **kwargs):
    fpr, tpr, _ = metrics.roc_curve(true_labels, predicted_labels)
    plt.plot(fpr, tpr, **kwargs)
    plt.legend(**kwargs)
    plt.show()



def check_data_sanity(df: pd.DataFrame) -> pd.DataFrame:
    """Check the data sanity
    This function will check if the data format conforms to what our model expects 

    """
    df.columns = df.columns.str.strip().str.lower().str.replace("_","").str.replace('\W','',regex=True)
    df_cols = df.columns.tolist()
    for i in len(df_cols):
        df_cols[i] = re.sub(r'(?<=v).*(?=a)', "_", df_cols[i])
        df_cols[i] = re.sub(r'(?<=cdr3).*(?=a)', "_", df_cols[i])
        df_cols[i] = re.sub(r'(?<=a).*(?=seq)', "_", df_cols[i])
        df_cols[i] = re.sub(r'(?<=v).*(?=b)', "_", df_cols[i])
        df_cols[i] = re.sub(r'(?<=cdr3).*(?=b)', "_", df_cols[i])
        df_cols[i] = re.sub(r'(?<=b).*(?=seq)', "_", df_cols[i])
        df_cols[i] = re.sub(r'(?<=mhc).*(?=a)', "_",df_cols[i])
        df_cols[i] = re.sub(r'(?<=mhc).*(?=b)', "_", df_cols[i])
        df_cols[i] = re.sub(r'(?<=tcr).*(?=species)', "_", df_cols[i])
        df_cols[i] = re.sub(r'(?<=pmhc).*(?=species)', "_", df_cols[i])


    common_column_names = ["v_a", "cdr3_a", "v_a_seq", "v_b", "cdr3_b", "v_b_seq",\
                            "peptide", "mhc_a", "mhc_b", "tcr_species", "pmhc_species"]

    all(item in df_cols for item in common_column_names)

    df = df[common_column_names]

    pass 

#################################################################
#################################################################
#################################################################
#################################################################
#################################################################

directory = os.fsencode("/project/DPDS/Wang_lab/shared/pMTnet_v2/data/ipd_imgt/" + ESM_FILE)
def build_mhc_dictionary(directory):
    mhc_dic = {}
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        esm_file = torch.load(os.path.join("/project/DPDS/Wang_lab/shared/pMTnet_v2/data/ipd_imgt/" + ESM_FILE, filename))
        mhc_dic[filename.split(".")[0]] = esm_file["representations"][ESM_LAYER]
    return mhc_dic









def mhcMap(dataset, allele, mhc_dic):
    mhc_array = np.zeros((len(dataset), 1, 380, ESM_DIM), dtype=np.float32)
    mhc_seen = dict()
    for pos, mhc in enumerate(dataset[allele]):
        try:
            mhc_array[pos, 0] = mhc_seen[mhc]
        except:
            mhc_array[pos, 0] = esmmapping(mhc,mhc_dic)
            mhc_seen[mhc] = mhc_array[pos, 0]
    return mhc_array



def pmhcEncoder(model, source_dataset, model_device):
    x_p = torch.Tensor(peptideMap(source_dataset, aa_dict_atchley, "peptide", 30)).to(model_device)
    x_a = torch.Tensor(mhcMap(source_dataset, "mhca", mhc_dic)).to(model_device)
    x_b = torch.Tensor(mhcMap(source_dataset, "mhcb", mhc_dic)).to(model_device)
    encoded, output = model(x_p, x_a, x_b)
    return encoded


def esmmapping(mhc,mhc_dic):
    mhc_encoding = mhc_dic[mhc].numpy()
    num_padding = 380-mhc_encoding.shape[0]
    return np.concatenate((mhc_encoding, np.zeros((num_padding,ESM_DIM),dtype='float32')), axis=0)


def preprocess(filedir, mhc_dic, a_allele, b_allele): 
    #1. input file path is valid or not
    print('Processing: '+filedir)
    if not os.path.exists(filedir):
        print('Invalid file path: ' + filedir)
        return 0
    dataset = pd.read_csv(filedir, header=0, sep="\t")
    print("Number of rows in raw dataset: " + str(dataset.shape[0]))
    dataset=dataset.dropna()
    print("Number of rows in this dataset after dropping NA: " + str(dataset.shape[0]))
    #2. antigen peptide longer than 30 will be dropped
    num_row = dataset.shape[0]
    dataset_antigen_dropped = dataset[dataset.peptide.str.len()>30]
    dataset=dataset[dataset.peptide.str.len()<=30]
    if((num_row-dataset.shape[0])>0):
        print(str(num_row-dataset.shape[0])+' antigens longer than ' + str(30) + 'aa are dropped:')
        print(dataset_antigen_dropped)
    #3. input MHC that is not in the ESM dictionary will be dropped
    num_row = dataset.shape[0]
    mhc_dic_keys = set(mhc_dic.keys())
    dataset_mhc_alpha_dropped = dataset[~dataset[a_allele].isin(mhc_dic_keys)]
    dataset_mhc_beta_dropped = dataset[~dataset[b_allele].isin(mhc_dic_keys)]
    dataset = dataset[dataset[a_allele].isin(mhc_dic_keys)]
    dataset = dataset[dataset[b_allele].isin(mhc_dic_keys)]
    if((num_row-dataset.shape[0])>0):
        print(str(num_row-dataset.shape[0])+' MHCs without ESM embedding are dropped:')
        print(pd.unique(dataset_mhc_alpha_dropped[a_allele]))
        print(pd.unique(dataset_mhc_beta_dropped[b_allele]))
#    dataset = dataset.sample(frac=1)
    dataset = dataset.reset_index(drop=True)
    print("Number of rows in processed dataset: " + str(dataset.shape[0]))
    return dataset








def set_model(model_device):
    
    pMHCcheckpoint = torch.load("/work/DPDS/s213303/pmtnetv2/script/pytorch_model/pMHC_Copy42_Seed3000Channel200Batch200LR0.001Epoch20.pth",map_location=model_device)
    vGdVAEacheckpoint = torch.load("/work/DPDS/s213303/pmtnetv2/script/pytorch_model/vgene_dvae_Copy26_5neuron_Seed100Channel180Batch100LR0.0001Epoch390.pth",map_location=model_device)
    vGdVAEbcheckpoint = torch.load("/work/DPDS/s213303/pmtnetv2/script/pytorch_model/vgene_dvae_Copy29_5neuron_Seed100Channel180Batch100LR0.0001Epoch365.pth",map_location=model_device)
    cdr3VAEacheckpoint = torch.load("/work/DPDS/s213303/pmtnetv2/script/pytorch_model/cdr3_a_vae_Copy128_Seed100Bottle30Batch200Lr0.0005N_Trans6Channel150Embed3Epoch225.pth",map_location=model_device)
    cdr3VAEbcheckpoint = torch.load("/work/DPDS/s213303/pmtnetv2/script/pytorch_model/cdr3_b_vae_Copy132_Seed100Bottle30Batch200Lr0.0005N_Trans6Channel150Embed3Epoch170.pth",map_location=model_device)
    
    CLmodel_new = pMHCTCR(temperature=TEMPERATURE).to(model_device)
    
    pMHCmodel_loaded = pMHC().to(model_device)
    pMHCmodel_loaded.load_state_dict(pMHCcheckpoint['net'])

    vGdVAEamodel_loaded = vGdVAEa().to(model_device)
    vGdVAEamodel_loaded.load_state_dict(vGdVAEacheckpoint['net'])

    vGdVAEbmodel_loaded = vGdVAEb().to(model_device)
    vGdVAEbmodel_loaded.load_state_dict(vGdVAEbcheckpoint['net'])

    cdr3VAEamodel_loaded = cdr3VAEa().to(model_device)
    cdr3VAEamodel_loaded.load_state_dict(cdr3VAEacheckpoint['net'])

    cdr3VAEbmodel_loaded = cdr3VAEb().to(model_device)
    cdr3VAEbmodel_loaded.load_state_dict(cdr3VAEbcheckpoint['net'])
    
    return CLmodel_new, pMHCmodel_loaded, vGdVAEamodel_loaded, vGdVAEbmodel_loaded, cdr3VAEamodel_loaded, cdr3VAEbmodel_loaded



