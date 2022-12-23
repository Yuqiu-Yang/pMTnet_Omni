# Data IO
import os 
import pandas as pd 

# Numeric manipulation 
import numpy as np 

# Strig operation 
import re

# ROC curve
from sklearn import metrics 
from matplotlib import pyplot as plt

# Typing 
from typing import Union, Optional


def check_column_names(df, background_tcrs_dir="./data/background_tcrs/"):
    common_column_names = ["va", "cdr3a", "vaseq", "vb", "cdr3b", "vbseq",\
                            "peptide", "mhca", "mhcb", "tcr_species", "pmhc_species"]
    df.columns = df.columns.str.strip().str.lower().str.replace("_","").str.replace('\W','',regex=True)
    df_cols = df.columns.tolist()
    for i in range(len(df_cols)):
        df_cols[i] = re.sub(r'(?<=tcr).*(?=species)', "_", df_cols[i])
        df_cols[i] = re.sub(r'(?<=pmhc).*(?=species)', "_", df_cols[i])
    df.columns = df_cols

    user_df_set = set(df_cols)
    required_set = set(common_column_names)
    diff_set = required_set - user_df_set

    error_messages = "".join(["Column " + name + " can not be found.\n"\
                     for name in ["cdr3a", "cdr3b", "peptide", "mhca", "mhcb",\
                      "tcr_species", "pmhc_species"] if name in diff_set])
    
    if error_messages != "":
        raise Exception(error_messages) 

    if ("va" in diff_set) and ("vaseq" in diff_set):
        raise Exception("At least one of va and vaseq need to exist")

    if ("vb" in diff_set) and ("vbseq" in diff_set):
        raise Exception("At least one of vb and vbseq need to exist")

    if not set(df['tcr_species'].values).issubset(set(['human', 'mouse'])):
        raise Exception("tcr_species have to be human or mouse")

    if all(name in df_cols for name in common_column_names):
        return df[common_column_names]
    else:
        human_alpha_tcrs = pd.read_csv(os.path.join(background_tcrs_dir, "human_alpha.txt"), sep="\t", header=0)[["va", "vaseq"]].drop_duplicates()
        human_beta_tcrs = pd.read_csv(os.path.join(background_tcrs_dir, "human_beta.txt"), sep="\t", header=0)[["vb", "vbseq"]].drop_duplicates()
        mouse_alpha_tcrs = pd.read_csv(os.path.join(background_tcrs_dir, "mouse_alpha.txt"), sep="\t", header=0)[["va", "vaseq"]].drop_duplicates()
        mouse_beta_tcrs = pd.read_csv(os.path.join(background_tcrs_dir, "mouse_beta.txt"), sep="\t", header=0)[["vb", "vbseq"]].drop_duplicates()
        
        human_df = df[df['tcr_species'] == "human"]
        mouse_df = df[df['tcr_species'] == "mouse"]

        if ("va" in diff_set) and ("vaseq" not in diff_set):
            # if va is missing but we have its sequence 
            human_df = human_df.merge(human_alpha_tcrs, on="vaseq", how='left')
            mouse_df = mouse_df.merge(mouse_alpha_tcrs, on="vaseq", how='left')
        if ("va" not in diff_set) and ("vaseq" in diff_set):
            # if we have the name but we are missing the sequence 
            human_df = human_df.merge(human_alpha_tcrs, on="va", how='left')
            mouse_df = mouse_df.merge(mouse_alpha_tcrs, on="va", how='left')
        if ("vb" in diff_set) and ("vbseq" not in diff_set):
            # if vb is missing but we have its sequence 
            human_df = human_df.merge(human_beta_tcrs, on="vbseq", how='left')
            mouse_df = mouse_df.merge(mouse_beta_tcrs, on="vbseq", how='left')
        if ("vb" not in diff_set) and ("vbseq" in diff_set):
            # if we have the name but we are missing the sequence 
            human_df = human_df.merge(human_beta_tcrs, on="vb", how='left')
            mouse_df = mouse_df.merge(mouse_beta_tcrs, on="vb", how='left')

        df = pd.concat([human_df, mouse_df], axis=0, ignore_index=True)
        return df[common_column_names]


def check_data_sanity(df: pd.DataFrame,\
                      mhc_path: str) -> pd.DataFrame:
    """Check the data sanity
    This function will check if the data format conforms to what our model expects 

    """
    # We drop NAs first 
    df=df.dropna()
    print("Number of rows in this dataset after dropping NA: " + str(df.shape[0]))
    
    #2. antigen peptide longer than 30 will be dropped
    num_row = df.shape[0]
    df_antigen_dropped = df[df.peptide.str.len()>30]
    df = df[df.peptide.str.len()<=30]
    if((num_row-df.shape[0])>0):
        print(str(num_row-df.shape[0])+' antigens longer than ' + str(30) + 'aa are dropped:')
        print(df_antigen_dropped)
    
    #3. input MHC that is not in the ESM dictionary will be dropped
    num_row = df.shape[0]
    with open(mhc_path, 'r') as f:
        mhc_dic_keys = f.read().splitlines()
    mhc_dic_keys = set(mhc_dic_keys)
    df_mhc_alpha_dropped = df[~df["mhca"].isin(mhc_dic_keys)]
    df_mhc_beta_dropped = df[~df["mhcb"].isin(mhc_dic_keys)]
    df = df[df["mhca"].isin(mhc_dic_keys)]
    df = df[df["mhcb"].isin(mhc_dic_keys)]
    if((num_row-df.shape[0])>0):
        print(str(num_row-df.shape[0])+' MHCs without ESM embedding are dropped:')
        print(pd.unique(df_mhc_alpha_dropped["mhca"]))
        print(pd.unique(df_mhc_beta_dropped["mhcb"]))
    df = df.reset_index(drop=True)
    print("Number of rows in processed dataset: " + str(df.shape[0]))

    # Check Amino Acids are valid 
    df['vaseq'] = check_amino_acids(df["vaseq"].to_frame())
    df['vbseq'] = check_amino_acids(df["vbseq"].to_frame())
    df['cdr3a'] = check_amino_acids(df["cdr3a"].to_frame())
    df['cdr3b'] = check_amino_acids(df["cdr3b"].to_frame())
    df['peptide'] = check_amino_acids(df["peptide"].to_frame())

    return df

def check_amino_acids(df_column):
    aa_set = set([*'ARDNCEQGHILKMFPSTWYV'])
    for r in range(df_column.shape[0]):
        wrong_aa = [aa for aa in df_column.iloc[r,0] if aa not in aa_set]
        for aa in wrong_aa:
            df_column.iloc[r,0] = df_column.iloc[r,0].replace(aa, "_")
    return df_column


def read_file(file_path: str,\
             sep: str,\
             header: int,\
             background_tcr_dir: str,\
             mhc_path: str):
    # Read in the dataframe 
    df = pd.read_csv(file_path, header=header, sep=sep)
    print("Number of rows in raw dataset: " + str(df.shape[0]))
    # Check column names 
    df = check_column_names(df, background_tcr_dir)
    df = check_data_sanity(df, mhc_path)
    return df 


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



