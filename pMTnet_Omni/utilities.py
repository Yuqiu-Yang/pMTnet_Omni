# Data IO
import os
import warnings
import pandas as pd

# Numeric manipulation
import numpy as np
import math
import random

# PyTorch
import torch

# String operation
import re

# ROC curve
# from sklearn import metrics
import matplotlib as mpl
from matplotlib import pyplot as plt

# Progress bar
from tqdm import tqdm

# Typing
from typing import Tuple, Union, Optional


def check_column_names(df: pd.DataFrame,
                       background_tcrs_dir: str = "./data/background_tcrs/") -> pd.DataFrame:
    """Check if the columns are correct 

    The main purpose of this function is to make sure that the dataframe 
    provided by the users contains necessary columns so that 
    it can be used by subsequent functions 

    This function checks and corrects the column names provided in the user dataframe.

    What's more, it can also automatically create some missing columns.

    The column name to be used in all the following functions are:

    "va": name of alpha V gene 

    "cdr3a": amino acid sequence of CDR3 region on the alpha chain

    "vaseq": amino acid sequence of the alpha V gene 

    "vb": name of beta V gene

    "cdr3b": amino acid sequence of CDR3 region on the beta chain

    "vbseq": amino acid sequence of the beta V gene 

    "peptide": amino acid sequence of the peptide presented by mhc 

    "mhca": name of mhc alpha chain 

    "mhcb": name of mhc beta chain 

    "mhcaseq": sequence of mhca

    "mhcbseq" : sequence of mhcb

    "tcr_species": species of TCR (human or mouse)

    "pmhc_species": species of peptide-mhc

    This function will NOT create mhca, mhcb, mhcaseq, mhcbseq.
    It will keep the original column mhc, mhcseq.

    Parameters
    ----------
    df: pd.DataFrame
        A pandas dataframe containing pairing data 
    background_tcrs_dir: str
        The directory with background TCR datasets 

    Returns
    ---------
    pd.DataFrame:
        A pandas dataframe with corrected column names and possibly inferred missing information 

    """
    # Define the column names to be used
    common_column_names = ["va", "cdr3a", "vaseq", "vb", "cdr3b", "vbseq",
                           "peptide", "mhc", "mhcseq",
                           "tcr_species", "pmhc_species"]
    # Get rid of spaces, convert names to lowercase, remove special characters
    print("Checking column names...\n")
    df.columns = df.columns.str.replace(' ', '').str.lower().str.replace(
        "_", "").str.replace(r'\W', '', regex=True)
    df_cols = df.columns.tolist()
    for i in range(len(df_cols)):
        # We get rid of white spaces from each column
        df[df_cols[i]] = df[df_cols[i]].str.replace(' ', '')
        df_cols[i] = re.sub(r'(?<=tcr).*(?=species)', "_", df_cols[i])
        df_cols[i] = re.sub(r'(?<=pmhc).*(?=species)', "_", df_cols[i])
    df.columns = df_cols
    # As we already have a large number of mhcs
    # we do not expect users to come up with new ones
    if 'mhcseq' not in df_cols:
        df['mhcseq'] = ''
    ##################################################
    # We first make sure that all required columns are present
    # in the dataframe
    ##################################################
    # Get names not presented in the user dataframe
    user_df_set = set(df_cols)
    required_set = set(common_column_names)
    diff_set = required_set - user_df_set

    error_messages = "".join(["Column " + name + " can not be found.\n"
                              for name in ["cdr3a", "cdr3b", "peptide", "mhc",
                                           "tcr_species", "pmhc_species"] if name in diff_set])

    if error_messages != "":
        raise Exception(error_messages)

    if ("va" in diff_set) and ("vaseq" in diff_set):
        raise Exception("At least one of va and vaseq need to exist")

    if ("vb" in diff_set) and ("vbseq" in diff_set):
        raise Exception("At least one of vb and vbseq need to exist")

    ##################################################
    # We then check tcr_species and pmhc_species
    # as later steps depend on these two columns
    ##################################################
    df["tcr_species"] = df["tcr_species"].str.lower().str.replace(' ', '').str.replace(
        "_", "").str.replace(r'\W', '', regex=True)
    df["pmhc_species"] = df["pmhc_species"].str.lower().str.replace(' ', '').str.replace(
        "_", "").str.replace(r'\W', '', regex=True)

    if not set(df['tcr_species'].values).issubset(set(['human', 'mouse'])):
        raise Exception("tcr_species have to be human or mouse")

    if all(name in df_cols for name in common_column_names):
        print('Column names look fine\n')
        return df[common_column_names]
    else:
        print('Seems like some information is missing.\nNo worries, we got you covered\n')
        human_alpha_tcrs = pd.read_csv(os.path.join(
            background_tcrs_dir, "human_alpha.txt"), sep="\t", header=0)[["va", "vaseq"]].drop_duplicates()
        human_beta_tcrs = pd.read_csv(os.path.join(
            background_tcrs_dir, "human_beta.txt"), sep="\t", header=0)[["vb", "vbseq"]].drop_duplicates()
        mouse_alpha_tcrs = pd.read_csv(os.path.join(
            background_tcrs_dir, "mouse_alpha.txt"), sep="\t", header=0)[["va", "vaseq"]].drop_duplicates()
        mouse_beta_tcrs = pd.read_csv(os.path.join(
            background_tcrs_dir, "mouse_beta.txt"), sep="\t", header=0)[["vb", "vbseq"]].drop_duplicates()

        human_df = df[df['tcr_species'] == "human"]
        mouse_df = df[df['tcr_species'] == "mouse"]

        if ("va" in diff_set) and ("vaseq" not in diff_set):
            # if va is missing but we have its sequence
            human_df = human_df.merge(human_alpha_tcrs, on="vaseq", how='left')
            mouse_df = mouse_df.merge(mouse_alpha_tcrs, on="vaseq", how='left')
        if ("va" not in diff_set) and ("vaseq" in diff_set):
            # if we have the name but we are missing the sequence
            # We first curate the data a little bit
            human_df['va'] = human_df['va'].str.replace(' ','').str.replace('.','-').str.replace(':', '*')
            mouse_df['va'] = mouse_df['va'].str.replace(' ','').str.replace('.','-').str.replace(':', '*')
            # Then we left merge 
            human_df = human_df.merge(human_alpha_tcrs, on="va", how='left')
            mouse_df = mouse_df.merge(mouse_alpha_tcrs, on="va", how='left')
        if ("vb" in diff_set) and ("vbseq" not in diff_set):
            # if vb is missing but we have its sequence
            human_df = human_df.merge(human_beta_tcrs, on="vbseq", how='left')
            mouse_df = mouse_df.merge(mouse_beta_tcrs, on="vbseq", how='left')
        if ("vb" not in diff_set) and ("vbseq" in diff_set):
            # if we have the name but we are missing the sequence
            human_df['vb'] = human_df['vb'].str.replace(' ','').str.replace('.','-').str.replace(':', '*')
            mouse_df['vb'] = mouse_df['vb'].str.replace(' ','').str.replace('.','-').str.replace(':', '*')
            # Then we left merge 
            human_df = human_df.merge(human_beta_tcrs, on="vb", how='left')
            mouse_df = mouse_df.merge(mouse_beta_tcrs, on="vb", how='left')

        df = pd.concat([human_df, mouse_df], axis=0, ignore_index=True)
        return df[common_column_names].reset_index(drop=True)


def get_mhc_class(df: pd.DataFrame) -> pd.DataFrame:
    """Infer the MHC classes and create columns mhca and mhcb

    The input df should be the output of the check_column_names function.

    Parameters
    ---------
    df: pd.DataFrame
        A pandas dataframe containing mhc, mhcseq, and pmhc_species

    Returns
    ---------
    pd.DataFrame
        A column of a pandas dataframe with the inferred MHC classes, 
        MHCs on the alpha chain and the beta chain 
    """
    df = df.reset_index(drop=True)
    # Human class I starts with A, B, or C
    human_class_i = ("A", "B", "C")
    # Class II starts with DP, DQ, or DR
    human_class_ii = ("DP", "DQ", "DR")
    # Mouse class I starts with D, K, L, Q
    mouse_class_i = tuple(["H-2-" + i for i in ["D", "K", "L", "Q"]])
    # Class II starts with IA, IE
    mouse_class_ii = tuple(["H-2-" + i for i in ["IA", "IE"]])

    mhc_class = []
    mhca = []
    mhcb = []
    mhcaseq = []
    mhcbseq = []

    for i in range(df.shape[0]):
        mhc = df.at[i, "mhc"]
        mhcseq = df.at[i, "mhcseq"]
        # We clean up the input a little bit
        # We remove HLA and H2 which will be added backed if necessary
        # so as to make sure the format is coherent
        mhc = mhc.replace(' ', '').replace(
            '-', '').replace('HLA', '').replace('H2', '')
        mhcseq = mhcseq.replace(' ', '')
        pmhc_species = df.at[i, "pmhc_species"]
        if pmhc_species == "human":
            if mhc.startswith(human_class_i):
                # For human class I, mhca will be the
                # input mhc. mhcb will be human_microglobulin
                mhc_class.append('human class i')
                mhca.append(mhc)
                mhcb.append("human_microglobulin")
                mhcaseq.append(mhcseq)
                mhcbseq.append('')
            elif mhc.startswith(human_class_ii):
                # For human class II, usually alpha beta will both
                # be provided
                # However, for DR, its possible that only beta is provided
                mhc_class.append('human class ii')
                mhc_list = mhc.split('/')
                mhcseq_list = mhcseq.split('/')
                if len(mhcseq_list) == 1:
                    mhcseq_list.append('')
                if (len(mhc_list) == 1) and (mhc.startswith("DR")):
                    mhca.append("DRA*01:01")
                    mhcb.append(mhc)
                    mhcaseq.append('')
                    mhcbseq.append(mhcseq)
                else:
                    mhca.append(mhc_list[0])
                    mhcb.append(mhc_list[1])
                    mhcaseq.append(mhcseq_list[0])
                    mhcbseq.append(mhcseq_list[1])
            else:
                raise ValueError("Class of " + mhc + " can not be determined")
        elif pmhc_species == "mouse":
            mhc = 'H-2-'+mhc
            if mhc.startswith(mouse_class_i):
                # For mouse Class I, it's on the alpha
                # beta will be mouse_microglobulin
                mhc_class.append('mouse class i')
                mhca.append(mhc)
                mhcb.append('mouse_microglobulin')
                mhcaseq.append(mhcseq)
                mhcbseq.append('')
            elif mhc.startswith(mouse_class_ii):
                # For mouse Class II, alpha and beta
                # will share the same mhc
                mhcseq_list = mhcseq.split('/')
                if len(mhcseq_list) == 1:
                    mhcseq_list.append('')
                mhc_class.append('mouse class ii')
                mhca.append(mhc+"_alpha")
                mhcb.append(mhc+"_beta")
                mhcaseq.append(mhcseq_list[0])
                mhcbseq.append(mhcseq_list[1])
            else:
                raise ValueError("Class of " + mhc + " can not be determined")
        else:
            raise ValueError(pmhc_species + " is not human or mouse")

    return pd.DataFrame(data={"mhca": mhca,
                              "mhcb": mhcb,
                              "mhcaseq": mhcaseq,
                              "mhcbseq": mhcbseq,
                              "mhc_class": mhc_class})


def check_data_sanity(df: pd.DataFrame,
                      mhc_path: str = "./data/data_for_encoders/valid_mhc.txt") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Check the data sanity
    This function will check if the data format conforms to what our model expects

    Parameters
    ---------
    df: pd.DataFrame
        A pandas dataframe containing pairing data
    mhc_path: str
        The file path to valid mhcs 

    Returns
    ---------
    Tuple[df.DataFrame, df.DataFrame, df.DataFrame, df.DataFrame]
        Four pandas dataframe containing curated pairing data,
        pairs with peptides longer than 30,
        problematic mhca, and problematic mhcb
    """
    print("Check data format...\n")
    # We drop NAs first
    df = df.dropna()
    print("Number of rows in this dataset after dropping NA: " +
          str(df.shape[0]))

    # 2. antigen peptide longer than 30 will be dropped
    num_row = df.shape[0]
    df_antigen_dropped = df[df.peptide.str.len() > 30]
    df = df[df.peptide.str.len() <= 30]
    if((num_row-df.shape[0]) > 0):
        print(
            str(num_row-df.shape[0])+' antigens longer than ' + str(30) + 'aa are dropped:')
        print(df_antigen_dropped)

    # 3. input MHC that is not in the ESM dictionary and
    # whose corresponding sequence are not provided will be dropped
    num_row = df.shape[0]
    with open(mhc_path, 'r') as f:
        mhc_dic_keys = f.read().splitlines()
    mhc_dic_keys = set(mhc_dic_keys)
    df_mhc_alpha_dropped = df[(~df["mhca"].isin(mhc_dic_keys)) &
                              (df['mhcaseq'] == '')]
    df_mhc_beta_dropped = df[(~df["mhcb"].isin(mhc_dic_keys)) &
                             (df['mhcbseq'] == '')]
    df = df[(df["mhca"].isin(mhc_dic_keys)) | (df['mhcaseq'] != '')]
    df = df[(df["mhcb"].isin(mhc_dic_keys)) | (df['mhcbseq'] != '')]
    df['mhca_use_seq'] = ~df['mhca'].isin(mhc_dic_keys)
    df['mhcb_use_seq'] = ~df['mhcb'].isin(mhc_dic_keys)

    if((num_row-df.shape[0]) > 0):
        print(str(num_row-df.shape[0]) +
              ' MHCs without ESM embedding and not corresponding sequences are dropped:')
        print(pd.unique(df_mhc_alpha_dropped["mhca"]))
        print(pd.unique(df_mhc_beta_dropped["mhcb"]))
    df = df.reset_index(drop=True)
    print("Number of rows in processed dataset: " + str(df.shape[0]))

    # Check Amino Acids are valid
    print("Checking if provided amino acids are valid\n")
    df['vaseq'] = check_amino_acids(df["vaseq"].to_frame())
    df['vbseq'] = check_amino_acids(df["vbseq"].to_frame())
    df['cdr3a'] = check_amino_acids(df["cdr3a"].to_frame())
    df['cdr3b'] = check_amino_acids(df["cdr3b"].to_frame())
    df['peptide'] = check_amino_acids(df["peptide"].to_frame())

    return df, df_antigen_dropped, df_mhc_alpha_dropped, df_mhc_beta_dropped


def check_amino_acids(df_column: pd.DataFrame) -> pd.DataFrame:
    """Check amino acids are valid 
    This function checks if the amino acids in one column of a dataframe are valid amino acids

    Parameters
    ---------
    df_column: pd.DataFrame
        One column of a dataframe 

    Returns
    --------
    pd.DataFrame
        Currated column with invalid aa replaced by "_"

    """
    aa_set = set([*'ARDNCEQGHILKMFPSTWYV'])
    print('Checking amino acids...\n')
    for r in tqdm(range(df_column.shape[0])):
        wrong_aa = [aa for aa in df_column.iloc[r, 0] if aa not in aa_set]
        for aa in wrong_aa:
            df_column.iloc[r, 0] = df_column.iloc[r, 0].replace(aa, "_")
    return df_column


def read_file(file_path: str,
              sep: str = ",",
              header: int = 0,
              background_tcr_dir: str = "./data/background_tcrs/",
              mhc_path: str = "./data/data_for_encoders/valid_mhc.txt") -> pd.DataFrame:
    """Reads in user dataframe and performs some basic data curation 

    Parameters:
    -----------
    file_path: str
        Path to the dataframe
    sep: str
        Same as sep in pd.DataFrame
    header: int
        Indicator of header position 
    background_tcr_dir: str
        The directory with background TCR datasets 
    mhc_path: str
        The file path to valid mhcs 

    Returns
    -------
    pd.DataFrame
        A curated pandas dataframe 

    """
    print('Attempting to read in the dataframe...\n')
    df = pd.read_csv(file_path, header=header, sep=sep)
    print("Number of rows in raw dataset: " + str(df.shape[0]))
    # Check column names
    df = check_column_names(df, background_tcr_dir)
    mhc_class = get_mhc_class(df=df)
    df = pd.concat([df, mhc_class], axis=1, ignore_index=False)
    df = df.drop(['mhc', 'mhcseq'], axis=1)
    # Perform some basic curation
    df, _, _, _ = check_data_sanity(df, mhc_path)
    return df


def plot_validation_curve(result_dicts: dict,
                          plot_type: str,
                          primary_type: str,
                          secondary_types: Optional[Union[str, list]] = None,
                          title: Optional[str] = None,
                          figure_save_path: Optional[str] = None) -> None:
    """Plot ROC and/or PRC

    This function is used to plot ROC curves and PR (precision-recall) curves 
    based on the validate method of the pMTnet_Omni_class 
    
    The function will first seek out all data (x, y, auc) satisfying primary_type
    then plot multiple curves based on secondary_types

    Parameters
    ----------
    result_dicts : dict
        This should be the output of the validate method. 
        It should be a nested dictionary
        
        {"complete": {"overall": {"fpr": np.ndarray,
        
                                  "tpr": np.ndarray,
                                  
                                  "precision": np.ndarray,
                                  
                                  "recall": np.ndarray,
                                  
                                  "auroc": float,
                                  
                                  "auprc": float},
                                  
                      "human overall": {},
                      
                      ...}, 
                      
         "alpha missing": {},
         
         "beta missing": {}}
    plot_type : str
        Either roc or prc 
    primary_type : str
        It can either be one of the 3 missing patterns or one of the 7 mhc classes 
    secondary_types : Optional[Union[str, list]], optional
        It can either be one of the 3 missing patterns or one of the 7 mhc classes 
    title : Optional[str], optional
        The plot title. If missing, the primary type will be used 
    figure_save_path : str, optional
        The path where the figure where be saved 

    """
    # The nams used in the code are not suitable for plotting
    # Therefore, when creating the title, we will use their mappings
    type_map = {"complete": "Both Chains",
                "alpha missing": "Beta Chain",
                "beta missing": "Alpha Chain",
                "overall": "All",
                "human overall": "Human",
                "mouse overall": "Mouse",
                "human class i": "Human C1",
                "human class ii": "Human C2",
                "mouse class i": "Mouse C1",
                "mouse class ii": "Mouse C2"}

    if plot_type == "roc":
        x_key = "fpr"
        x_label = '1 - Specificity'
        y_key = "tpr"
        y_label = 'Sensitivity'
        auc_key = "auroc"
        legend_pos = "lower right"
    elif plot_type == "prc":
        x_key = "recall"
        x_label = "Recall"
        y_key = "precision"
        y_label = "Precision"
        auc_key = "auprc"
        legend_pos = "lower left"
    else:
        raise Exception("Only roc or prc are acceptable.\n")

    if isinstance(secondary_types, str):
        secondary_types = [secondary_types]
    
    if primary_type in ["complete", "alpha missing", "beta missing"]:
        if result_dicts[primary_type] is None:
            raise Exception(primary_type + " has not been computed.\n")
        if secondary_types is None:
            secondary_types = list(result_dicts[primary_type].keys())

        plot_dict = {s_type: {"x": result_dicts[primary_type][s_type][x_key],
                              "y": result_dicts[primary_type][s_type][y_key],
                              "auc": result_dicts[primary_type][s_type][auc_key]} for s_type in secondary_types}

    elif primary_type in ["overall", "human overall", "mouse overall",
                          "human class i", "human class ii",
                          "mouse class i", "mouse class ii"]:
        if secondary_types is None:
            secondary_types = ["complete", "alpha missing", "beta missing"]

        plot_dict = {s_type: {"x": result_dicts[s_type][primary_type][x_key],
                              "y": result_dicts[s_type][primary_type][y_key],
                              "auc": result_dicts[s_type][primary_type][auc_key]} for s_type in secondary_types if result_dicts[s_type] is not None}

    else:
        raise Exception("Unknown type\n")

    if title is None:
        title = type_map[primary_type]

    colors = mpl.colormaps["inferno"].resampled(len(secondary_types) + 1)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(title, fontsize=60)

    counter = 0
    for s_type in plot_dict:
        plt.plot(plot_dict[s_type]["x"],
                 plot_dict[s_type]["y"],
                 linewidth=10, color=colors.colors[counter],
                 label=type_map[s_type]+": AUC = "+str(plot_dict[s_type]["auc"]))
        counter += 1

    plt.xlabel(x_label, fontsize=55)
    plt.ylabel(y_label, fontsize=55)

    plt.xticks(fontsize=45)
    plt.yticks(fontsize=45)

    plt.legend(loc=legend_pos, fontsize=55)
    plt.tight_layout()
    
    if figure_save_path is not None:
        plt.savefig(figure_save_path)


def batchify_check_size(check_size: list = [10, 20],
                        minibatch_size: int = 10) -> Tuple[list, list]:
    """Batchify check size list

    Parameters
    ---------
    check_size: list
        List of sizes to check 
    minibatch_size: int 
        Size of the minibatches in the dataset class 

    Returns
    --------
    Tuple[list, list]
        The first value is the batch sizes for the dataloader class
        The second value is for indicating if the check size is finished

    """
    batch_size = []
    batch_finished_indicator = []
    for size in check_size:
        if size <= minibatch_size:
            # if the size is less than the size of the dataset
            # we simply load that many data
            batch_size.append(size)
            batch_finished_indicator.append(True)
        else:
            # Otherwise, we batchify
            n_batch = math.ceil(size/minibatch_size)
            i = 1
            while i < n_batch:
                batch_size.append(minibatch_size)
                # Add False to the indicator to let
                # the program know that we are not finished
                # checking for this size yet
                batch_finished_indicator.append(False)
                i += 1
            batch_size.append(minibatch_size)
            batch_finished_indicator.append(True)
    return batch_size, batch_finished_indicator


def partition_dataframe(df: pd.DataFrame,
                        partition_size: int) -> dict:
    """Partitions a given dataframe into chunks of the given size 

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to be partitioned 
    partition_size: int 
        The chunk size

    Returns
    ---------
    dict
        A dictionary of all the partitions 

    """
    partitions = {}
    n_partitions = math.ceil(df.shape[0]/partition_size)
    column_names = df.columns
    for i in range(n_partitions):
        partitions[i] = df[i *
                           partition_size: (i+1)*partition_size].reset_index(drop=True)
        partitions[i].columns = column_names
    return partitions


def setup_seed(seed: int) -> None:
    """Setup seed 

    Parameters
    ----------
    seed: int
        Seed

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def dict_to_dir(directory_dict: dict,
                pt_file_number_limit: int = 2) -> None:
    """Print out the directory structure given its dictionary representation

    Parameters
    -----------
    directory_dict: dict
        A nested dictionary containing the structure of the directory
    pt_file_number_limit: int
        Number of .PT files to be printed 

    """
    space = '    '
    branch = '│   '
    tee = '├── '
    last = '└── '

    def inner(dir_dict, prefix: str = ''):
        non_pt_file_list = [x for x in list(
            dir_dict['files']) if not x.lower().endswith(".pt")]
        pt_file_list = [x for x in list(
            dir_dict['files']) if x.lower().endswith(".pt")]

        contents = list(dir_dict['dirs'].keys()) + \
            non_pt_file_list + pt_file_list[:pt_file_number_limit]
        n_dirs = len(dir_dict['dirs'].keys())
        pointers = [tee] * (len(contents) - 1) + [last]
        dir_counter = 0
        for pointer, name in zip(pointers, contents):
            dir_counter += 1
            if dir_counter <= n_dirs:
                yield prefix + pointer + name
                extension = branch if pointer == tee else space
                yield from inner(dir_dict['dirs'][name], prefix=prefix+extension)
            else:
                yield prefix + pointer + name

    iterator = inner(directory_dict)
    for line in iterator:
        print(line)


def path_to_dict(path: str,
                 d: dict = {'dirs': {}, 'files': set()}) -> dict:
    """Turns a directory into its nested dictionary representation

    Parameters
    ----------
    path: str
        The root path of the directory
    d: dict
        A dictionary (mainly used during recursion)
    Returns
    --------
    dict
        A nested dictionary

    """
    for x in os.listdir(path):
        if (x.startswith(".")) or (x == '__pycache__'):
            continue
        sub_path = os.path.join(path, x)
        name = os.path.basename(sub_path)
        if os.path.isdir(sub_path):
            d['dirs'][name] = {'dirs': {}, 'files': set()}
            path_to_dict(sub_path, d['dirs'][name])
        else:
            d['files'].add(name)
    return d


def check_data_dir_structure(path: str) -> Tuple[list, str]:
    """Check if the given directory satisfies the requirements for a data directory

    Parameters
    -----------
    path: str
        The root path of the directory

    Returns
    ---------
    Tuple[list, str]
        File paths for encoders and the classifier model


    """
    user_data_dir_dict = path_to_dict(
        path=path, d={'dirs': {}, 'files': set()})
    required_dir_structure = {"dirs": {"background_tcrs": {'dirs': {}, 'files': set(["human_alpha.txt",
                                                                                     "human_alpha_embedding.txt",
                                                                                    "human_beta.txt",
                                                                                     "human_beta_embedding.txt",
                                                                                     "mouse_alpha.txt",
                                                                                     "mouse_alpha_embedding.txt",
                                                                                     "mouse_beta.txt",
                                                                                     "mouse_beta_embedding.txt"])},
                                       "data_for_encoders": {'dirs': {"mhc_dict": {'dirs': {}, 'files': set([".PT files here"])},
                                                                      "name_dict": {'dirs': {}, 'files': set([".PT files here"])}},
                                                             'files': set(['Atchley_factors.csv', 'valid_mhc.txt'])},
                                       "encoder_model_weights": {'dirs': {}, 'files': set(["v_alpha_encoder.pth",
                                                                                           "v_beta_encoder.pth",
                                                                                          "cdr3_alpha_encoder.pth",
                                                                                           "cdr3_beta_encoder.pth",
                                                                                           "pmhc_encoder.pth"])},
                                       "classifier_model_weights": {'dirs': {}, 'files': set(["classifier.pth"])}},
                              "files": set()}
    if (not set(required_dir_structure['dirs'].keys()).issubset(set(user_data_dir_dict['dirs'].keys()))) or \
        (not required_dir_structure['dirs']['background_tcrs']['files'].issubset(user_data_dir_dict['dirs']['background_tcrs']['files'])) or \
        (not set(required_dir_structure['dirs']['data_for_encoders']['dirs'].keys()).issubset(set(user_data_dir_dict['dirs']['data_for_encoders']['dirs'].keys()))) or \
            (not required_dir_structure['dirs']['data_for_encoders']['files'].issubset(user_data_dir_dict['dirs']['data_for_encoders']['files'])):
        dict_to_dir(required_dir_structure, pt_file_number_limit=2)
        raise Exception("See above for the required directory structure.\n")

    encoder_path = []
    for m in ["v_alpha_encoder.pth",
              "v_beta_encoder.pth",
              "cdr3_alpha_encoder.pth",
              "cdr3_beta_encoder.pth",
              "pmhc_encoder.pth"]:
        if m not in user_data_dir_dict['dirs']['encoder_model_weights']['files']:
            warnings.warn(m + " can not be found. Will use random weights.\n")
            encoder_path.append(None)
        else:
            encoder_path.append(os.path.join(path, "encoder_model_weights", m))

    if "classifier.pth" not in user_data_dir_dict['dirs']['classifier_model_weights']['files']:
        warnings.warn("classifier.pth" +
                      " can not be found. Will use random weights.\n")
        classifier_path = None
    else:
        classifier_path = os.path.join(
            path, "classifier_model_weights/classifier.pth")
    return encoder_path, classifier_path
