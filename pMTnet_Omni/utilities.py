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
from sklearn import metrics
from matplotlib import pyplot as plt

# Progress bar
from tqdm import tqdm

# Typing
from typing import Tuple, Union


def check_column_names(df: pd.DataFrame,
                       background_tcrs_dir: str = "./data/background_tcrs/") -> pd.DataFrame:
    """Check if the columns are correct 
    This function checks and corrects the column names provided in the user dataframe.
    What's more, it can also automatically create some missing columns.
    The column name to be used in all the following functions are:
    "va": name of alpha V gene (NAMING CONVENTION HERE)
    "cdr3a": amino acid sequence of CDR3 region on the alpha chain
    "vaseq": amino acid sequence of the alpha V gene 
    "vb": name of beta V gene
    "cdr3b": amino acid sequence of CDR3 region on the beta chain
    "vbseq": amino acid sequence of the beta V gene 
    "peptide": amino acid sequence of the peptide presented by mhc 
    "mhca": name of mhc alpha chain (NAMING CONVENTION HERE)
    "mhcb": name of mhc beta chain 
    "tcr_species": species of TCR (human or mouse)
    "pmhc_species": species of peptide-mhc

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
                           "peptide", "mhca", "mhcb", "tcr_species", "pmhc_species"]
    # Get rid of spaces, convert names to lowercase, remove special characters
    print("Checking column names...\n")
    df.columns = df.columns.str.replace(' ','').str.lower().str.replace(
        "_", "").str.replace(r'\W', '', regex=True)
    df_cols = df.columns.tolist()
    for i in range(len(df_cols)):
        df_cols[i] = re.sub(r'(?<=tcr).*(?=species)', "_", df_cols[i])
        df_cols[i] = re.sub(r'(?<=pmhc).*(?=species)', "_", df_cols[i])
    df.columns = df_cols
    # Get names not presented in the user dataframe
    user_df_set = set(df_cols)
    required_set = set(common_column_names)
    diff_set = required_set - user_df_set

    error_messages = "".join(["Column " + name + " can not be found.\n"
                              for name in ["cdr3a", "cdr3b", "peptide", "mhca", "mhcb",
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


def check_data_sanity(df: pd.DataFrame,
                      mhc_path: str = "./data/data_for_encoders/valid_mhc.txt") -> pd.DataFrame:
    """Check the data sanity
    This function will check if the data format conforms to what our model expects

    Parameters
    ---------
    df: pd.DataFrame
        A pandas dataframe containing pairing data
    mhc_path: str
        The file path to valide mhcs 

    Returns
    ---------
    df.DataFrame
        A pandas dataframe containing currated pairing data
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

    # 3. input MHC that is not in the ESM dictionary will be dropped
    num_row = df.shape[0]
    with open(mhc_path, 'r') as f:
        mhc_dic_keys = f.read().splitlines()
    mhc_dic_keys = set(mhc_dic_keys)
    df_mhc_alpha_dropped = df[~df["mhca"].isin(mhc_dic_keys)]
    df_mhc_beta_dropped = df[~df["mhcb"].isin(mhc_dic_keys)]
    df = df[df["mhca"].isin(mhc_dic_keys)]
    df = df[df["mhcb"].isin(mhc_dic_keys)]
    if((num_row-df.shape[0]) > 0):
        print(str(num_row-df.shape[0]) +
              ' MHCs without ESM embedding are dropped:')
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

    return df


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


def get_mhc_class(df: pd.DataFrame) -> pd.DataFrame:
    """Infer the MHC classes

    Parameters
    ---------
    df: pd.DataFrame
        A pandas dataframe containing mhca, mhcb

    Returns
    ---------
    pd.DataFrame
        A column of a pandas dataframe with the inferred MHC classes 
    """
    df = df.reset_index()
    # Human class I starts with A, B, or C
    human_class_i = ("A", "B", "C")
    # Class II starts with DP, DQ, or DR
    human_class_ii = ("DP", "DQ", "DR")
    # Mouse class I starts with D, K, L, Q
    mouse_class_i = tuple(["H-2-" + i for i in ["D", "K", "L", "Q"]])
    # Class II starts with IA, IE
    mouse_class_ii = tuple(["H-2-" + i for i in ["IA", "IE"]])

    mhc_class = []
    for i in range(df.shape[0]):
        mhca = df.at[i, "mhca"]
        mhcb = df.at[i, "mhcb"]
        if (mhca.startswith(human_class_i)) or (mhcb.startswith(human_class_i)):
            mhc_class.append('human class i')
        elif (mhca.startswith(human_class_ii)) or (mhcb.startswith(human_class_ii)):
            mhc_class.append('human class ii')
        elif (mhca.startswith(mouse_class_i)) or (mhcb.startswith(mouse_class_i)):
            mhc_class.append('mouse class i')
        elif (mhca.startswith(mouse_class_ii)) or (mhcb.startswith(mouse_class_ii)):
            mhc_class.append('mouse class ii')
        else:
            raise ValueError("The class of neither "+mhca +
                             " nor "+mhcb+" can be recognized.")
    return pd.DataFrame(data={"mhc_class": mhc_class})


def read_file(file_path: str,
              sep: str = ",",
              header: int = 0,
              background_tcr_dir: str = "./data/background_tcrs/",
              mhc_path: str = "./data/data_for_encoders/valid_mhc.txt") -> pd.DataFrame:
    """Reads in user dataframe and performs some basic data curration 

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
        The file path to valide mhcs 

    Returns
    -------
    pd.DataFrame
        A currated pandas dataframe 

    """
    print('Attempting to read in the dataframe...\n')
    df = pd.read_csv(file_path, header=header, sep=sep)
    print("Number of rows in raw dataset: " + str(df.shape[0]))
    # Check column names
    df = check_column_names(df, background_tcr_dir)
    # Perform some basic curation
    df = check_data_sanity(df, mhc_path)
    mhc_class = get_mhc_class(df=df)
    df = pd.concat([df, mhc_class], axis=1, ignore_index=False)
    return df


def get_auroc(true_labels: Union[np.ndarray, pd.DataFrame],
              predictions: Union[np.ndarray, pd.DataFrame],
              ndigits: int = 3) -> float:
    """Compute AUROC

    Parameters
    ---------
    true_labels: Union[np.ndarray, pd.DataFrame]
        The true labels 
    predictions: Union[np.ndarray, pd.DataFrame]
        The output of the model 

    Returns
    --------
    float
        AUROC

    """
    auc = metrics.roc_auc_score(true_labels, predictions)
    return round(auc, ndigits=ndigits)


def plot_roc_curve(true_labels: Union[np.ndarray, pd.DataFrame],
                   predictions: Union[np.ndarray, pd.DataFrame],
                   label: str = "") -> None:
    """Plot roc curve 

    Parameters
    ---------
    true_labels: Union[np.ndarray, pd.DataFrame]
        The true labels 
    predictions: Union[np.ndarray, pd.DataFrame]
        The output of the model 
    label:
        Plot title 

    """
    fpr, tpr, _ = metrics.roc_curve(true_labels, predictions)
    f = plt.figure()
    plt.plot(fpr, tpr)
    plt.title(label=label)
    return f


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
