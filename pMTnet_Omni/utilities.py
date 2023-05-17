# Data IO
import os
import warnings
import pandas as pd
from copy import deepcopy

# Numeric manipulation
import numpy as np
import math
import random

# PyTorch
import torch

# ROC curve
# from sklearn import metrics
import matplotlib as mpl
from matplotlib import pyplot as plt

# Typing
from typing import Tuple, Union, Optional


def plot_validation_curve(result_dicts: dict,
                          plot_type: str,
                          primary_type: str,
                          secondary_types: Optional[Union[str, list]] = None,
                          title: Optional[str] = None,
                          style_sheet: str='seaborn-v0_8-white',
                          figsize: tuple=(20, 20),
                          title_fontsize: float=60,
                          linewidth: float=10,
                          xlabel_fontsize: float=55,
                          ylabel_fontsize: float=55,
                          xticks_fontsize: float=45,
                          yticks_fontsize: float=45,
                          xlim: list=[0,1],
                          ylim: list=[0,1],
                          axis: str="on",
                          legend_position: str="best",
                          legend_fontsize: float=55,
                          handlelength: float=2,
                          legend_title_fontsize: float=55,
                          bbox_to_anchor = (1,1),
                          visible_spines: bool=False,
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
    type_map = {"complete": "Both",
                "alpha missing": "Beta",
                "beta missing": "Alpha",
                "overall": "All",
                "human overall": "Human",
                "mouse overall": "Mouse",
                "human class i": "Human C1",
                "human class ii": "Human C2",
                "mouse class i": "Mouse C1",
                "mouse class ii": "Mouse C2",
                "class i": "Class I",
                "class ii": "Class II"}

    if plot_type == "roc":
        x_key = "fpr"
        x_label = '1 - Specificity'
        y_key = "tpr"
        y_label = 'Sensitivity'
        auc_key = "auroc"
    elif plot_type == "prc":
        x_key = "recall"
        x_label = "Recall"
        y_key = "precision"
        y_label = "Precision"
        auc_key = "auprc"
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
                          "mouse class i", "mouse class ii",
                          "class i", "class ii"]:
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

    plt.style.use(style_sheet)
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=title_fontsize)

    counter = 0
    for s_type in plot_dict:
        plt.plot(plot_dict[s_type]["x"],
                 plot_dict[s_type]["y"],
                 linewidth=linewidth, color=colors.colors[counter],
                 label=type_map[s_type]+": "+str(plot_dict[s_type]["auc"]))
        counter += 1

    plt.xlabel(x_label, fontsize=xlabel_fontsize)
    plt.ylabel(y_label, fontsize=ylabel_fontsize)

    plt.xticks(fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.axis(axis)
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(visible_spines)
    
    plt.legend(loc=legend_position,
               bbox_to_anchor=bbox_to_anchor,
               fontsize=legend_fontsize, 
               handlelength=handlelength,
               title="AUC", title_fontsize=legend_title_fontsize)
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


def split_dataframe(df: pd.DataFrame,
                    by_columns: list,
                    partition_size: int=500) -> Tuple[dict, dict]:
    """Split a dataframe by columns 

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be splitted 
    by_columns : list
        Columns to be used for splitting 
    partition_size: int 
        The chunk size
    Returns
    -------
    dict
        A dictionary of section of the original dataframe. 
        The key would be unique values from the dataframe. 
    """
    splits = {}
    subdf = df[by_columns].groupby(by_columns,
                                   sort=False, axis=0, dropna=False).size().reset_index()

    subdf.drop(columns=[0], axis=1, inplace=True)
    
    ref_dict = {"partitions": partition_dataframe(df=subdf,
                                                  partition_size=partition_size)}
    for i in range(subdf.shape[0]):
        key = tuple(subdf.iloc[i,:])
        value = deepcopy(df)
        for n, col in enumerate(by_columns):
            value = value[value[col] == key[n]].reset_index(drop=True)
        splits[key] = value   
        ref_dict[key] = None
    return splits, ref_dict


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


def check_data_dir_structure(path: str) -> Tuple[list, list]:
    """Check if the given directory satisfies the requirements for a data directory

    Parameters
    -----------
    path: str
        The root path of the directory

    Returns
    ---------
    Tuple[list, list]
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
                                       "classifier_model_weights": {'dirs': {}, 'files': set(["classifier*.pth"])}},
                              "files": set()}
    if (not set(required_dir_structure['dirs'].keys()).issubset(set(user_data_dir_dict['dirs'].keys()))):
        print("Detected data directory structure:\n")
        dict_to_dir(user_data_dir_dict, pt_file_number_limit=1)
        print("The required data directory structure is:\n")
        dict_to_dir(required_dir_structure, pt_file_number_limit=1)
        basic_missing = [x for x in set(required_dir_structure['dirs'].keys()) - set(user_data_dir_dict['dirs'].keys())]
        raise Exception("Detected missing directories are:\n" + "; ".join(basic_missing) + "\n")
        
    if (not required_dir_structure['dirs']['background_tcrs']['files'].issubset(user_data_dir_dict['dirs']['background_tcrs']['files'])) or \
        (not set(required_dir_structure['dirs']['data_for_encoders']['dirs'].keys()).issubset(set(user_data_dir_dict['dirs']['data_for_encoders']['dirs'].keys()))) or \
            (not required_dir_structure['dirs']['data_for_encoders']['files'].issubset(user_data_dir_dict['dirs']['data_for_encoders']['files'])):
        print("Detected data directory structure:\n")
        dict_to_dir(user_data_dir_dict, pt_file_number_limit=2)
        print("The required data directory structure is:\n")
        dict_to_dir(required_dir_structure, pt_file_number_limit=2)
        
        background_missing = [x for x in required_dir_structure['dirs']['background_tcrs']['files'] - user_data_dir_dict['dirs']['background_tcrs']['files']]
        encoders_data_file_missing = [x for x in required_dir_structure['dirs']['data_for_encoders']['files'] - user_data_dir_dict['dirs']['data_for_encoders']['files']]
        encoders_data_dir_missing = [x for x in set(required_dir_structure['dirs']['data_for_encoders']['dirs'].keys()) - set(user_data_dir_dict['dirs']['data_for_encoders']['dirs'].keys())]
        
        raise Exception("Detected missing files in the background_tcrs directory are:\n" + "; ".join(background_missing) + 
                        "\n Detected missing files in the data_for_encoders directory are:\n" + "; ".join(encoders_data_file_missing) + 
                        "\nDetected missing directories in the data_for_encoders directory are:\n" + "; ".join(encoders_data_dir_missing) + "\n")

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
    
    classifier_path = []
    for m in user_data_dir_dict['dirs']['classifier_model_weights']['files']:
        if m.startswith("classifier") and m.endswith(".pth"):
            c_path = os.path.join(path, "classifier_model_weights", m)
            classifier_path.append(c_path)
    if len(classifier_path) == 0:
        warnings.warn("classifier.pth" +
                      " can not be found. Will use random weights.\n")
        classifier_path.append(None)
                
    return encoder_path, classifier_path
