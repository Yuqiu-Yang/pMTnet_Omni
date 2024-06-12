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

# Typing
from typing import Tuple, Union, Optional


def batchify_check_size(check_size: Union[int, list] = [10, 20],
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
    if isinstance(check_size, int):
        check_size = [check_size]
    for size in check_size:
        if size <= minibatch_size:
            # if the size is less than the size of the dataset
            # we simply load that many data
            batch_size.append(size)
            batch_finished_indicator.append(True)
        else:
            # Otherwise, we batchify
            n_batch = math.ceil(size/minibatch_size)
            size_left = size
            i = 1
            while i < n_batch:
                batch_size.append(minibatch_size)
                size_left -= minibatch_size
                # Add False to the indicator to let
                # the program know that we are not finished
                # checking for this size yet
                batch_finished_indicator.append(False)
                i += 1
            batch_size.append(size_left)
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
        partitions[i] = df[i*partition_size: (i+1)*partition_size].reset_index(drop=True)
        partitions[i].columns = column_names
    return partitions


def amalgamate_partitions(partition_dict: dict) -> pd.DataFrame:
    """Amalgamate partitions back to one data frame 

    Parameters
    ----------
    partition_dict : dict
        A dictionary of data frames 

    Returns
    -------
    pd.DataFrame
        A dataframe 
    """
    temp = [partition_dict[partition] for partition in partition_dict]
    df = pd.concat(temp, axis=0, join='inner', ignore_index=True)
    return df


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


def check_data_dir_structure(path: str) -> Tuple[list, list, str]:
    """Check if the given directory satisfies the requirements for a data directory

    Parameters
    -----------
    path: str
        The root path of the directory

    Returns
    ---------
    Tuple[list, list, str]
        File paths for encoders and the classifier model and the model information 


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
                                       "background_pmhcs": {'dirs': {}, 'files': set(['pmhc_embedding.csv'])},
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
    model_info_path = ""
    for m in user_data_dir_dict['dirs']['classifier_model_weights']['files']:
        if m.startswith("classifier") and m.endswith(".pth"):
            c_path = os.path.join(path, "classifier_model_weights", m)
            classifier_path.append(c_path)
        if m.startswith("model_info"):
            model_info_path = os.path.join(path, "classifier_model_weights", m)
    if len(classifier_path) == 0:
        warnings.warn("classifier.pth" +
                      " can not be found. Will use random weights.\n")
        classifier_path.append(None)
                
    return encoder_path, classifier_path, model_info_path
