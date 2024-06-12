import pytest
import os 
import pandas as pd
import numpy as np

from pMTnet_Omni.utilities import   batchify_check_size, \
                                    partition_dataframe, \
                                    amalgamate_partitions, \
                                    setup_seed, \
                                    path_to_dict, \
                                    dict_to_dir, \
                                    check_data_dir_structure


##########################
# batchify_check_size


@pytest.mark.parametrize("check_size, minibatch_size, expected", [
    ([10, 50], 30, ([10, 30, 20], [True, False, True]))
])
def test_batchify_check_size(check_size, minibatch_size, expected):
    batch_size, batch_finished_indicator = batchify_check_size(check_size=check_size,
                                                               minibatch_size=minibatch_size)
    assert (batch_size == expected[0]) and (
        batch_finished_indicator == expected[1])

#########################
# partition_dataframe


@pytest.mark.parametrize("df, partition_size, expected", [
    (pd.DataFrame({'a': [1, 2, 3], 'b':[4, 5, 6]}), 2, [2, 1])
])
def test_partition_dataframe(df, partition_size, expected):
    partitions = partition_dataframe(df=df, partition_size=partition_size)
    assert (partitions[0].shape[0] == expected[0]) and (
        partitions[1].shape[0] == expected[1])

#########################
# amalgamate_partitions

@pytest.mark.parametrize("partition_dict, expected",[
    ({0: pd.DataFrame({'a': [1, 2, 3], 'b':[4, 5, 6]}),
      1: pd.DataFrame({'a': [1, 2, 3], 'b':[4, 5, 6]})}, [6,2])
])
def test_amalgamate_partitions(partition_dict, expected):
    df = amalgamate_partitions(partition_dict=partition_dict)
    assert (df.shape[0] == expected[0]) and (df.shape[1] == expected[1])

########################
# setup_seed

@pytest.mark.parametrize("seed", [
    (42)
]
)
def test_setup_seed(seed):
    try:
        setup_seed(seed=seed)
        assert True
    except:
        assert False

######################
# path_to_dict


@pytest.mark.parametrize("path", [
    ("./tests/test_data")
])
def test_path_to_dict(path):
    d = path_to_dict(path=path, d={'dirs': {}, 'files': set()})
    assert set(["background_tcrs", "data_for_encoders", "encoder_model_weights", "classifier_model_weights"]).issubset(set(d['dirs'].keys())) and \
        ("test_df.csv" in d['files'])

# dict_to_dir


@pytest.mark.parametrize("path", [
    ("./tests/test_data")
])
def test_dict_to_dir(path):
    d = path_to_dict(path=path, d={'dirs': {}, 'files': set()})
    try:
        dict_to_dir(directory_dict=d, pt_file_number_limit=2)
        assert True
    except:
        assert False

# check_data_dir_structure


@pytest.mark.parametrize("path", [
    ("./tests/test_data")
])
def test_check_data_dir_structure(path):
    encoder_path, classifier_path, model_info_path = check_data_dir_structure(path=path)
    assert all([x is None for x in encoder_path]) and all([x is None for x in classifier_path]) and (model_info_path == "")


@pytest.mark.parametrize("path", [
    ("./tests/"),
    ("./tests/error_data_dir/")
])
def test_check_data_dir_structure_error(path):
    try:
        encoder_path, classifier_path, model_info_path = check_data_dir_structure(path=path)
        assert False
    except:
        assert True
