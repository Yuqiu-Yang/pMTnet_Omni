import pytest
import os 
import numpy as np

from pMTnet_Omni_Document import read_file
from pMTnet_Omni.pMTnet_Omni_class import pMTnet_Omni_class

extra_mhc_dict = {"AAAA": np.zeros((380, 1280))}

@pytest.mark.parametrize("seed, extra_mhc_dict, partition_size, expected", [
    (42, extra_mhc_dict, 5, 4),
    (None, extra_mhc_dict, 2, 4),
    (42, "./tests/test_data/extra_mhc_dict.json", 5, 4),
    (None, "./tests/test_data/extra_mhc_dict.json", 3, 4)
])
def test_pMTnet_Omni_class_partitions(seed, extra_mhc_dict, partition_size, expected):
    
    df, _ = read_file(file_path='./tests/test_data/test_df.csv',
                      sep=",")
    pmtnet = pMTnet_Omni_class(model_device=None,
                               data_dir="./tests/test_data/",
                               seed=seed,
                               extra_mhc_dict=extra_mhc_dict,
                               partition_size=partition_size)
    pmtnet._read_user_df(user_data=df)
    pmtnet._encode_user_df()
    assert (pmtnet.user_df.shape[0] == expected)


@pytest.mark.parametrize("compute_percentile_rank, output_file_path", [
    (True, './tests/test_data/test_df_results.csv'),
    (False, './tests/test_data/test_df_results.csv')
])
def test_pMTnet_Omni_class_prediction_method(compute_percentile_rank, output_file_path):
    df, _ = read_file(file_path='./tests/test_data/test_df.csv',
                      sep=",")
    pmtnet = pMTnet_Omni_class(model_device="cpu",
                               data_dir="./tests/test_data/",
                               seed=None)
    pmtnet.read_user_df(user_data=df)
    try:
        pmtnet.predict(compute_percentile_rank=compute_percentile_rank,
                       output_file_path=output_file_path,
                       check_size=10, load_size=10, minibatch_size=10)
        try:
            os.remove('./tests/test_data/test_df_results.csv')
        except:
            pass 
        assert True
    except:
        assert False


