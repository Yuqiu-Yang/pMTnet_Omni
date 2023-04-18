import pytest
import os 
import pandas as pd
import numpy as np

from pMTnet_Omni_Document.data_curation import read_file
from pMTnet_Omni.pMTnet_Omni_class import pMTnet_Omni_class

extra_mhc_dict = {"AAAA": np.zeros((380, 1280))}

@pytest.mark.parametrize("seed, extra_mhc_dict, partition_size, create_incomplete_data, expected", [
    (42, extra_mhc_dict, 5, False, [1, 0, 0, 1]),
    (None, extra_mhc_dict, 2, False, [2, 0, 0, 2]),
    (42, "./tests/test_data/extra_mhc_dict.json", 5, True, [1, 1, 1, 1]),
    (None, "./tests/test_data/extra_mhc_dict.json", 3, True, [2, 2, 2, 2])
])
def test_pMTnet_Omni_class_partitions(seed, extra_mhc_dict, partition_size, create_incomplete_data, expected):
    
    df, _ = read_file(file_path='./tests/test_data/test_df.csv',
                      sep=",")
    pmtnet = pMTnet_Omni_class(model_device=None,
                               data_dir="./tests/test_data/",
                               seed=seed,
                               extra_mhc_dict=extra_mhc_dict)
    pmtnet.read_user_df(
        user_data=df, partition_size=partition_size)
    pmtnet._encode_user_df()
    if create_incomplete_data:
        pmtnet._create_incomplete_data()
    assert (len(pmtnet.user_df_dict['complete']) == expected[0]) and \
        (len(pmtnet.user_tcr_embedding_dict['alpha missing']) == expected[1]) and \
        (len(pmtnet.user_tcr_embedding_dict['beta missing']) == expected[2]) and \
        (len(pmtnet.user_pmhc_embedding_dict['complete']) == expected[3])


@pytest.mark.parametrize("partition_size, create_incomplete_data, load_embedding", [
    (5, False, False),
    (2, True, True)
])
def test_pMTnet_Omni_class_prediction(partition_size, create_incomplete_data, load_embedding):
    df, _ = read_file(file_path='./tests/test_data/test_df.csv',
                      save_results=True,
                      sep=",")
    pmtnet = pMTnet_Omni_class(model_device="cpu",
                               data_dir="./tests/test_data/",
                               seed=None)
    pmtnet.read_user_df(
        user_data="./tests/test_data/df_curated.csv", partition_size=partition_size, sep=",")
    pmtnet._encode_user_df()
    if create_incomplete_data:
        pmtnet._create_incomplete_data()
    pmtnet._get_user_df_output()
    pmtnet._compute_percentile_rank(rank_threshold=0.05, B=2,
                                    load_embedding=load_embedding,
                                    replacement=True,
                                    check_size=[5, 10],
                                    load_size=5,
                                    minibatch_size=2,
                                    log_file_path=None,
                                    model_id=0)
    df_dict, tcr_dict, pmhc_dict = pmtnet._amalgamate()
    os.remove("./tests/test_data/df_curated_antigen_dropped.csv")
    os.remove("./tests/test_data/df_curated_invalid_v.csv")
    os.remove("./tests/test_data/df_curated_mhc_alpha_dropped.csv")
    os.remove("./tests/test_data/df_curated_mhc_beta_dropped.csv")
    os.remove("./tests/test_data/mhc_seq_dict.json")
    assert ("percentile_rank_m_0_0" in df_dict['complete'].columns) and\
        ("percentile_rank_m_0_1" in df_dict['complete'].columns) and\
        (df_dict['complete'].shape[0] == 4) and\
        (tcr_dict['complete'].shape[1] == 70) and\
        (pmhc_dict['complete'].shape[1] == 30) and\
        ((not create_incomplete_data) or (len(df_dict['alpha missing']) == 4)) and\
        ((not create_incomplete_data) or (
            len(df_dict['beta missing']) == 4))


@pytest.mark.parametrize("create_incomplete_data, compute_percentile_rank, output_file_path", [
    (True, True, './tests/test_data/test_df_results.csv'),
    (False, True, './tests/test_data/test_df_results.csv'),
    (False, False, './tests/test_data/test_df_results.csv')
])
def test_pMTnet_Omni_class_prediction_method(create_incomplete_data, compute_percentile_rank, output_file_path):
    df, _ = read_file(file_path='./tests/test_data/test_df.csv',
                      sep=",")
    pmtnet = pMTnet_Omni_class(model_device="cpu",
                               data_dir="./tests/test_data/",
                               seed=None)
    pmtnet.read_user_df(user_data=df)
    try:
        pmtnet.predict(create_incomplete_data=create_incomplete_data,
                       compute_percentile_rank=compute_percentile_rank,
                       output_file_path=output_file_path)
        try:
            os.remove('./tests/test_data/test_df_results_complete.csv')
            os.remove('./tests/test_data/test_df_results_alpha missing.csv')
            os.remove('./tests/test_data/test_df_results_beta missing.csv')
        except:
            pass 
        assert True
    except:
        assert False


@pytest.mark.parametrize("create_incomplete_data, true_labels", [
    (False, [1, 0, 1, 0]),
    (False, np.array([1, 0, 1, 0])),
    (True, pd.DataFrame({"label": [1, 0, 1, 0]})),
])
def test_pMTnet_Omni_validation(create_incomplete_data, true_labels):
    df, _ = read_file(file_path='./tests/test_data/test_df.csv',
                      sep=",")
    pmtnet = pMTnet_Omni_class(model_device="cpu",
                               data_dir="./tests/test_data/",
                               seed=None)
    pmtnet.read_user_df(user_data=df, partition_size=5)
    pmtnet._encode_user_df()
    if create_incomplete_data:
        pmtnet._create_incomplete_data()
    pmtnet._get_user_df_output()
    pmtnet._compute_percentile_rank(rank_threshold=1, B=1,
                                    load_embedding=True,
                                    replacement=True,
                                    check_size=[5, 10],
                                    load_size=5,
                                    minibatch_size=2,
                                    log_file_path='./tests/test_data/pmtnet_omni_predict_log.txt',
                                    model_id=0)
    for b in range(1):
        for missingness in pmtnet.user_df_dict:
            if len(pmtnet.user_df_dict[missingness]) == 0:
                continue
            for partition in pmtnet.user_df_dict[missingness]:
                col_names = ["percentile_rank_m_" + str(model_id) + "_"+str(b) for model_id in range(pmtnet.ensemble_size)]
                pmtnet.user_df_dict[missingness][partition]["percentile_rank_" +str(b)] = pmtnet.user_df_dict[missingness][partition][col_names].mean(axis=1)
    os.remove("./tests/test_data/pmtnet_omni_predict_log_0.txt")
    try:
        pmtnet.validate(true_labels=true_labels)
        assert True
    except:
        assert False
