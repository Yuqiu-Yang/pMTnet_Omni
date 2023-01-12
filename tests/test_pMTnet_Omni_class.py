import pytest 
import pandas as pd 
import numpy as np 

from pMTnet_Omni.pMTnet_Omni_class import pMTnet_Omni_class

@pytest.mark.parametrize("seed, partition_size, create_incomplete_data, expected", [
    (42, 5, False, [1, 0, 0, 1]),
    (None, 2, False, [2, 0, 0, 2]),
    (42, 5, True, [1, 1, 1, 1]),
    (None, 3, True, [2, 2, 2, 2])
])
def test_pMTnet_Omni_class_partitions(seed, partition_size, create_incomplete_data, expected):
    pmtnet = pMTnet_Omni_class(model_device="cpu",
                               background_data_dir="./tests/test_data/background_tcrs/",
                               encoder_data_dir="./tests/test_data/data_for_encoders/",
                               classifier_checkpoint_path=None,
                               vGdVAEacheckpoint_path=None,
                               vGdVAEbcheckpoint_path=None,
                               cdr3VAEacheckpoint_path=None,
                               cdr3VAEbcheckpoint_path=None,
                               pMHCcheckpoint_path=None,
                               seed=seed)
    pmtnet.read_user_df(user_data_path='./tests/test_data/test_df.csv', partition_size=partition_size)
    pmtnet.encode_user_df()
    if create_incomplete_data:
        pmtnet.create_incomplete_data()
    assert (len(pmtnet.user_df_dict['complete']) == expected[0]) and \
            (len(pmtnet.user_tcr_embedding_dict['alpha missing']) == expected[1]) and \
            (len(pmtnet.user_tcr_embedding_dict['beta missing']) == expected[2]) and \
            (len(pmtnet.user_pmhc_embedding_dict['complete']) == expected[3])

@pytest.mark.parametrize("partition_size, create_incomplete_data, load_embedding" ,[
    (5, False, False),
    (2, True, True)
])
def test_pMTnet_Omni_class_prediction(partition_size, create_incomplete_data, load_embedding):
    pmtnet = pMTnet_Omni_class(model_device="cpu",
                               background_data_dir="./tests/test_data/background_tcrs/",
                               encoder_data_dir="./tests/test_data/data_for_encoders/",
                               classifier_checkpoint_path=None,
                               vGdVAEacheckpoint_path=None,
                               vGdVAEbcheckpoint_path=None,
                               cdr3VAEacheckpoint_path=None,
                               cdr3VAEbcheckpoint_path=None,
                               pMHCcheckpoint_path=None,
                               seed=None)
    pmtnet.read_user_df(user_data_path='./tests/test_data/test_df.csv', partition_size=partition_size)
    pmtnet.encode_user_df()
    if create_incomplete_data:
        pmtnet.create_incomplete_data()
    pmtnet.get_user_df_output()
    pmtnet.predict(rank_threshold=0.05, B=2,
                   load_embedding=load_embedding, 
                   replacement=True, 
                   check_size=[5,10], 
                   load_size=5,
                   minibatch_size=2,
                   log_file_path='./tests/test_data/pmtnet_omni_predict_log.txt')
    df_dict = pmtnet._unpartition()
    assert ("rank_0" in df_dict['complete'].columns) and ("rank_1" in df_dict['complete'].columns) and \
            (df_dict['complete'].shape[0] == 4) and \
                ((not create_incomplete_data) or (len(df_dict['alpha missing']) == 4)) and \
                   ((not create_incomplete_data) or (len(df_dict['beta missing']) == 4)) 


@pytest.mark.parametrize("create_incomplete_data, true_labels", [
    (False, [1,0,1,0]),
    (False, np.array([1,0,1,0])),
    (True, pd.DataFrame({"label":[1,0,1,0]})),
])
def test_pMTnet_Omni_validation(create_incomplete_data, true_labels):
    pmtnet = pMTnet_Omni_class(model_device="cpu",
                            background_data_dir="./tests/test_data/background_tcrs/",
                            encoder_data_dir="./tests/test_data/data_for_encoders/",
                            classifier_checkpoint_path=None,
                            vGdVAEacheckpoint_path=None,
                            vGdVAEbcheckpoint_path=None,
                            cdr3VAEacheckpoint_path=None,
                            cdr3VAEbcheckpoint_path=None,
                            pMHCcheckpoint_path=None,
                            seed=None)
    pmtnet.read_user_df(user_data_path='./tests/test_data/test_df.csv', partition_size=5)
    pmtnet.encode_user_df()
    if create_incomplete_data:
        pmtnet.create_incomplete_data()
    pmtnet.get_user_df_output()
    pmtnet.predict(rank_threshold=1, B=1,
                   load_embedding=True, 
                   replacement=True, 
                   check_size=[5,10], 
                   load_size=5,
                   minibatch_size=2,
                   log_file_path='./tests/test_data/pmtnet_omni_predict_log.txt')
    try:
        pmtnet.validate(true_labels=true_labels,
                        roc_plot_path='./tests/test_data/roc_plots.pdf')
        assert True
    except:
        assert False