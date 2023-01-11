import pytest 
import pandas as pd 
import numpy as np 

from pMTnet_Omni.encoders.encoder_class import encoder_class

test_df = pd.read_csv("./tests/test_data/test_df.csv")
test_df1 = pd.DataFrame(np.random.randn(10, 70))
column_names = ["va" for i in range(5)]
column_names.extend(["vb" for i in range(5)])
column_names.extend(["cdr3a" for i in range(30)])
column_names.extend(["cdr3b" for i in range(30)])
test_df1.columns = column_names

@pytest.mark.parametrize("model_divce, vGdVAEacheckpoint_path, vGdVAEbcheckpoint_path, cdr3VAEacheckpoint_path, cdr3VAEbcheckpoint_path,pMHCcheckpoint_path, df, is_embedding, verbose, expected", [
    ("cpu", None, None, None, None, None, test_df, False, True, [70, 30]),
    ("cpu", None, None, None, None, None, test_df1, True, False, [70, None])
])
def test_encoder_class(model_divce, vGdVAEacheckpoint_path, vGdVAEbcheckpoint_path, cdr3VAEacheckpoint_path, cdr3VAEbcheckpoint_path, 
                          pMHCcheckpoint_path, df, is_embedding, verbose, expected):
    encoder = encoder_class(encoder_data_dir='./tests/test_data/data_for_encoders/',
                            model_device=model_divce, 
                                vGdVAEacheckpoint_path=vGdVAEacheckpoint_path,
                                vGdVAEbcheckpoint_path=vGdVAEbcheckpoint_path,
                                cdr3VAEacheckpoint_path=cdr3VAEacheckpoint_path,
                                cdr3VAEbcheckpoint_path=cdr3VAEbcheckpoint_path,
                                pMHCcheckpoint_path=pMHCcheckpoint_path)
    tcr_embedding, pmhc_embedding = encoder.encode(df=df, is_embedding=is_embedding, verbose=verbose)                         
    assert (tcr_embedding.shape[1] == expected[0]) and ((pmhc_embedding is expected[1]) or (pmhc_embedding.shape[1] == expected[1]))