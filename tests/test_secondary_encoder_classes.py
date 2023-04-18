import pytest
import pandas as pd
import numpy as np
import torch
import csv
import os
import shutil
from tqdm import tqdm

from pMTnet_Omni_Document.data_curation import read_file
from pMTnet_Omni.encoders.pmhc_encoder_class import pmhc_encoder_class
from pMTnet_Omni.encoders.tcr_encoder_class import tcr_encoder_class

# Load dummy weights if actual weights do not exist
# pmhc
try:
    from pMTnet_Omni.models.pmhc_encoder_model import pMHC
except ImportError:
    from pMTnet_Omni.models.load_dummies import pMHC
# va
try:
    from pMTnet_Omni.models.v_alpha_encoder_model import vGdVAEa
except ImportError:
    from pMTnet_Omni.models.load_dummies import vGdVAEa
# vb
try:
    from pMTnet_Omni.models.v_beta_encoder_model import vGdVAEb
except ImportError:
    from pMTnet_Omni.models.load_dummies import vGdVAEb
# cdr3a
try:
    from pMTnet_Omni.models.cdr3_alpha_encoder_model import cdr3VAEa
except ImportError:
    from pMTnet_Omni.models.load_dummies import cdr3VAEa
# cdr3b
try:
    from pMTnet_Omni.models.cdr3_beta_encoder_model import cdr3VAEb
except ImportError:
    from pMTnet_Omni.models.load_dummies import cdr3VAEb



aa_dict_atchley = dict()
with open('./tests/test_data/data_for_encoders/Atchley_factors.csv', 'r') as aa:
    aa_reader = csv.reader(aa)
    next(aa_reader, None)
    for rows in aa_reader:
        aa_name = rows[0]
        aa_factor = rows[1:len(rows)]
        aa_dict_atchley[aa_name] = np.asarray(
            aa_factor, dtype='float')

mhc_files = os.listdir('./tests/test_data/data_for_encoders/name_dict')
if mhc_files[0].startswith('.'):
    mhc_files.pop(0)
mhc_dict = {}
for file_name in tqdm(mhc_files):
    with open("./tests/test_data/data_for_encoders/name_dict/"+file_name) as f:
        line = f.readline()
    mhc_dict[line] = "./tests/test_data/data_for_encoders/mhc_dict/" + file_name

test_df, _ = read_file("./tests/test_data/test_df.csv")


@pytest.mark.parametrize("model_device, pMHCcheckpoint_path, df, aa_dict_atchley, mhc_dict, verbose, expected", [
    ("cpu", None, test_df, aa_dict_atchley, mhc_dict, True, 30),
    ("cpu", "./tests/test_data/dummy_weights/dummy_pmhc.pth", test_df, aa_dict_atchley, mhc_dict, True, 30)
])
def test_pmhc_encoder_class(model_device, pMHCcheckpoint_path, df, aa_dict_atchley, mhc_dict, verbose, expected):
    if pMHCcheckpoint_path is not None:
        os.mkdir("./tests/test_data/dummy_weights/")
        with torch.no_grad():
            pmhc = pMHC()
            torch.save({
                "net": pmhc.state_dict()
            }, "./tests/test_data/dummy_weights/dummy_pmhc.pth")
            
    encoder = pmhc_encoder_class(
        model_device=model_device, pMHCcheckpoint_path=pMHCcheckpoint_path)
    embedding = encoder.encode(
        df=df, aa_dict_atchley=aa_dict_atchley, mhc_dict=mhc_dict, verbose=verbose)
    # purge the directory
    try:
        shutil.rmtree("./tests/test_data/dummy_weights/")
    except:
        pass 
    assert embedding.shape[1] == expected


test_df1 = pd.DataFrame(np.random.randn(10, 70))
column_names = ["va" for i in range(5)]
column_names.extend(["vb" for i in range(5)])
column_names.extend(["cdr3a" for i in range(30)])
column_names.extend(["cdr3b" for i in range(30)])
test_df1.columns = column_names


@pytest.mark.parametrize("model_divce, vGdVAEacheckpoint_path, vGdVAEbcheckpoint_path, cdr3VAEacheckpoint_path, cdr3VAEbcheckpoint_path, df, aa_dict_atchley, is_embedding, verbose, expected", [
    ("cpu", None, None, None, None, test_df, aa_dict_atchley, False, True, 70),
    ("cpu", None, None, None, None, test_df1, aa_dict_atchley, True, False, 70),
    ("cpu", "./tests/test_data/dummy_weights/dummy_va.pth", "./tests/test_data/dummy_weights/dummy_vb.pth",
     "./tests/test_data/dummy_weights/dummy_cdr3a.pth", "./tests/test_data/dummy_weights/dummy_cdr3b.pth", test_df1, aa_dict_atchley, True, False, 70)
])
def test_tcr_encoder_class(model_divce, vGdVAEacheckpoint_path, vGdVAEbcheckpoint_path, cdr3VAEacheckpoint_path, cdr3VAEbcheckpoint_path,
                           df, aa_dict_atchley, is_embedding, verbose, expected):
    if vGdVAEacheckpoint_path is not None:
        os.mkdir("./tests/test_data/dummy_weights/")
        with torch.no_grad():
            va = vGdVAEa()
            vb = vGdVAEb()
            cdr3a = cdr3VAEa()
            cdr3b = cdr3VAEb()
            torch.save({
                "net": va.state_dict()
            }, "./tests/test_data/dummy_weights/dummy_va.pth")
            torch.save({
                "net": vb.state_dict()
            }, "./tests/test_data/dummy_weights/dummy_vb.pth")
            torch.save({
                "net": cdr3a.state_dict()
            }, "./tests/test_data/dummy_weights/dummy_cdr3a.pth")
            torch.save({
                "net": cdr3b.state_dict()
            }, "./tests/test_data/dummy_weights/dummy_cdr3b.pth")
            
    encoder = tcr_encoder_class(model_device=model_divce,
                                vGdVAEacheckpoint_path=vGdVAEacheckpoint_path,
                                vGdVAEbcheckpoint_path=vGdVAEbcheckpoint_path,
                                cdr3VAEacheckpoint_path=cdr3VAEacheckpoint_path,
                                cdr3VAEbcheckpoint_path=cdr3VAEbcheckpoint_path)
    embedding = encoder.encode(
        df=df, aa_dict_atchley=aa_dict_atchley, is_embedding=is_embedding, verbose=verbose)
    # purge the directory 
    try:
        shutil.rmtree("./tests/test_data/dummy_weights/")
    except:
        pass 
    assert embedding.shape[1] == expected
