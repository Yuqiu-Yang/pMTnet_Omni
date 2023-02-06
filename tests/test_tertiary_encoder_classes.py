import pytest
import torch
import pandas as pd
import numpy as np
import csv
import os
from tqdm import tqdm

from pMTnet_Omni.utilities import read_file
from pMTnet_Omni.encoders.utilities import peptide_map, mhc_map
from pMTnet_Omni.encoders.cdr3_alpha_encoder_model import cdr3VAEa
from pMTnet_Omni.encoders.cdr3_beta_encoder_model import cdr3VAEb
from pMTnet_Omni.encoders.v_alpha_encoder_model import vGdVAEa
from pMTnet_Omni.encoders.v_beta_encoder_model import vGdVAEb
from pMTnet_Omni.encoders.pmhc_encoder_model import pMHC

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

test_df = read_file("./tests/test_data/test_df.csv",
                    background_tcr_dir="./tests/test_data/background_tcrs/",
                    mhc_path="./tests/test_data/data_for_encoders/valid_mhc.txt")

#####################
# CDR3A


@pytest.mark.parametrize("df, column_name, aa_dict_atchley, padding, verbose, expected", [
    (test_df, "cdr3a", aa_dict_atchley, 25, True, 30)
])
def test_cdr3VAEa(df, column_name, aa_dict_atchley, padding, verbose, expected):
    with torch.no_grad():
        model = cdr3VAEa()
        seq = torch.Tensor(peptide_map(df=df,
                                       column_name=column_name,
                                       aa_dict_atchley=aa_dict_atchley,
                                       padding=padding,
                                       verbose=verbose)).to("cpu")
        encoded, _, _, _ = model(seq)
    assert encoded.shape[1] == expected

#################
# CDR3B


@pytest.mark.parametrize("df, column_name, aa_dict_atchley, padding, verbose, expected", [
    (test_df, "cdr3b", aa_dict_atchley, 25, True, 30)
])
def test_cdr3VAEb(df, column_name, aa_dict_atchley, padding, verbose, expected):
    with torch.no_grad():
        model = cdr3VAEb()
        seq = torch.Tensor(peptide_map(df=df,
                                       column_name=column_name,
                                       aa_dict_atchley=aa_dict_atchley,
                                       padding=padding,
                                       verbose=verbose)).to("cpu")
        encoded, _, _, _ = model(seq)
    assert encoded.shape[1] == expected

################
#  VA


@pytest.mark.parametrize("df, column_name, aa_dict_atchley, padding, verbose, expected", [
    (test_df, "vaseq", aa_dict_atchley, 100, True, 5)
])
def test_vGdVAEa(df, column_name, aa_dict_atchley, padding, verbose, expected):
    with torch.no_grad():
        model = vGdVAEa()
        seq = torch.Tensor(peptide_map(df=df,
                                       column_name=column_name,
                                       aa_dict_atchley=aa_dict_atchley,
                                       padding=padding,
                                       verbose=verbose)).to("cpu")
        encoded, _, _, _ = model(seq)
    assert encoded.shape[1] == expected

################
#  VB


@pytest.mark.parametrize("df, column_name, aa_dict_atchley, padding, verbose, expected", [
    (test_df, "vbseq", aa_dict_atchley, 100, True, 5)
])
def test_vGdVAEb(df, column_name, aa_dict_atchley, padding, verbose, expected):
    with torch.no_grad():
        model = vGdVAEb()
        seq = torch.Tensor(peptide_map(df=df,
                                       column_name=column_name,
                                       aa_dict_atchley=aa_dict_atchley,
                                       padding=padding,
                                       verbose=verbose)).to("cpu")
        encoded, _, _, _ = model(seq)
    assert encoded.shape[1] == expected


@pytest.mark.parametrize("df, aa_dict_atchley, mhc_dict, verbose, expected", [
    [test_df, aa_dict_atchley, mhc_dict, True, 30]
])
def test_pMHC(df, aa_dict_atchley, mhc_dict, verbose, expected):
    with torch.no_grad():
        x_p = torch.Tensor(peptide_map(df=df,
                                       column_name="peptide",
                                       aa_dict_atchley=aa_dict_atchley,
                                       padding=30,
                                       verbose=verbose)).to("cpu")
        x_a = torch.Tensor(mhc_map(df=df,
                                   column_name="mhca",
                                   mhc_dict=mhc_dict,
                                   verbose=verbose)).to("cpu")
        x_b = torch.Tensor(mhc_map(df=df,
                                   column_name="mhcb",
                                   mhc_dict=mhc_dict,
                                   verbose=verbose)).to("cpu")

        model = pMHC()
        pmhc_embedding, _ = model(x_p, x_a, x_b)
    assert pmhc_embedding.shape[1] == expected
