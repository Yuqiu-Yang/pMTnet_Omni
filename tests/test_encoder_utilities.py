import pytest
import pandas as pd
import numpy as np
import csv
import os
from tqdm import tqdm
import torch

from pMTnet_Omni_Document.data_curation import read_file
from pMTnet_Omni.encoders.utilities import esm_mapping,\
    mhc_map,\
    aa_mapping,\
    peptide_map

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
mhc_dict["MAVMAPRTLLLLLSGALALTQTWAGSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQKMEPRAPWIEQEGPEYWDQETRNMKAHSQTDRANLGTLRGYYNQSEDGSHTIQIMYGCDVGPDGRFLRGYRQDAYDGKDYIALNEDLRSWTAADMAAQITKRKWEAVHAAEQRRVYLEGRCVDGLRRYLENGKETLQRTDPPKTHMTHHPISDHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWELSSQPTIPIVGIIAGLVLLGAVITGAVVAAVMWRRKSSDRKGGSYTQAASSDSAQGSDVSLTACKV"] = \
    torch.load(mhc_dict["A*03:01"]).numpy()

###########################
# mhc_map


@pytest.mark.parametrize("mhc, mhc_dict, use_seq, mhcseq", [
    ("A*03:01", mhc_dict, False, ""),
    ("A*01:01", mhc_dict, True, "MAVMAPRTLLLLLSGALALTQTWAGSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQKMEPRAPWIEQEGPEYWDQETRNMKAHSQTDRANLGTLRGYYNQSEDGSHTIQIMYGCDVGPDGRFLRGYRQDAYDGKDYIALNEDLRSWTAADMAAQITKRKWEAVHAAEQRRVYLEGRCVDGLRRYLENGKETLQRTDPPKTHMTHHPISDHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWELSSQPTIPIVGIIAGLVLLGAVITGAVVAAVMWRRKSSDRKGGSYTQAASSDSAQGSDVSLTACKV"),
])
def test_esm_mapping(mhc, mhc_dict, use_seq, mhcseq):
    try:

        result = esm_mapping(mhc=mhc,
                             mhc_dict=mhc_dict,
                             use_seq=use_seq,
                             mhcseq=mhcseq)
        assert True
    except:
        assert False


@pytest.mark.parametrize("column_name, mhc_dict, verbose", [
    ("mhca", mhc_dict, True),
    ("mhcb", mhc_dict, False),
])
def test_mhc_map(column_name, mhc_dict, verbose):
    test_df,_ = read_file("./tests/test_data/test_df.csv",
                        background_tcrs_dir='./tests/test_data/background_tcrs/',
                        mhc_path='./tests/test_data/data_for_encoders/valid_mhc.txt')
    try:
        result = mhc_map(df=test_df, column_name=column_name,
                         mhc_dict=mhc_dict, verbose=verbose)
        assert True
    except:
        assert False

#########################
# peptide_map aa_mapping


@pytest.mark.parametrize("peptide_seq, aa_dict_atchley, padding", [
    ("CALSDRYGGATNKLIF", aa_dict_atchley, 25)
])
def test_aa_mapping(peptide_seq, aa_dict_atchley, padding):
    try:
        result = aa_mapping(peptide_seq=peptide_seq,
                            aa_dict_atchley=aa_dict_atchley, padding=padding)
        assert True
    except:
        assert False


@pytest.mark.parametrize("column_name, aa_dict_atchley, padding, verbose", [
    ("cdr3b", aa_dict_atchley, 25, True),
    ("vaseq", aa_dict_atchley, 100, False),
])
def test_peptide_map(column_name, aa_dict_atchley, padding, verbose):
    test_df = pd.read_csv("./tests/test_data/test_df.csv")
    try:
        result = peptide_map(df=test_df, column_name=column_name,
                             aa_dict_atchley=aa_dict_atchley, padding=padding, verbose=verbose)
        assert True
    except:
        assert False
