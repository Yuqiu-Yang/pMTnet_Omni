import pytest
import pandas as pd
import numpy as np

from pMTnet_Omni.utilities import check_column_names, \
                                  check_data_sanity, \
                                  check_amino_acids, \
                                  get_mhc_class, \
                                  read_file, \
                                  get_auroc, \
                                  plot_roc_curve, \
                                  batchify_check_size, \
                                  partition_dataframe, \
                                  setup_seed


##############################################
# check_column_names
@pytest.mark.parametrize("columns, expected", [
    (
        ["va", "cdr3a", "vaseq", "vb", "cdr3b", "vbseq", "mhca", "mhcb", "peptide", "pMHC_SPECIES", "TCR_SPECIES"],
        ["va", "cdr3a", "vaseq", "vb", "cdr3b", "vbseq", "mhca", "mhcb", "peptide", "pmhc_species", "tcr_species"]
    ),
    (
        ["va", "cdr3a", "vb", "cdr3b", "vbseq", "mhca", "mhcb", "peptide", "pMHC_SPECIES", "TCR_SPECIES"],
        ["va", "cdr3a", "vaseq", "vb", "cdr3b", "vbseq", "mhca", "mhcb", "peptide", "pmhc_species", "tcr_species"]
    ),
    (
        ["cdr3a", "vaseq", "vb", "cdr3b", "vbseq", "mhca", "mhcb", "peptide", "pMHC_SPECIES", "TCR_SPECIES"],
        ["va", "cdr3a", "vaseq", "vb", "cdr3b", "vbseq", "mhca", "mhcb", "peptide", "pmhc_species", "tcr_species"]
    ),
    (
        ["va", "cdr3a", "vaseq", "cdr3b", "vbseq", "mhca", "mhcb", "peptide", "pMHC_SPECIES", "TCR_SPECIES"],
        ["va", "cdr3a", "vaseq", "vb", "cdr3b", "vbseq", "mhca", "mhcb", "peptide", "pmhc_species", "tcr_species"]
    ),
    (
        ["va", "cdr3a", "vaseq", "vb", "cdr3b", "mhca", "mhcb", "peptide", "pMHC_SPECIES", "TCR_SPECIES"],
        ["va", "cdr3a", "vaseq", "vb", "cdr3b", "vbseq", "mhca", "mhcb", "peptide", "pmhc_species", "tcr_species"]
    ),
]
)
def test_check_column_names(columns, expected):
    test_df = pd.read_csv("./tests/test_data/test_df.csv")
    test_df = test_df[columns]

    result_df = check_column_names(df=test_df, 
                                background_tcrs_dir="./tests/test_data/background_tcrs/")
    assert set(result_df.columns) == set(expected)

# check_column_names with errors
@pytest.mark.parametrize("columns", [
    (
        ["va", "vaseq", "vb", "cdr3b", "vbseq", "mhca", "mhcb", "peptide", "pMHC_SPECIES", "TCR_SPECIES"]
    ),
    (
        ["va", "cdr3a", "vb", "cdr3b", "vbseq", "mhcb", "peptide", "pMHC_SPECIES", "TCR_SPECIES"]
    ),
    (
        ["cdr3a", "vb", "cdr3b", "vbseq", "mhca", "mhcb", "peptide", "pMHC_SPECIES", "TCR_SPECIES"]
    ),
    (
        ["va", "cdr3a", "vaseq", "cdr3b", "mhca", "mhcb", "peptide", "pMHC_SPECIES", "TCR_SPECIES"]
    ),
]
)
def test_check_column_names_errors(columns):
    test_df = pd.read_csv("./tests/test_data/test_df.csv")
    test_df = test_df[columns]
    try:
        result_df = check_column_names(df=test_df, 
                                    background_tcrs_dir="./tests/test_data/background_tcrs/")
        assert False
    except:
        assert True

# check_column_names with errors with species
@pytest.mark.parametrize("from_species, to_species", [
    (
        "human", "ape"
    ),
    (
        "mouse", "rat"
    )
]
)
def test_check_column_names_error_species(from_species, to_species):
    test_df = pd.read_csv("./tests/test_data/test_df.csv")
    test_df["TCR_SPECIES"].loc[test_df['TCR_SPECIES'] == from_species] = to_species
    try:
        result_df = check_column_names(df=test_df, 
                                    background_tcrs_dir="./tests/test_data/background_tcrs/")
        assert False
    except:
        assert True

##########################################
# check_data_sanity and check_amino_acids 

@pytest.mark.parametrize("df_column, expected", [
    (pd.DataFrame({"aa": ["READY"]}), pd.DataFrame({"aa": ["READY"]})),
    (pd.DataFrame({"aa": ["LOL"]}), pd.DataFrame({"aa": ["L_L"]})),
])
def test_check_amino_acids(df_column, expected):
    df_result = check_amino_acids(df_column=df_column)
    assert df_result.equals(expected)

@pytest.mark.parametrize("mhca_seq, mhcb_seq, peptide", [
    ("A*03:01", "human_microglobulin", "RINATLETKRINATLETKRINATLETKRINATLETKRINATLETK"),
    ("A*03:01", "human_globulin", "RINATLETK")
]
)
def test_check_data_sanity(mhca_seq, mhcb_seq, peptide):
    test_df = pd.read_csv("./tests/test_data/test_df.csv")
    test_df.at[0, "mhca"] = mhca_seq
    test_df.at[0, "mhcb"] = mhcb_seq
    test_df.at[0, "peptide"] = peptide
    result_df = check_data_sanity(df=test_df,
                                  mhc_path='./tests/test_data/data_for_encoders/valid_mhc.txt')
    assert result_df.shape[0] == 3

#########################################
# get_mhc_class 
@pytest.mark.parametrize("input, expected", [
    (   
        ['A', 'DP', 'H-2-D', 'H-2-IA'],
        ['human class i', 'human class ii', "mouse class i", "mouse class ii"]
    )
])
def test_get_mhc_class(input, expected):
    test_df = pd.read_csv("./tests/test_data/test_df.csv")
    test_df['mhca'] = input
    result_df = get_mhc_class(df=test_df)
    assert all(result_df["mhc_class"] == expected)

@pytest.mark.parametrize("input", [
    (   
        ['Z', 'DP', 'H-2-D', 'H-2-IA']
    )
])
def test_get_mhc_class_error(input):
    test_df = pd.read_csv("./tests/test_data/test_df.csv")
    test_df['mhca'] = input
    try:
        result_df = get_mhc_class(df=test_df)
        assert False 
    except: 
        assert True

###############################
# read_file 
@pytest.mark.parametrize("file_path, sep, header", [
    ("./tests/test_data/test_df.csv", ",", 0),
])
def test_read_file(file_path, sep, header):
    try:
        result_df = read_file(file_path=file_path,
                             sep=sep,
                             header=header,
                             background_tcr_dir="./tests/test_data/background_tcrs/",
                             mhc_path='./tests/test_data/data_for_encoders/valid_mhc.txt')
        assert True
    except:
        assert False

###########################
# get_auroc & plot_roc_curve 
@pytest.mark.parametrize("true_laels, predictions, expected", [
    (np.array([1,0]), np.array([0.5,0.5]), 0.5)
])
def test_get_auroc(true_laels, predictions, expected):
    auroc = get_auroc(true_labels=true_laels, predictions=predictions,ndigits=3)
    assert abs(auroc - expected) < 0.0001

@pytest.mark.parametrize("true_laels, predictions", [
    (np.array([1,0]), np.array([0.5,0.5]))
])
def test_plot_roc_curve(true_laels, predictions):
    try:
        plot_roc_curve(true_labels=true_laels, predictions=predictions, label="")
        assert True
    except:
        assert False

##########################
# batchify_check_size
@pytest.mark.parametrize("check_size, minibatch_size, expected", [
    ([10, 50], 30, ([10, 30, 30], [True, False, True]))
])
def test_batchify_check_size(check_size, minibatch_size, expected):
    batch_size, batch_finished_indicator = batchify_check_size(check_size=check_size,
                                                              minibatch_size=minibatch_size)    
    assert (batch_size == expected[0]) and (batch_finished_indicator == expected[1])                                        

#########################
# partition_dataframe
@pytest.mark.parametrize("df, partition_size, expected",[
    (pd.DataFrame({'a':[1,2,3], 'b':[4,5,6]}), 2, [2, 1])
])
def test_partition_dataframe(df, partition_size, expected):
    partitions = partition_dataframe(df=df, partition_size=partition_size)
    assert (partitions[0].shape[0] == expected[0]) and (partitions[1].shape[0] == expected[1])

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
