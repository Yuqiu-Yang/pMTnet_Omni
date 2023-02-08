import pytest
import pandas as pd
import numpy as np

from pMTnet_Omni.utilities import check_column_names, \
    get_mhc_class, \
    check_data_sanity, \
    check_amino_acids, \
    read_file, \
    plot_validation_curve, \
    batchify_check_size, \
    partition_dataframe, \
    setup_seed, \
    path_to_dict, \
    dict_to_dir, \
    check_data_dir_structure


##############################################
# check_column_names
@pytest.mark.parametrize("columns, expected", [
    (
        ["va", "cdr3a", "vaseq", "vb", "cdr3b", "vbseq", "mhc",
            "peptide", "pMHC_SPECIES", "TCR_SPECIES"],
        ["va", "cdr3a", "vaseq", "vb", "cdr3b", "vbseq", "mhc",
            "mhcseq", "peptide", "pmhc_species", "tcr_species"]
    ),
    (
        ["va", "cdr3a", "vb", "cdr3b", "vbseq", "mhc",
            "peptide", "pMHC_SPECIES", "TCR_SPECIES"],
        ["va", "cdr3a", "vaseq", "vb", "cdr3b", "vbseq", "mhc",
            "mhcseq", "peptide", "pmhc_species", "tcr_species"]
    ),
    (
        ["cdr3a", "vaseq", "vb", "cdr3b", "vbseq", "mhc",
            "peptide", "pMHC_SPECIES", "TCR_SPECIES"],
        ["va", "cdr3a", "vaseq", "vb", "cdr3b", "vbseq", "mhc",
            "mhcseq", "peptide", "pmhc_species", "tcr_species"]
    ),
    (
        ["va", "cdr3a", "vaseq", "cdr3b", "vbseq", "mhc",
            "peptide", "pMHC_SPECIES", "TCR_SPECIES"],
        ["va", "cdr3a", "vaseq", "vb", "cdr3b", "vbseq", "mhc",
            "mhcseq", "peptide", "pmhc_species", "tcr_species"]
    ),
    (
        ["va", "cdr3a", "vaseq", "vb", "cdr3b", "mhc",
            "peptide", "pMHC_SPECIES", "TCR_SPECIES"],
        ["va", "cdr3a", "vaseq", "vb", "cdr3b", "vbseq", "mhc",
            "mhcseq", "peptide", "pmhc_species", "tcr_species"]
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
        ["va", "vaseq", "vb", "cdr3b", "vbseq", "mhc",
         "peptide", "pMHC_SPECIES", "TCR_SPECIES"]
    ),
    (
        ["cdr3a", "vb", "cdr3b", "vbseq", "mhc",
            "peptide", "pMHC_SPECIES", "TCR_SPECIES"]
    ),
    (
        ["va", "cdr3a", "vaseq", "cdr3b", "mhc",
            "peptide", "pMHC_SPECIES", "TCR_SPECIES"]
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
    test_df["TCR_SPECIES"].loc[test_df['TCR_SPECIES']
                               == from_species] = to_species
    try:
        result_df = check_column_names(df=test_df,
                                       background_tcrs_dir="./tests/test_data/background_tcrs/")
        assert False
    except:
        assert True

#########################################
# get_mhc_class


@pytest.mark.parametrize("input, expected", [
    (
        ['A', 'DPA/DPB', ' H-2 - D', 'H2- IA'],
        ['human class i', 'human class ii',
         "mouse class i", "mouse class ii"]
    ),
    (
        ['HLAA', 'HLA-DR', ' IA', 'H2- IA'],
        ['human class i', 'human class ii',
         "mouse class ii",  "mouse class ii"]
    )
])
def test_get_mhc_class(input, expected):
    test_df = pd.read_csv("./tests/test_data/test_df.csv")
    test_df['mhc'] = input
    df = check_column_names(test_df,
                            background_tcrs_dir="./tests/test_data/background_tcrs/")
    result_df = get_mhc_class(df=df)
    assert all(result_df["mhc_class"] == expected)


@pytest.mark.parametrize("input", [
    (
        ['Z', 'DO/DO', 'H-2-A', 'H2-IB']
    )
])
def test_get_mhc_class_error(input):
    test_df = pd.read_csv("./tests/test_data/test_df.csv")
    df = check_column_names(test_df,
                            background_tcrs_dir="./tests/test_data/background_tcrs/")
    df['mhc'] = input
    try:
        result_df = get_mhc_class(df=df)
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


@pytest.mark.parametrize("mhc, peptide", [
    ("A*03:01",
     "RINATLETKRINATLETKRINATLETKRINATLETKRINATLETK"),
    ("A*77:77", "RINATLETK")
]
)
def test_check_data_sanity(mhc, peptide):
    test_df = pd.read_csv("./tests/test_data/test_df.csv")
    test_df.at[0, "mhc"] = mhc
    test_df.at[0, "peptide"] = peptide
    df = check_column_names(test_df,
                            background_tcrs_dir="./tests/test_data/background_tcrs/")
    mhc_class = get_mhc_class(df=test_df)
    df = pd.concat([df, mhc_class], axis=1, ignore_index=False)
    result_df = check_data_sanity(df=df,
                                  mhc_path='./tests/test_data/data_for_encoders/valid_mhc.txt')
    assert result_df.shape[0] == 3

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
# plot_validation_curve


@pytest.mark.parametrize("result_dicts, plot_type, primary_type, secondary_types, title, figure_save_path", [
    ({"complete": {"overall": {"fpr": np.array([0.2, 0.5]),
                               "tpr": np.array([0.5, 0.7]),
                               "precision": np.array([0.8, 0.4]),
                               "recall": np.array([0.3, 0.7]),
                               "auroc": 0.5,
                               "auprc": 0.7},
                   "human overall": {"fpr": np.array([0.2, 0.5]),
                                     "tpr": np.array([0.5, 0.7]),
                                     "precision": np.array([0.8, 0.4]),
                                     "recall": np.array([0.3, 0.7]),
                                     "auroc": 0.5,
                                     "auprc": 0.7},
                   "mouse overall": {"fpr": np.array([0.2, 0.5]),
                                     "tpr": np.array([0.5, 0.7]),
                                     "precision": np.array([0.8, 0.4]),
                                     "recall": np.array([0.3, 0.7]),
                                     "auroc": 0.5,
                                     "auprc": 0.7},
                   "human class i": {"fpr": np.array([0.2, 0.5]),
                                     "tpr": np.array([0.5, 0.7]),
                                     "precision": np.array([0.8, 0.4]),
                                     "recall": np.array([0.3, 0.7]),
                                     "auroc": 0.5,
                                     "auprc": 0.7},
                   "human class ii": {"fpr": np.array([0.2, 0.5]),
                                      "tpr": np.array([0.5, 0.7]),
                                      "precision": np.array([0.8, 0.4]),
                                      "recall": np.array([0.3, 0.7]),
                                      "auroc": 0.5,
                                      "auprc": 0.7},
                   "mouse class i": {"fpr": np.array([0.2, 0.5]),
                                     "tpr": np.array([0.5, 0.7]),
                                     "precision": np.array([0.8, 0.4]),
                                     "recall": np.array([0.3, 0.7]),
                                     "auroc": 0.5,
                                     "auprc": 0.7},
                   "mouse class ii": {"fpr": np.array([0.2, 0.5]),
                                      "tpr": np.array([0.5, 0.7]),
                                      "precision": np.array([0.8, 0.4]),
                                      "recall": np.array([0.3, 0.7]),
                                      "auroc": 0.5,
                                      "auprc": 0.7}},
      "alpha missing": None,
      "beta missing": None}, "roc", "complete", "overall", None, "./tests/test_data/test_plot.png"),
    ({"complete": {"overall": {"fpr": np.array([0.2, 0.5]),
                               "tpr": np.array([0.5, 0.7]),
                               "precision": np.array([0.8, 0.4]),
                               "recall": np.array([0.3, 0.7]),
                               "auroc": 0.5,
                               "auprc": 0.7},
                   "human overall": {"fpr": np.array([0.2, 0.5]),
                                     "tpr": np.array([0.5, 0.7]),
                                     "precision": np.array([0.8, 0.4]),
                                     "recall": np.array([0.3, 0.7]),
                                     "auroc": 0.5,
                                     "auprc": 0.7},
                   "mouse overall": {"fpr": np.array([0.2, 0.5]),
                                     "tpr": np.array([0.5, 0.7]),
                                     "precision": np.array([0.8, 0.4]),
                                     "recall": np.array([0.3, 0.7]),
                                     "auroc": 0.5,
                                     "auprc": 0.7},
                   "human class i": {"fpr": np.array([0.2, 0.5]),
                                     "tpr": np.array([0.5, 0.7]),
                                     "precision": np.array([0.8, 0.4]),
                                     "recall": np.array([0.3, 0.7]),
                                     "auroc": 0.5,
                                     "auprc": 0.7},
                   "human class ii": {"fpr": np.array([0.2, 0.5]),
                                      "tpr": np.array([0.5, 0.7]),
                                      "precision": np.array([0.8, 0.4]),
                                      "recall": np.array([0.3, 0.7]),
                                      "auroc": 0.5,
                                      "auprc": 0.7},
                   "mouse class i": {"fpr": np.array([0.2, 0.5]),
                                     "tpr": np.array([0.5, 0.7]),
                                     "precision": np.array([0.8, 0.4]),
                                     "recall": np.array([0.3, 0.7]),
                                     "auroc": 0.5,
                                     "auprc": 0.7},
                   "mouse class ii": {"fpr": np.array([0.2, 0.5]),
                                      "tpr": np.array([0.5, 0.7]),
                                      "precision": np.array([0.8, 0.4]),
                                      "recall": np.array([0.3, 0.7]),
                                      "auroc": 0.5,
                                      "auprc": 0.7}},
      "alpha missing": None,
      "beta missing": None}, "prc", "overall", None, "Trial title", None),
    ({"complete": {"overall": {"fpr": np.array([0.2, 0.5]),
                               "tpr": np.array([0.5, 0.7]),
                               "precision": np.array([0.8, 0.4]),
                               "recall": np.array([0.3, 0.7]),
                               "auroc": 0.5,
                               "auprc": 0.7},
                   "human overall": {"fpr": np.array([0.2, 0.5]),
                                     "tpr": np.array([0.5, 0.7]),
                                     "precision": np.array([0.8, 0.4]),
                                     "recall": np.array([0.3, 0.7]),
                                     "auroc": 0.5,
                                     "auprc": 0.7},
                   "mouse overall": {"fpr": np.array([0.2, 0.5]),
                                     "tpr": np.array([0.5, 0.7]),
                                     "precision": np.array([0.8, 0.4]),
                                     "recall": np.array([0.3, 0.7]),
                                     "auroc": 0.5,
                                     "auprc": 0.7},
                   "human class i": {"fpr": np.array([0.2, 0.5]),
                                     "tpr": np.array([0.5, 0.7]),
                                     "precision": np.array([0.8, 0.4]),
                                     "recall": np.array([0.3, 0.7]),
                                     "auroc": 0.5,
                                     "auprc": 0.7},
                   "human class ii": {"fpr": np.array([0.2, 0.5]),
                                      "tpr": np.array([0.5, 0.7]),
                                      "precision": np.array([0.8, 0.4]),
                                      "recall": np.array([0.3, 0.7]),
                                      "auroc": 0.5,
                                      "auprc": 0.7},
                   "mouse class i": {"fpr": np.array([0.2, 0.5]),
                                     "tpr": np.array([0.5, 0.7]),
                                     "precision": np.array([0.8, 0.4]),
                                     "recall": np.array([0.3, 0.7]),
                                     "auroc": 0.5,
                                     "auprc": 0.7},
                   "mouse class ii": {"fpr": np.array([0.2, 0.5]),
                                      "tpr": np.array([0.5, 0.7]),
                                      "precision": np.array([0.8, 0.4]),
                                      "recall": np.array([0.3, 0.7]),
                                      "auroc": 0.5,
                                      "auprc": 0.7}},
      "alpha missing": None,
      "beta missing": None}, "roc", "complete", None, None, "./tests/test_data/test_plot.png")
])
def test_plot_validation_curve(result_dicts, plot_type, primary_type, secondary_types, title, figure_save_path):
    try:
        plot_validation_curve(result_dicts=result_dicts,
                              plot_type=plot_type,
                              primary_type=primary_type,
                              secondary_types=secondary_types,
                              title=title,
                              figure_save_path=figure_save_path)
        assert True
    except:
        assert False

###################
# plot_validation_curve_errors


@pytest.mark.parametrize("result_dicts, plot_type, primary_type, secondary_types, title, figure_save_path", [
    ({"complete": {"overall": {"fpr": np.array([0.2, 0.5]),
                               "tpr": np.array([0.5, 0.7]),
                               "precision": np.array([0.8, 0.4]),
                               "recall": np.array([0.3, 0.7]),
                               "auroc": 0.5,
                               "auprc": 0.7},
                   "human overall": {"fpr": np.array([0.2, 0.5]),
                                     "tpr": np.array([0.5, 0.7]),
                                     "precision": np.array([0.8, 0.4]),
                                     "recall": np.array([0.3, 0.7]),
                                     "auroc": 0.5,
                                     "auprc": 0.7},
                   "mouse overall": {"fpr": np.array([0.2, 0.5]),
                                     "tpr": np.array([0.5, 0.7]),
                                     "precision": np.array([0.8, 0.4]),
                                     "recall": np.array([0.3, 0.7]),
                                     "auroc": 0.5,
                                     "auprc": 0.7},
                   "human class i": {"fpr": np.array([0.2, 0.5]),
                                     "tpr": np.array([0.5, 0.7]),
                                     "precision": np.array([0.8, 0.4]),
                                     "recall": np.array([0.3, 0.7]),
                                     "auroc": 0.5,
                                     "auprc": 0.7},
                   "human class ii": {"fpr": np.array([0.2, 0.5]),
                                      "tpr": np.array([0.5, 0.7]),
                                      "precision": np.array([0.8, 0.4]),
                                      "recall": np.array([0.3, 0.7]),
                                      "auroc": 0.5,
                                      "auprc": 0.7},
                   "mouse class i": {"fpr": np.array([0.2, 0.5]),
                                     "tpr": np.array([0.5, 0.7]),
                                     "precision": np.array([0.8, 0.4]),
                                     "recall": np.array([0.3, 0.7]),
                                     "auroc": 0.5,
                                     "auprc": 0.7},
                   "mouse class ii": {"fpr": np.array([0.2, 0.5]),
                                      "tpr": np.array([0.5, 0.7]),
                                      "precision": np.array([0.8, 0.4]),
                                      "recall": np.array([0.3, 0.7]),
                                      "auroc": 0.5,
                                      "auprc": 0.7}},
      "alpha missing": None,
      "beta missing": None}, "precision-recall curve", "complete", "overall", None, None),
    ({"complete": {"overall": {"fpr": np.array([0.2, 0.5]),
                               "tpr": np.array([0.5, 0.7]),
                               "precision": np.array([0.8, 0.4]),
                               "recall": np.array([0.3, 0.7]),
                               "auroc": 0.5,
                               "auprc": 0.7},
                   "human overall": {"fpr": np.array([0.2, 0.5]),
                                     "tpr": np.array([0.5, 0.7]),
                                     "precision": np.array([0.8, 0.4]),
                                     "recall": np.array([0.3, 0.7]),
                                     "auroc": 0.5,
                                     "auprc": 0.7},
                   "mouse overall": {"fpr": np.array([0.2, 0.5]),
                                     "tpr": np.array([0.5, 0.7]),
                                     "precision": np.array([0.8, 0.4]),
                                     "recall": np.array([0.3, 0.7]),
                                     "auroc": 0.5,
                                     "auprc": 0.7},
                   "human class i": {"fpr": np.array([0.2, 0.5]),
                                     "tpr": np.array([0.5, 0.7]),
                                     "precision": np.array([0.8, 0.4]),
                                     "recall": np.array([0.3, 0.7]),
                                     "auroc": 0.5,
                                     "auprc": 0.7},
                   "human class ii": {"fpr": np.array([0.2, 0.5]),
                                      "tpr": np.array([0.5, 0.7]),
                                      "precision": np.array([0.8, 0.4]),
                                      "recall": np.array([0.3, 0.7]),
                                      "auroc": 0.5,
                                      "auprc": 0.7},
                   "mouse class i": {"fpr": np.array([0.2, 0.5]),
                                     "tpr": np.array([0.5, 0.7]),
                                     "precision": np.array([0.8, 0.4]),
                                     "recall": np.array([0.3, 0.7]),
                                     "auroc": 0.5,
                                     "auprc": 0.7},
                   "mouse class ii": {"fpr": np.array([0.2, 0.5]),
                                      "tpr": np.array([0.5, 0.7]),
                                      "precision": np.array([0.8, 0.4]),
                                      "recall": np.array([0.3, 0.7]),
                                      "auroc": 0.5,
                                      "auprc": 0.7}},
      "alpha missing": None,
      "beta missing": None}, "prc", "alpha missing", None, None, None),
    ({"complete": {"overall": {"fpr": np.array([0.2, 0.5]),
                               "tpr": np.array([0.5, 0.7]),
                               "precision": np.array([0.8, 0.4]),
                               "recall": np.array([0.3, 0.7]),
                               "auroc": 0.5,
                               "auprc": 0.7},
                   "human overall": {"fpr": np.array([0.2, 0.5]),
                                     "tpr": np.array([0.5, 0.7]),
                                     "precision": np.array([0.8, 0.4]),
                                     "recall": np.array([0.3, 0.7]),
                                     "auroc": 0.5,
                                     "auprc": 0.7},
                   "mouse overall": {"fpr": np.array([0.2, 0.5]),
                                     "tpr": np.array([0.5, 0.7]),
                                     "precision": np.array([0.8, 0.4]),
                                     "recall": np.array([0.3, 0.7]),
                                     "auroc": 0.5,
                                     "auprc": 0.7},
                   "human class i": {"fpr": np.array([0.2, 0.5]),
                                     "tpr": np.array([0.5, 0.7]),
                                     "precision": np.array([0.8, 0.4]),
                                     "recall": np.array([0.3, 0.7]),
                                     "auroc": 0.5,
                                     "auprc": 0.7},
                   "human class ii": {"fpr": np.array([0.2, 0.5]),
                                      "tpr": np.array([0.5, 0.7]),
                                      "precision": np.array([0.8, 0.4]),
                                      "recall": np.array([0.3, 0.7]),
                                      "auroc": 0.5,
                                      "auprc": 0.7},
                   "mouse class i": {"fpr": np.array([0.2, 0.5]),
                                     "tpr": np.array([0.5, 0.7]),
                                     "precision": np.array([0.8, 0.4]),
                                     "recall": np.array([0.3, 0.7]),
                                     "auroc": 0.5,
                                     "auprc": 0.7},
                   "mouse class ii": {"fpr": np.array([0.2, 0.5]),
                                      "tpr": np.array([0.5, 0.7]),
                                      "precision": np.array([0.8, 0.4]),
                                      "recall": np.array([0.3, 0.7]),
                                      "auroc": 0.5,
                                      "auprc": 0.7}},
      "alpha missing": None,
      "beta missing": None}, "roc", "both missing", None, None, None)
])
def test_plot_validation_curve_errors(result_dicts, plot_type, primary_type, secondary_types, title, figure_save_path):
    try:
        plot_validation_curve(result_dicts=result_dicts,
                              plot_type=plot_type,
                              primary_type=primary_type,
                              secondary_types=secondary_types,
                              title=title,
                              figure_save_path=figure_save_path)
        assert False
    except:
        assert True


##########################
# batchify_check_size


@pytest.mark.parametrize("check_size, minibatch_size, expected", [
    ([10, 50], 30, ([10, 30, 30], [True, False, True]))
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
    encoder_path, classifier_path = check_data_dir_structure(path=path)
    assert all([x is None for x in encoder_path]) and (classifier_path is None)


@pytest.mark.parametrize("path", [
    ("./tests/")
])
def test_check_data_dir_structure_error(path):
    try:
        encoder_path, classifier_path = check_data_dir_structure(path=path)
        assert False
    except:
        assert True
