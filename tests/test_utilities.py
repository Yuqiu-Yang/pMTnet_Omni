import pytest
import pandas as pd
import numpy as np

from pMTnet_Omni.utilities import plot_validation_curve, \
                                    batchify_check_size, \
                                    partition_dataframe, \
                                    setup_seed, \
                                    path_to_dict, \
                                    dict_to_dir, \
                                    check_data_dir_structure

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
