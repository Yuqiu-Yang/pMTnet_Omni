import os 
import sys 
from io import StringIO

import pytest 

from pMTnet_Omni import __main__
import pMTnet_Omni


# Version, author list, and help
def test_parser_version():
    screen_stdout = sys.stdout
    string_stdout = StringIO()
    sys.stdout = string_stdout
    
    try:
        __main__.parser.parse_args(["--version"])
    except SystemExit:
        output = string_stdout.getvalue()
        expected = pMTnet_Omni.__version__ + "\n"
        assert output == expected
        sys.stdout = screen_stdout
    else:
        sys.stdout = screen_stdout
        assert False

def test_parser_author():
    screen_stdout = sys.stdout
    string_stdout = StringIO()
    sys.stdout = string_stdout
    
    try:
        __main__.parser.parse_args(["--author"])
    except SystemExit:
        output = string_stdout.getvalue()
        expected = pMTnet_Omni.__author__ + "\n"
        assert output == expected
        sys.stdout = screen_stdout
    else:
        sys.stdout = screen_stdout
        assert False


@pytest.mark.parametrize("cmdarg", ["--help", "-h"])
def test_parser_help(cmdarg):
    screen_stdout = sys.stdout
    string_stdout = StringIO()
    sys.stdout = string_stdout
    
    try:
        __main__.parser.parse_args([cmdarg])
    except SystemExit:
        output = string_stdout.getvalue()
        assert "usage: " in output
        sys.stdout = screen_stdout
    else:
        sys.stdout = screen_stdout
        assert False


@pytest.mark.parametrize("model_device, data_dir, seed, extra_mhc_dict, user_data,\
                        partition_size, compute_percentile_rank,\
                        check_size, load_size, minibatch_size, output_file_path", [
    ("cpu", './tests/test_data', None, './tests/test_data/extra_mhc_dict.json', './tests/test_data/df_curated.csv', 2, False, 10, 5, 5, "./tests/test_data/test_df_result.csv"),
    ("gpu", './tests/test_data', None, './tests/test_data/extra_mhc_dict.json', './tests/test_data/df_curated.csv', 2, True, 10, 5, 5, "./tests/test_data/test_df_result.csv"),
    ("cpu", './tests/test_data', 42, None, './tests/test_data/df_curated.csv', 2, False, 10, 5, 5, "./tests/test_data/test_df_result.csv")
])
def test_pmtnet_omni_main(mocker, model_device, data_dir, seed, extra_mhc_dict, user_data,
                        partition_size, compute_percentile_rank,
                        check_size, load_size, minibatch_size, output_file_path):
    cmdargs: mocker.MagicMock = mocker.MagicMock() 
    cmdargs.model_device = model_device
    cmdargs.data_dir = data_dir
    cmdargs.seed = seed
    cmdargs.extra_mhc_dict = extra_mhc_dict
    cmdargs.user_data = user_data
    cmdargs.partition_size = partition_size
    cmdargs.compute_percentile_rank = compute_percentile_rank
    cmdargs.check_size = check_size
    cmdargs.load_size = load_size
    cmdargs.minibatch_size = minibatch_size
    cmdargs.output_file_path = output_file_path

    try:
        __main__.main(cmdargs=cmdargs)
    except SystemExit:
        assert True 
    except:
        assert False 


@pytest.mark.parametrize("model_device, data_dir, seed, user_data,\
                        partition_size, compute_percentile_rank,\
                        check_size, load_size, minibatch_size,\
                        output_file_path", [
    ("cpu", './tests/test_data', None, None, 2, False, 10, 5, 5, "./tests/test_data/test_df_result.csv"),
    ("cpu", './tests/test_data', None, './tests/test_data/df_curated.csv', 2, False, 10, 5, 5, "./tests/test_data/test_df_result.txt"),
    ("gpu", './tests/test_data', None, './tests/test_data/df_curated.csv', 2, False, 10, 5, 5, None),
    ("cpu", './tests/test_data', 42, './tests/test_data/df_curated.json', 2, False, 10, 5, 5, "./tests/test_data/test_df_result.csv")
])
def test_pmtnet_omni_main_error(mocker, model_device, data_dir, seed, user_data,
                        partition_size, compute_percentile_rank, 
                        check_size, load_size, minibatch_size, output_file_path):
    cmdargs: mocker.MagicMock = mocker.MagicMock() 
    cmdargs.model_device = model_device
    cmdargs.data_dir = data_dir
    cmdargs.seed = seed
    cmdargs.user_data = user_data
    cmdargs.partition_size = partition_size
    cmdargs.compute_percentile_rank = compute_percentile_rank
    cmdargs.check_size = check_size
    cmdargs.load_size = load_size
    cmdargs.minibatch_size = minibatch_size
    cmdargs.output_file_path = output_file_path

    try:
        __main__.main(cmdargs=cmdargs)
    except SystemExit:
        assert False 
    except:
        assert True 