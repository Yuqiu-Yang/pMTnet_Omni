import os
import sys
import torch
import argparse

from pMTnet_Omni import __version__, __author__
from pMTnet_Omni.pMTnet_Omni_class import pMTnet_Omni_class

parser = argparse.ArgumentParser(description="pMTnet Omni")

parser.add_argument("--version", action="version",
                    version=__version__, help="Display the version of the software")
parser.add_argument("--author", action="version", version=__author__,
                    help="Check the author list of the algorithm")

# Initialize pMTnet_Omni class
parser.add_argument(
    "--model_device", choices=["cpu", "gpu", "cuda:0"], default="cpu", help="Choose the model device")
parser.add_argument("--data_dir", default="./data",
                    help="The data directory containing data necessary for model setups")
parser.add_argument("--seed", type=int, nargs="?",
                    help="Seed for reproducibility")

# User input
parser.add_argument("--user_data_path", help="The path the user data file")
parser.add_argument("--partition_size", type=int,
                    default=7000, help="Size for each partition")

# Prediction
parser.add_argument("--rank_threshold", type=float, default=0.03,
                    help="The rank percentile threshold greater than which further varification is NOT conducted")
parser.add_argument("--not_load_embedding", action="store_true",
                    help="NOT load embeddings of the background tcrs. THIS WILL BE SLOOOOOW")
parser.add_argument("--check_size", nargs='+', type=int, default=[
                    1000, 10000, 100000], help="The numbers of background tcrs to be used to check against the user results")
parser.add_argument("--load_size", type=int, default=1000000,
                    help="The number of background tcrs to be loaded by the dataloaders")
parser.add_argument("--minibatch_size", type=int, default=50000,
                    help="The number of background tcrs to be sampled each time a dataloader is called")
parser.add_argument("--log_file_path",
                    help="The file path to save the output log file")

# User output
parser.add_argument("--output_file_path",
                    help="The file path to the output file")

# Maybe consider
# config.yaml


def main(cmdargs: argparse.Namespace):
    """The main method for pMTnet Omni

    Parameters:
    ----------
    cmdargs: argparse.Namespace
        The command line argments and flags 
    """
    # Set model device
    if cmdargs.model_device == "cpu":
        model_device = "cpu"
    else:
        model_device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using "+str(model_device)+"\n")

    user_data_path = cmdargs.user_data_path
    if user_data_path is None:
        raise Exception("User data path NOT provided")
    
    file_extention = os.path.splitext(user_data_path)[1]
    if file_extention == ".txt":
        sep = "\t"
    elif file_extention == ".csv":
        sep = ","
    else:
        raise Exception("Only .txt and .csv are accepted.")

    if cmdargs.output_file_path is None:
        raise Exception("Output file path NOT provided")

    if os.path.splitext(cmdargs.output_file_path)[1] != ".csv":
        raise Exception(
            "Currently, only .csv files can be used as output files")

    # Initialize pmtnet
    pmtnet = pMTnet_Omni_class(model_device=model_device,
                               data_dir=cmdargs.data_dir,
                               seed=cmdargs.seed)

    # Read user data file
    pmtnet.read_user_df(user_data_path=user_data_path,
                        sep=sep,
                        header=0,
                        partition_size=cmdargs.partition_size)

    # Encode user file
    pmtnet.encode_user_df()
    # Get model raw output
    pmtnet.get_user_df_output()

    # Prediction using rank against background
    load_embedding = not cmdargs.not_load_embedding

    pmtnet.predict(rank_threshold=cmdargs.rank_threshold,
                   load_embedding=load_embedding,
                   replacement=True,
                   check_size=list(cmdargs.check_size),
                   load_size=cmdargs.load_size,
                   minibatch_size=cmdargs.minibatch_size,
                   log_file_path=cmdargs.log_file_path)

    # Amalagamate all the predictions
    df_dict, _, _ = pmtnet.amalgamate()
    df_dict['complete'].to_csv(cmdargs.output_file_path,
                               sep=",",
                               header=True,
                               index=False)
    sys.exit(0)


if __name__ == "__main__":
    cmdargs = parser.parse_args()
    main(cmdargs=cmdargs)
