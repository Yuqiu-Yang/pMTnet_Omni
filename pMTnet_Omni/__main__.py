import os 
import sys
import argparse

from pMTnet_Omni import __version__, __author__

parser = argparse.ArgumentParser(description="pMTnet Omni")

parser.add_argument("--version", action="version", version=__version__, help="Display the version of the software")
parser.add_argument("--author", action="version", version=__author__, help="Check the author list of the algorithm")

def main(cmdargs: argparse.Namespace):
    """The main method for pMTnet Omni

    Parameters:
    ----------
    cmdargs: argparse.Namespace
        The command line argments and flags 
    """

    print("Well nothing much here yet XD")

    sys.exit(0)


if __name__ == "__main__":
    cmdargs = parser.parse_args()
    main(cmdargs=cmdargs)