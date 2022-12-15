# Data IO
import pandas as pd 

# Numeric manipulation 
import numpy as np 

# Typing 
from typing import Union

def read_file(file_path: str,
             sep: str=",",
             header: bool=True) -> Union[np.ndarray, pd.DataFrame]:
    """This function reads in .csv or .txt files 

    The function provides a simple wrapper of pd.read_csv 
    and performs some rudementary sanity check

    Parameters
    ----------
    file_path: str
        The location of the file containing TCR-pMHC pair(s)
    sep: str
        Seperation used in the file 
    header: bool
        Whether or not the file contains a header

    Returns
    ------
    pairing_data: np.ndarray
        An numpy ndarray containing curated pairs 

    """
    pairing_data = pd.read_csv(file_path, sep=sep, header=header)

    return pairing_data