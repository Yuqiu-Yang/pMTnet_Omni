import pytest
import pandas as pd
from pMTnet_Omni.utilities import read_file


@pytest.mark.parametrize("file_path, sep, header", [
    (None, ",", True),
])
def test_read_file(file_path, sep, header):
    f = read_file(file_path=file_path, sep=sep, header=header)
    assert isinstance(f, pd.DataFrame)