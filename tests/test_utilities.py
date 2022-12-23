import pytest
import pandas as pd



@pytest.mark.parametrize("file_path, sep, header", [
    (None, ",", True),
])
def test_read_file(file_path, sep, header):
    assert isinstance(1, int)