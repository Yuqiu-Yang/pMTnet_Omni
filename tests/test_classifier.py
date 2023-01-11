import pytest
import torch
import numpy as np 

from pMTnet_Omni.classifier import pMHCTCR

@pytest.mark.parametrize("pmhctcr", [
    (torch.randn(10, 100))
])
def test_classifier(pmhctcr):
    model = pMHCTCR()
    logits = model(pmhctcr)
    pred = model.predict(pmhctcr)
    assert torch.is_tensor(logits) and isinstance(pred, np.ndarray)