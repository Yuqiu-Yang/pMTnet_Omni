# Numeric manipulation
import numpy as np

# PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F


class pMHCTCR(nn.Module):
    def __init__(self,
                 proj_dim_mi: int = 50,
                 feat_dim: int = 1) -> None:
        """The main pMTnet_Omni classifier

        Parameters
        ----------
         
        """
        super(pMHCTCR, self).__init__()
        # self.temperature = temperature
        # Proj for pMHC
        self.Proj = nn.Sequential(
            nn.Linear(100, proj_dim_mi),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(proj_dim_mi, feat_dim)
        )
        # self.Proj1 = nn.Sequential(
        #     nn.Linear(30, proj_pmhc_dim_mi),
        #     nn.ReLU(),
        #     nn.Linear(proj_pmhc_dim_mi, feat_dim)
        # )
        # # Proj for TCR dim_in is 5*2+30*2
        # self.Proj2 = nn.Sequential(
        #     nn.Linear(70, proj_tcr_dim_mi),
        #     nn.ReLU(),
        #     nn.Linear(proj_tcr_dim_mi, feat_dim)
        # )

    def forward(self,
                pmhctcr: torch.tensor) -> torch.tensor:
        """Forward pass of the classifier

        Paramaters
        ---------

        Returns
        ---------
        torch.tensor
            The logit of the binding probability
        
        """
        # Zpmhc = F.normalize(self.Proj1(pmhc))
        # Ztcr = F.normalize(self.Proj2(tcr))
        # logits = torch.div(torch.diagonal(
        #     torch.mm(Zpmhc, Ztcr.T)), self.temperature)
        logits = self.Proj(pmhctcr)
        return logits

    def predict(self,
                pmhctcr: torch.tensor) -> np.ndarray:
        """Predict the binding probability of a given TCR-pMHC pair

        This is basically the forward pass but with no gradient.
        We also convert the result to numpy array.

        Paramaters
        ---------

        Returns
        ---------
        np.ndarray
            The logit of the binding probability
        
        """
        self.eval()
        with torch.no_grad():
            logits = self.Proj(pmhctcr)
            return logits.to("cpu").numpy()
