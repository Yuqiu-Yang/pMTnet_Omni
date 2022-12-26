# PyTorch modules 
import torch
import torch.nn as nn
import torch.nn.functional as F


class pMHCTCR(nn.Module):
    def __init__(self,\
                 temperature: float=0.1,\
                 proj_pmhc_dim_mi: int=50,\
                 proj_tcr_dim_mi: int=70,\
                 feat_dim: int=70) -> None:
        """The main pMTnet_Omni classifier

        Parameters
        ----------
        temperature: float
            Temperature
        proj_pmhc_dim_mi: int
            Latent dimension for pmhc 
        proj_tcr_dim_mi: int
            Latent dimension for tcr
        feat_dim: int
            Feature dimension 
         
        """
        super(pMHCTCR, self).__init__()
        self.temperature = temperature
        # Proj for pMHC
        self.Proj1 = nn.Sequential(
            nn.Linear(30, proj_pmhc_dim_mi),
            nn.ReLU(),
            nn.Linear(proj_pmhc_dim_mi, feat_dim)
        )
        # Proj for TCR dim_in is 5*2+30*2
        self.Proj2 = nn.Sequential(
            nn.Linear(70, proj_tcr_dim_mi),
            nn.ReLU(),
            nn.Linear(proj_tcr_dim_mi, feat_dim)
        )
    
    def forward(self, tcr, pmhc):
        Zpmhc = F.normalize(self.Proj1(pmhc))
        Ztcr = F.normalize(self.Proj2(tcr))
        logits = torch.div(torch.diagonal(torch.mm(Zpmhc, Ztcr.T)),self.temperature)
        return logits
    
    def predict(self, tcr, pmhc):
        self.eval()
        with torch.no_grad():
            Zpmhc = F.normalize(self.Proj1(pmhc))
            Ztcr = F.normalize(self.Proj2(tcr))
            logits = torch.div(torch.diagonal(torch.mm(Zpmhc, Ztcr.T)),self.temperature)
            return logits.numpy()