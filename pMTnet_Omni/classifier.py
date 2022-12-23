import torch
import torch.nn as nn
import torch.nn.functional as F

# input dataset, like vcdr3pmhc
# output Zpmhc * Ztcr
class pMHCTCR(nn.Module):
    def __init__(self, temperature=0.1,\
                 proj_pmhc_dim_mi=50,\
                 proj_tcr_dim_mi=70,\
                 feat_dim=70):
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