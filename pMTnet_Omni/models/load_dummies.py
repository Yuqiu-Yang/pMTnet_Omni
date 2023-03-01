import numpy as np 
# PyTorch module
import torch
import torch.nn as nn

class vGdVAEa(nn.Module):
    def __init__(self):
        super().__init__()
        self.mapping = nn.Linear(100, 5)

    def forward(self, x):
        x.squeeze_(dim=1)
        x=torch.sum(x, dim=2)
        encoded=self.mapping(x)
        return encoded, encoded, encoded, encoded

class vGdVAEb(nn.Module):
    def __init__(self):
        super().__init__()
        self.mapping = nn.Linear(100, 5)

    def forward(self, x):
        x.squeeze_(dim=1)
        x=torch.sum(x, dim=2)
        encoded=self.mapping(x)
        return encoded, encoded, encoded, encoded

class cdr3VAEa(nn.Module):
    def __init__(self):
        super().__init__()
        self.mapping = nn.Linear(25, 30)

    def forward(self, x):
        x.squeeze_(dim=1)
        x=torch.sum(x, dim=2)
        encoded=self.mapping(x)
        return encoded, encoded, encoded, encoded

class cdr3VAEb(nn.Module):
    def __init__(self):
        super().__init__()
        self.mapping = nn.Linear(25, 30)

    def forward(self, x):
        x.squeeze_(dim=1)
        x=torch.sum(x, dim=2)
        encoded=self.mapping(x)
        return encoded, encoded, encoded, encoded

class pMHC(nn.Module):
    def __init__(self):
        super().__init__()
        self.mapping = nn.Linear(790, 30)

    def forward(self, x_p, x_a, x_b):
        x_p.squeeze_(dim=1)
        x_p=torch.sum(x_p, dim=2)
        x_a.squeeze_(dim=1)
        x_a=torch.sum(x_a, dim=2)
        x_b.squeeze_(dim=1)
        x_b=torch.sum(x_b, dim=2)
        
        x=torch.cat((x_p, x_a, x_b), dim=1)
        encoded_act=self.mapping(x)
        
        return encoded_act, encoded_act


class pMHCTCR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mapping = nn.Linear(100, 1)

    def forward(self,
                pmhctcr: torch.tensor) -> torch.tensor:
        
        logits = self.mapping(pmhctcr)
        return logits

    def predict(self,
                pmhctcr: torch.tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits = self.mapping(pmhctcr)
            return logits.to("cpu").numpy()


class ensemble_pMHCTCR:
    def __init__(self,
                 model_device=None,
                 classifier_path=[None]) -> None:
        
        # Initialize each model 
        self.model_list = [pMHCTCR() for _ in range(1)]
        
    def predict(self,
                pmhc_embedding: torch.tensor,
                tcr_embedding: torch.tensor) -> np.ndarray:
        results = np.zeros((pmhc_embedding.shape[0], 1))
        for i in range(1):
            logits = self.model_list[i].predict(pmhc_embedding=pmhc_embedding,
                                                tcr_embedding=tcr_embedding)
            results[:, i] = logits
        return results
