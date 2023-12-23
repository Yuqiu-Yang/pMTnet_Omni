# Numeric manipulation
import numpy as np

# PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union

class pMHCTCR(nn.Module):
    def __init__(self,
                 temperature: float=0.1) -> None:
        """The main pMTnet_Omni classifier

        Parameters
        ----------
        temperature: float 
            Temperature in constrastive learning 
        """
        super(pMHCTCR, self).__init__()
        self.temperature = temperature
        
         # Proj for pMHC
        self.Proj1 = nn.Sequential(
            nn.Linear(30, 40),
            nn.ReLU(),
            nn.Linear(40, 50),
            nn.ReLU(),
            nn.Linear(50, 60),
            nn.ReLU(),
            nn.Linear(60, 70)
        )
        
        self.linearTCR = nn.Sequential(
            nn.Linear(70, 70),
            nn.ReLU(),
            nn.Linear(70, 70)
        )
        
        # Proj for TCR dim_in is 5*2+30*2
        self.Proj2 = nn.Sequential(
            nn.Linear(70, 70),
            nn.ReLU(),
            nn.Linear(70, 70)
        )

    def forward(self,
                pmhc_embedding: torch.tensor,
                tcr_embedding: torch.tensor) -> torch.tensor:
        """Forward pass of the classifier

        Parameters
        -----------
        pmhc_embedding: torch.tensor
            The embedding of pmhc 
        tcr_embedding: torch.tensor
            The embedding of tcr
        
        Returns
        ---------
        torch.tensor
            The logit of the binding probability
        
        """
        tcr_embedding = self.linearTCR(tcr_embedding)
        
        Zpmhc = F.normalize(self.Proj1(pmhc_embedding))

        Ztcr = F.normalize(self.Proj2(tcr_embedding))

        logits = torch.div(torch.diagonal(torch.mm(Zpmhc,Ztcr.T)),self.temperature)
        
        return logits

    def predict(self,
                pmhc_embedding: torch.tensor,
                tcr_embedding: torch.tensor) -> np.ndarray:
        """Predict the binding probability of a given TCR-pMHC pair

        This is basically the forward pass but with no gradient.
        We also convert the result to numpy array.

        Parameters
        ------------
        pmhc_embedding: torch.tensor
            The embedding of pmhc 
        tcr_embedding: torch.tensor
            The embedding of tcr
            
        Returns
        ---------
        np.ndarray
            The logit of the binding probability
        
        """
        self.eval()
        with torch.no_grad():
            tcr_embedding = self.linearTCR(tcr_embedding)
        
            Zpmhc = F.normalize(self.Proj1(pmhc_embedding))

            Ztcr = F.normalize(self.Proj2(tcr_embedding))

            logits = torch.div(torch.diagonal(torch.mm(Zpmhc,Ztcr.T)),self.temperature)
            return logits.to("cpu").numpy()


class ensemble_pMHCTCR:
    def __init__(self,
                 model_device: Union[str, torch.device],
                 classifier_path: list) -> None:
        """Initialize the ensemble of classifiers 
        
        Parameters
        ----------
        model_device : Union[str, torch.device]
            Model device to use 
        classifier_path : list
            The path to the classifiers 
        """
        # Set model device 
        if isinstance(model_device, str):
            model_device = torch.device(model_device)
        self.model_device = model_device
        
        self.ensemble_size = len(classifier_path)
        # Initialize each model 
        self.model_list = [pMHCTCR().to(self.model_device) for _ in range(self.ensemble_size)]
        
        print("Attempt to load the ensemble of classifiers\n")
        for counter, classifier_checkpoint_path in enumerate(classifier_path):
            if classifier_checkpoint_path is not None:
                classifier_checkpoint = torch.load(
                    classifier_checkpoint_path, map_location=self.model_device)
                self.model_list[counter].load_state_dict(classifier_checkpoint['net'])
                print("Successfully loaded model " + str(counter) + "\n")
            else:
                print("Check point is not provided. Use random weights\n")
        
    def predict(self,
                pmhc_embedding: torch.tensor,
                tcr_embedding: torch.tensor) -> np.ndarray:
        """Predict affinity score using each model in the ensemble

        Parameters
        ----------
        pmhc_embedding : torch.tensor
            The embedding of pmhc 
        tcr_embedding : torch.tensor
            The embedding of tcr

        Returns
        -------
        np.ndarray
            The logit of the binding probability for each model 
        """
        results = np.zeros((pmhc_embedding.shape[0], self.ensemble_size))
        for i in range(self.ensemble_size):
            logits = self.model_list[i].predict(pmhc_embedding=pmhc_embedding,
                                                tcr_embedding=tcr_embedding)
            results[:, [i]] = logits.reshape(-1,1)
        return results
