# Data IO 
import pandas as pd

# Numeric manipulation
import numpy as np 

# PyTorch
import torch 

# Utilities functions 
from pMTnet_Omni.utilities import partition_dataframe, coroutine

# Typing
from typing import Optional, Union


class pmhc_species_tcr_class:
    def __init__(self,
                 pmhc_id: int,
                 tcr_ids: list,
                 pmhc_species: str,
                 tcr_species: str,
                 ensemble_size: int) -> None:
        """The most basic pmhc-tcr class
        
        Since regardless of the species of the pMHC, there is a possibility that 
        there will be TCRs from multiple species paired with it, this class holds 
        TCRs from one of those species. 

        Parameters
        ----------
        pmhc_id : int
            The unique id assigned to the pMHC
        tcr_ids : list
            A list of unique ids assigned to the paired TCRs of the chosen species 
        pmhc_species : str
            The species of the pMHC 
        tcr_species : str
            The species of the TCRs  
        ensemble_size : int
            Number of classifiers in the ensemble of the pMTnet Omni model 
        """
        # Store basic info 
        self.pmhc_id = pmhc_id
        self.pmhc_species = pmhc_species
        self.tcr_ids = tcr_ids
        self.tcr_species = tcr_species
        self.ensemble_size = ensemble_size
        # Compute the number of tcrs 
        self.n_tcrs = len(self.tcr_ids)
        # The prediction of pMTnet Omni consists of two steps 
        # One: Compute the raw logits and we can also compute the internal ranks 
        # logits will be the average over the ensemble 
        self.endo_df = pd.DataFrame(np.zeros((self.n_tcrs, ensemble_size + 2)),
                                    columns=["logits_" + str(m) for m in range(ensemble_size)] + \
                                            ["logits", "endo_ranks"],
                                    index=self.tcr_ids)
        # self.logits_by_model = np.full((self.n_tcrs, ensemble_size), np.inf)
        # self.logits = np.full((self.n_tcrs, 1), np.inf)
        # self.endo_ranks = np.zeros((self.n_tcrs, 1))
        # Two: Compute the ranks against some background tcrs 
        # We will compute the percentile by model first 
        # Percentile rank will be the average over the ensemble 
        self.exo_df = pd.DataFrame(np.zeros((self.n_tcrs, ensemble_size * 2 + 1)),
                                    columns=["ranks_" + str(m) for m in range(ensemble_size)] + \
                                            ["percentile_ranks_" + str(m) for m in range(ensemble_size)] + \
                                            ["percentile_ranks"],
                                    index=self.tcr_ids)
        # self.exo_ranks_by_model = np.zeros((self.n_tcrs, ensemble_size))
        # self.exo_percentile_ranks_by_model = np.zeros((self.n_tcrs, ensemble_size))
        # self.exo_percentile_ranks = np.zeros((self.n_tcrs, 1))
    
    def get_pmhc_tcr(self,
                     background_tcr_embedding: Optional[torch.tensor]=None) -> pd.DataFrame:
        """Generate a dataframe with pairing pmhc ids and tcr ids 

        There are two situations under which the function will be called. 
        When computing raw logits, we need a dataframe pairing the pmhc id with 
        all the tcrs 
        When validating against background, we only need the pmhc id. However, 
        for the sake of coherence, the tcr "ids" will be 0,..., number of background
        
        Parameters
        ----------
        background_tcr_embedding : Optional[torch.tensor], optional
            Embedding tensor of a batch of background tcrs, by default None

        Returns
        -------
        pd.DataFrame
            A dataframe with pairing pmhc ids and tcr ids 
        """
        if background_tcr_embedding is None:
            # If background is not provided, we know this is internal 
            n_rows = self.n_tcrs
            tcrs = self.tcr_ids
        else:
            # When validating against background 
            n_rows = background_tcr_embedding.shape[0]
            tcrs = [i for i in range(n_rows)]
        df = pd.DataFrame({'pmhc': [self.pmhc_id] * n_rows,
                           'tcr': tcrs}, dtype=int)
        return df
    
    def update_endo_ranks(self, logits: np.ndarray) -> None:
        """Store raw logits and compute internal ranks 

        Parameters
        ----------
        logits : np.ndarray
            logits computed by the ensemble 
        """
        logits_col = ["logits_" + str(m) for m in range(self.ensemble_size)]
        self.endo_df[logits_col] = logits
        # logits will be the average of the ensemble 
        self.endo_df['logits'] = self.endo_df[logits_col].mean(axis=1)
        temp = self.endo_df['logits'].argsort()[::-1]
        self.endo_df.loc[temp, 'endo_ranks'] = np.arange(self.n_tcrs)
        
        # self.logits_by_model = deepcopy(logits)
        # self.logits = np.mean(self.logits_by_model, axis=1).reshape(-1,1)
        
        # temp = self.logits[:,0].argsort()[::-1]
        # self.endo_ranks[temp,0] = np.arange(self.n_tcrs)
    
    def update_exo_ranks(self, 
                         logits: np.ndarray) -> None:
        """Update external ranks 
        
        Unlike the computation of internal ranks, when validating against background,
        we often do not have all the data. They arrive in mini-batches. 

        Parameters
        ----------
        logits : np.ndarray
            logits computed by the ensemble 
        """
        for m in range(self.ensemble_size):
            for tcr in self.tcr_ids:
                self.exo_df.at[tcr, "ranks_"+str(m)] += sum(logits[:, m] > self.endo_df.at[tcr, "logits_"+str(m)])
        
        # for model_id in range(self.ensemble_size):
        #     for i in range(self.n_tcrs):
        #         self.exo_ranks_by_model[i, model_id] += sum(logits[:, model_id] > self.logits_by_model[i, model_id])   
    
    def compute_percentile_ranks(self,
                                 background_size: int) -> None:
        """Compute percentile ranks 

        Parameters
        ----------
        background_size : int
            Total number of background tcrs 
        """
        # When computing the final percentile ranks, we include 
        # results from the background tcrs as well as 
        # the user given pairs 
        # This way, the percentile rankings would be consistent with the logits 
        percentile_col = ["percentile_ranks_" + str(m) for m in range(self.ensemble_size)]
        rank_col = ["ranks_" + str(m) for m in range(self.ensemble_size)]
        self.exo_df[percentile_col] = (self.exo_df[rank_col].values + np.tile(self.endo_df[['endo_ranks']].values, (1, self.ensemble_size)))/(background_size + self.n_tcrs)
        self.exo_df['percentile_ranks'] = self.exo_df[percentile_col].mean(axis=1)
        
        # When computing the final percentile ranks, we include 
        # results from the background tcrs as well as 
        # the user given pairs 
        # This way, the percentile rankings would be consistent with the logits  
        # self.exo_percentile_ranks_by_model = (self.exo_ranks_by_model+np.tile(self.endo_ranks, (1, self.ensemble_size)))/(background_size + self.n_tcrs)
        # self.exo_percentile_ranks = np.mean(self.exo_percentile_ranks_by_model, axis=1)
    
    def reset_exo_ranks(self) -> None:
        """Resets all external results 
        
        This is used when the validation has passed one check size 
        and is moving on to the next one 
        """
        self.exo_df.iloc[:] = 0
        # self.exo_ranks_by_model = np.zeros((self.n_tcrs, self.ensemble_size))
        # self.exo_percentile_ranks_by_model = np.zeros((self.n_tcrs, self.ensemble_size))
        # self.exo_percentile_ranks = np.zeros((self.n_tcrs, 1))
    
    def safe_to_stop(self,
                     rank_threshold: float) -> bool:
        """Check if ALL ranks are higher than the threshold 

        Parameters
        ----------
        rank_threshold : float
            Rank threshold 

        Returns
        -------
        bool
            If ALL ranks are higher than the threshold 
        """
        return np.all(self.exo_df['percentile_ranks'].values > rank_threshold)
        
        
class pmhc_tcr_class:
    def __init__(self,
                 pmhc_id: int,
                 tcr_ids: list,
                 pmhc_species: str,
                 tcr_species: list,
                 ensemble_size: int) -> None:
        """The intermediate pmhc-tcr class 
        
        For each pmhc_tcr class, it contains two pmhc_species_tcr_class 

        Parameters
        ----------
        pmhc_id : int
            The unique id assigned to the pMHC
        tcr_ids : list
            A list of unique ids assigned to the paired TCRs of the chosen species 
        pmhc_species : str
            The species of the pMHC 
        tcr_species : list
            The species of the TCRs  
        ensemble_size : int
            Number of classifiers in the ensemble of the pMTnet Omni model 
        """
        self.pmhc_id = pmhc_id
        self.pmhc_species = pmhc_species
        
        tcr_ids_by_species = {"human": [],
                              "mouse": []}
        for i, tcr_id in enumerate(tcr_ids):
            tcr_ids_by_species[tcr_species[i].lower()].append(tcr_id)
        
        self.pmhc_species_tcr_objects = {"human": pmhc_species_tcr_class(pmhc_id=self.pmhc_id,
                                                                         tcr_ids=tcr_ids_by_species['human'],
                                                                         pmhc_species=self.pmhc_species,
                                                                         tcr_species="human",
                                                                         ensemble_size=ensemble_size),
                                         "mouse": pmhc_species_tcr_class(pmhc_id=self.pmhc_id,
                                                                         tcr_ids=tcr_ids_by_species['mouse'],
                                                                         pmhc_species=self.pmhc_species,
                                                                         tcr_species="mouse",
                                                                         ensemble_size=ensemble_size)}
    def __call__(self,
                 tcr_species: str,
                 background_tcr_embedding: Optional[torch.tensor]=None):
        df = self.get_pmhc_tcr(tcr_species=tcr_species,
                               background_tcr_embedding=background_tcr_embedding)
        return df
    
    
    def get_pmhc_tcr(self,
                     tcr_species: str,
                     background_tcr_embedding: Optional[torch.tensor]=None) -> pd.DataFrame:

        df = self.pmhc_species_tcr_objects[tcr_species].get_pmhc_tcr(background_tcr_embedding=background_tcr_embedding)
        
        return df
    
    def update_endo_ranks(self, 
                          tcr_species: str,
                          logits: np.ndarray) -> None:
        self.pmhc_species_tcr_objects[tcr_species].update_endo_ranks(logits=logits)
    
    def update_exo_ranks(self, 
                         tcr_species: str,
                         logits: np.ndarray) -> None:
        self.pmhc_species_tcr_objects[tcr_species].update_exo_ranks(logits=logits)
    
    def compute_percentile_ranks(self,
                                 tcr_species: str,
                                 background_size: int) -> None:
        self.pmhc_species_tcr_objects[tcr_species].compute_percentile_ranks(background_size=background_size)
        
    def reset_exo_ranks(self, 
                        tcr_species: str) -> None:
        self.pmhc_species_tcr_objects[tcr_species].reset_exo_ranks()

    def safe_to_stop(self,
                     rank_threshold: float) -> bool:
        temp = []
        for tcr_species in self.pmhc_species_tcr_objects:
            temp.append(self.pmhc_species_tcr_objects[tcr_species].safe_to_stop(rank_threshold=rank_threshold))
        return np.all(temp)
    
    
class pmhc_class:
    def __init__(self,
                 pmhc_df: pd.DataFrame,
                 tcr_df: pd.DataFrame,
                 id_df: pd.DataFrame,
                 other_df: pd.DataFrame,
                 ensemble_size: int,
                 partition_size: int) -> None:
        """The highest level of pmhc-tcr classes 
        
        The pMTnet Omni class is supposed to only interact with this class and this 
        class will interact with other sub-classes. 
        
        The rationale behind this design is that we desire to separate computation 
        and data storage as much as possible. The pMTnet Omni class will ask this 
        class for pieces of data and perform some computation and store the results 
        back to this class. 

        Parameters
        ----------
        pmhc_df : pd.DataFrame
            The pMHC dataframe that maps pmhc id to its corresponding information 
        tcr_df : pd.DataFrame
            The TCR dataframe that maps tcr id to its corresponding information 
        id_df : pd.DataFrame
            The id dataframe that maps pmhc id to its corresponding tcr ids
        other_df : pd.DataFrame
            Other part of the original user_df that has nothing to do with prediction along with the original observation id 
        ensemble_size : int
            The number of models in the ensemble 
        partition_size: int 
            Size of each partition. Used to save GPU memory.
        """
        
        # Store the dataframes 
        self.pmhc_df = pmhc_df
        self.tcr_df = tcr_df
        self.id_df = id_df
        self.other_df = other_df
        self.ensemble_size = ensemble_size
        self.partition_size = partition_size
        # Create pmhc and tcr embeddings 
        self.pmhc_embedding = None
        self.tcr_embedding = None
        # pmhc_tcr_objects dictionary holds all the instances of the pmhc_tcr_class
        # indexed by pmhc id 
        self.pmhc_tcr_objects = {}
        # For each pmhc_id, we extract all the corresponding tcr ids 
        # as well as the species information and use them to create 
        # pmhc_tcr_class object 
        for row in range(self.id_df.shape[0]):
            pmhc_id = self.id_df.at[row, 'pmhc_id']
            tcr_ids = self.id_df.at[row, 'tcr_id']
            pmhc_species = self.pmhc_df.loc[pmhc_id, "pmhc_species"]
            tcr_species = list(self.tcr_df.loc[tcr_ids, "tcr_species"])
            
            self.pmhc_tcr_objects[pmhc_id] = pmhc_tcr_class(pmhc_id=pmhc_id,
                                                            tcr_ids=tcr_ids,
                                                            pmhc_species=pmhc_species,
                                                            tcr_species=tcr_species,
                                                            ensemble_size=self.ensemble_size)
        # Create objects that will later be used to interact with 
        # pMTnet_Omni_class 
        # socket will be used to accept the data send back 
        self.socket = self._socket()
        # These will be used to temporarily store data
        self.temp_storage_list = []
        self.temp_storage_single = None
        # This tells the class which member to iterate over
        self.iteration_mode = None
        # This directs the data to its appropriate storage place 
        self.destination = "list"
        
    @coroutine
    def _socket(self) -> None:
        """This acts as a socket that will keep receiving data 
        """
        while True:
            value = (yield)
            if self.destination == "list":
                self.temp_storage_list.append(value)
            else:
                self.temp_storage_single = value
    
    def send(self, value: Union[torch.tensor, float],
             destination: str="list") -> None:
        self.destination = destination
        self.socket.send(value)
    
    def clear_temp(self) -> None:
        """Clears both temporary storage places 
        """
        self.temp_storage_list = []
        self.temp_storage_single = None
        self.iteration_mode = None

    def compute_percentile_ranks(self,
                                 background_size) -> None:
        for pmhc_id in self.pmhc_tcr_objects: 
            self.pmhc_tcr_objects[pmhc_id].compute_percentile_ranks(tcr_species="human",
                                                                    background_size=background_size)
            self.pmhc_tcr_objects[pmhc_id].compute_percentile_ranks(tcr_species="mouse",
                                                                    background_size=background_size)
    
    def reset_exo_ranks(self) -> None:
        for pmhc_id in self.pmhc_tcr_objects: 
            self.pmhc_tcr_objects[pmhc_id].reset_exo_ranks(tcr_species="human")
            self.pmhc_tcr_objects[pmhc_id].reset_exo_ranks(tcr_species="mouse")
    
    def safe_to_stop(self,
                     rank_threshold: float) -> bool:
        temp = []
        for pmhc_id in self.pmhc_tcr_objects:
            temp.append(self.pmhc_tcr_objects[pmhc_id].safe_to_stop(rank_threshold=rank_threshold))
        return np.all(temp)
    
    def __call__(self, mode):
        self.iteration_mode = mode
        return self 
        
    def __iter__(self):
        self.temp_storage_list = []
        if self.iteration_mode == "encode_pmhc":
            partitions = partition_dataframe(df=self.pmhc_df, 
                                             partition_size=self.partition_size)
            for partition in partitions:
                yield partition
            
            self.pmhc_embedding = torch.cat(self.temp_storage_list, dim=0)
            self.clear_temp()
            
        elif self.iteration_mode == "encode_tcr":
            partitions = partition_dataframe(df=self.tcr_df, 
                                             partition_size=self.partition_size)
            for partition in partitions:
                yield partition
            
            self.tcr_embedding = torch.cat(self.temp_storage_list, dim=0)
            self.clear_temp()
            
        elif self.iteration_mode.startswith("predict_"):
            tcr_species = self.iteration_mode.split("_")[1] 
            for _, pmhc_tcr_object in self.pmhc_tcr_objects.items():
                self.temp_storage_list = []
                partitions = partition_dataframe(pmhc_tcr_object(tcr_species=tcr_species,
                                                                 background_tcr_embedding=self.temp_storage_single),
                                                 partition_size=self.partition_size)
                for partition in partitions:
                    pmhc_ids = partition["pmhc"].values
                    tcr_ids = partition["tcr"].values
                    pmhc_embedding = self.pmhc_embedding[pmhc_ids, :]
                    if self.temp_storage_single is None:
                        # If computing logits using provided pairs 
                        tcr_embedding = self.tcr_embedding[tcr_ids, :]
                    else:
                        # If using background tcrs 
                        tcr_embedding = self.temp_storage_single[tcr_ids, :]
                    
                    yield pmhc_embedding, tcr_embedding
                
                logits = np.concatenate(self.temp_storage_list, axis=0)
                if self.temp_storage_single is None:
                    pmhc_tcr_object.update_endo_ranks(tcr_species=tcr_species,
                                                      logits=logits)
                else:
                    pmhc_tcr_object.update_exo_ranks(tcr_species=tcr_species,
                                                      logits=logits)   
            self.clear_temp()
        else:
            raise Exception