# Data IO
import os
import json
import pandas as pd
from copy import deepcopy

# Numeric manipulation
import numpy as np
from sklearn import metrics

# PyTorch
import torch

# To entertain users
from tqdm import tqdm
# For recording model output log
from contextlib import redirect_stdout
# For saving roc curves in a pdf file
# from matplotlib.backends.backend_pdf import PdfPages

# Typing
from typing import Optional, Tuple, Union

# pMTnet_Omni modules
try:
    from pMTnet_Omni.models.classifier import ensemble_pMHCTCR
except ImportError:
    from pMTnet_Omni.models.load_dummies import ensemble_pMHCTCR
from pMTnet_Omni.encoders.encoder_class import encoder_class
from pMTnet_Omni.pmhc_tcr_class import pmhc_class
from pMTnet_Omni.background_tcr_loaders import background_species_tcr_dataset_class, background_species_tcr_dataloader_class
# from pMTnet_Omni_Document.data_curation import read_file
from pMTnet_Omni.utilities import batchify_check_size, setup_seed, check_data_dir_structure


class pMTnet_Omni_class:
    def __init__(self,
                 model_device: Optional[Union[str, torch.device]] = None,
                 data_dir: str = "./data/",
                 seed: Optional[int] = None,
                 extra_mhc_dict: Optional[Union[dict, str]] = None,
                 mutation_mode: str = "zero") -> None:
        """Initialize pMTnet_Omni_class object

        Parameters
        ---------
        model_device: Optional[Union[str, torch.device]]
            The device to use. Typically it would be cpu or cuda:0
        data_dir: str
            The directory containing data needed for model setups
        seed: Optional[int]
            The seed to be used 
        extra_mhc_dict: Optional[Union[dict, str]] = None
            A dictionary or a path to a .json file 
            containing extra MHC embeddings  
        mutation_mode: str
            How unknown AAs should be interpolated
            zero: 5 zeros 
            alanine: atchley factor for A
            average: average atchley factor 

        """
        # Set model device
        if model_device is None:
            self.model_device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
        elif isinstance(model_device, str):
            self.model_device = torch.device(model_device)
        else:
            self.model_device = model_device
        
        # Set the mutation mode 
        self.mutation_mode = mutation_mode
        
        # Get paths to required data 
        encoder_path, classifier_path = check_data_dir_structure(data_dir)

        self.background_data_dir = os.path.join(data_dir, "background_tcrs/")
        self.encoder_data_dir = os.path.join(data_dir, "data_for_encoders/")
        # Set seed
        if seed is not None:
            setup_seed(seed=seed)
        # Read in extra mhc dictionary
        self.extra_mhc_dict = None
        # extra mhc is supposed to a dictioanry 
        # whose keys are mhc sequences and 
        # whose values are the corresponding esm embeddings 
        if isinstance(extra_mhc_dict, dict):
            self.extra_mhc_dict = extra_mhc_dict
        if isinstance(extra_mhc_dict, str):
            with open(extra_mhc_dict, "r") as handle:
                temp_dict = json.load(handle)
            # We convert nested list to np.ndarray
            # Note that even if the dictionary is {}
            # this will give us {}
            self.extra_mhc_dict = {k: np.asarray(v) for k, v in temp_dict.items()}
        
        # pmhc object 
        self.pmhc_object = None
        
        # Create the encoder

        vGdVAEacheckpoint_path = encoder_path[0]
        vGdVAEbcheckpoint_path = encoder_path[1]
        cdr3VAEacheckpoint_path = encoder_path[2]
        cdr3VAEbcheckpoint_path = encoder_path[3]
        pMHCcheckpoint_path = encoder_path[4]

        self.encoder = encoder_class(encoder_data_dir=self.encoder_data_dir,
                                     model_device=self.model_device,
                                     vGdVAEacheckpoint_path=vGdVAEacheckpoint_path,
                                     vGdVAEbcheckpoint_path=vGdVAEbcheckpoint_path,
                                     cdr3VAEacheckpoint_path=cdr3VAEacheckpoint_path,
                                     cdr3VAEbcheckpoint_path=cdr3VAEbcheckpoint_path,
                                     pMHCcheckpoint_path=pMHCcheckpoint_path,
                                     extra_mhc_dict=self.extra_mhc_dict,
                                     mutation_mode=self.mutation_mode)

        # Initialize th clssifier model
        self.classifier_model = ensemble_pMHCTCR(model_device=self.model_device,
                                                classifier_path=classifier_path)
        self.ensemble_size = len(classifier_path)

    def read_user_df(self,
                      user_data: Union[str, pd.DataFrame],
                      partition_size: int=500,
                      verbose: bool = False,
                      **kwargs) -> None:
        """Read in user dataframe and encode pMHCs and TCRs 
        
        The function will first read in the dataframe/file provided by the user. 
        NOTE: The function DOES NOT perform data curation. This should be done 
        by another package/function. 
        
        The function will then assign a unique ID to each observation, pmhc, and 
        tcr. And it will group observations by pmhcs. 
        
        Parameters
        ----------
        user_data: str, pd.DataFrame
            The path to the user dataframe or the actual dataframe
        partition_size: int 
            Size of each partition. Used to save GPU memory.
        verbose: bool
            Whether or not to print out model process 
        **kwargs
            Other key-value pairs for the pd.read_csv function

        """
        print("Reading user data frame...\n")
        if isinstance(user_data, str):
            # if the path is provided 
            # we use pd.read_csv to read the data 
            # NOTE: WE WILL NOT CHECK THE DATA 
            # THIS SHOULD BE DONE VIA data_curation
            user_df = pd.read_csv(user_data, **kwargs)
        elif isinstance(user_data, pd.DataFrame):
            user_df = deepcopy(user_data)
        else:
            raise TypeError("user_data should either the path to the file or a pd.DataFrame")
        # Make sure that all the NaN are converted to ""
        user_df = user_df.fillna("")
        
        # We will perform a series of operations that aim to 
        # group the observations by pmhcs and tcrs 
        
        # First we index the observations 
        user_df['observation_id'] = user_df.index
        # Columns related to pmhc 
        pmhc_cols = ['peptide', 
                    'mhca',
                    'mhcb',
                    'mhcaseq',
                    'mhcbseq',
                    "mhc_class",
                    'pmhc_species',
                    "mhca_use_seq", 
                    "mhcb_use_seq"]
        # Columns related to tcrs 
        tcr_cols = ["va", "vb", "vaseq", "vbseq", "cdr3a", "cdr3b", "tcr_species"]
        # Generate pmhc_id and tcr_id for each observation 
        user_df["pmhc_id"] = user_df.groupby(pmhc_cols).ngroup()
        user_df["tcr_id"] = user_df.groupby(tcr_cols).ngroup()
        
        # For pmhc_df, we keep all observation ids 
        # and the first occurrence of that pmhc 
        pmhc_df = user_df.groupby(by=['pmhc_id'], 
                                  as_index=True).agg({**{"observation_id": list},
                                                      **{key: "first" for key in pmhc_cols}})
        # Sort by id (index) to make sure id 0 is the first one 
        pmhc_df.sort_index(axis=0, ascending=True, inplace=True)
        # Again, keep all observation ids 
        # and the first occurrence of that tcr 
        tcr_df = user_df.groupby(by=['tcr_id'], 
                                 as_index=True).agg({**{"observation_id": list},
                                                     **{key: "first" for key in tcr_cols}})
        tcr_df.sort_index(axis=0, ascending=True, inplace=True)
        # 
        id_df = user_df.groupby(['pmhc_id'], as_index=False)[["tcr_id"]].agg(lambda x: [y for y in x.unique()])
        
        # Other df 
        other_df = user_df[user_df.columns.difference(pmhc_cols + 
                                                      tcr_cols + 
                                                      ['pmhc_id', 'tcr_id'])]
        
        self.pmhc_object = pmhc_class(pmhc_df=pmhc_df,
                                      tcr_df=tcr_df,
                                      id_df=id_df,
                                      other_df=other_df,
                                      ensemble_size=self.ensemble_size,
                                      partition_size=partition_size)
        
        # Encode pmhc and tcrs 
        for partition in self.pmhc_object(mode='encode_pmhc'):
            _, p_embedding = self.encoder.encode(df=partition,
                                                 is_embedding=False,
                                                 verbose=False)
            self.pmhc_object.send(p_embedding.to('cpu'), 
                                  destination="list")
            
        for partition in self.pmhc_object(mode='encode_tcr'):
            t_embedding, _ = self.encoder.encode(df=partition,
                                                 is_embedding=False,
                                                 verbose=False)
            self.pmhc_object.send(t_embedding.to('cpu'), 
                                  destination="list")

    def _get_pmhc_tcr_output(self,
                             tcr_species: str,
                             background_tcr_embedding: Optional[torch.tensor]=None) -> None:
        self.pmhc_object.send(background_tcr_embedding,
                              destination="single")
        for pmhc_embedding, tcr_embedding in self.pmhc_object("predict_"+tcr_species):
            logits_p = self.classifier_model.predict(
                pmhc_embedding=pmhc_embedding.to(self.model_device),
                tcr_embedding=tcr_embedding.to(self.model_device)
            ) 
            self.pmhc_object.send(logits_p, 
                                  destination="list")
    
    def _compute_logit(self):
        for tcr_species in ['human', 'mouse']:
            self._get_pmhc_tcr_output(tcr_species=tcr_species,
                                      background_tcr_embedding=None)
    
    def _generate_background_dataloader(self,
                                        check_size: list,
                                        load_size: int,
                                        minibatch_size: int) -> Tuple[list, list, background_species_tcr_dataloader_class, background_species_tcr_dataloader_class]:
        """Generate background dataloaders
        This function though can be called explicitly by the users, is only intended to be used 
        internally. 

        Parameters 
        ----------
        check_size: list 
            The numbers of background tcrs to be used to check against the user results 
        load_size: int 
            The number of background tcrs to be loaded by the dataloaders 
        minibatch_size: int
            The number of background tcrs to be sampled each time a dataloader is called

        Returns
        ---------
        list: 
            The batch sizes 
        list: 
            A list of indicators indicating if the batch is over 
        background_species_tcr_dataset_class:
            Human background tcr dataloader
        background_species_tcr_dataset_class:
            Mouse background tcr dataloader

        """
        batch_size, batch_finished_indicator = batchify_check_size(check_size=check_size,
                                                                   minibatch_size=minibatch_size)
        background_human_dataset = background_species_tcr_dataset_class(data_dir=self.background_data_dir,
                                                                        species="human",
                                                                        load_embedding=True,
                                                                        load_size=load_size)
        background_mouse_dataset = background_species_tcr_dataset_class(data_dir=self.background_data_dir,
                                                                        species="mouse",
                                                                        load_embedding=True,
                                                                        load_size=load_size)
        background_human_dataloader = background_species_tcr_dataloader_class(background_human_dataset,
                                                                              batch_size=batch_size,
                                                                              replacement=True)
        background_mouse_dataloader = background_species_tcr_dataloader_class(background_mouse_dataset,
                                                                              batch_size=batch_size,
                                                                              replacement=True)
        return batch_size, batch_finished_indicator, background_human_dataloader, background_mouse_dataloader


    def _compute_percentile_rank(self,
                                 rank_threshold: Union[float, list, np.ndarray]=0.03,
                                 check_size: list = [1000, 10000, 100000],
                                 load_size: int = 1000000,
                                 minibatch_size: int = 50000) -> None:
        """Compute the rank percentiles 

        Parameters
        ----------
        rank_threshold: float, list, np.ndarray
            The rank percentile threshold greater than which further verification is NOT conducted 
        load_embedding: bool
            Whether or not to load embeddings of the background tcrs 
        replacement: bool
            Whether or not to sample with replacement
        check_size: list 
            The numbers of background tcrs to be used to check against the user results 
        load_size: int 
            The number of background tcrs to be loaded by the dataloaders 
        minibatch_size: int
            The number of background tcrs to be sampled each time a dataloader is called
        """
        
        # Make sure check size is in an ascending order
        # We sort the check size in an ascending order
        # and we make sure that the input is a list 
        check_size = np.sort(list(check_size))
        
        # We first make sure that the variable is an 1d np.array 
        # Note that this also works if the variable is a float 
        rank_threshold = np.array(rank_threshold).flatten()
        # We sort the thresholds in a descending order 
        rank_threshold = np.sort(rank_threshold)[::-1]
        # If check size - 1 is longer than threshold
        # we pad the minimum values 
        rank_threshold = np.pad(rank_threshold, 
                                pad_width=(0, np.max([0, len(check_size) - 1 - len(rank_threshold)])),
                                mode='minimum')
        # Finally, we pad a 0 at then end 
        # This is just so that the length of the threholds 
        # would be the same as the check size list 
        rank_threshold = np.pad(rank_threshold, 
                                pad_width=(0,1),
                                mode='constant',
                                constant_values=0.0)
        
        batch_size, batch_finished_indicator, background_human_dataloader, background_mouse_dataloader = self._generate_background_dataloader(check_size=check_size,
                                                                                                                                             load_size=load_size,
                                                                                                                                             minibatch_size=minibatch_size)
        
        # Compute background logits
        # and update ranks for each tcr 
        check_size_ind = 0
        b_size_so_far = 0
        for batch_size_ind, (human_background_tcr, mouse_background_tcr) in enumerate(zip(background_human_dataloader, background_mouse_dataloader)):
            b_size = batch_size[batch_size_ind]
            b_size_so_far += b_size
            
            human_background_tcr_embedding, _ = self.encoder.encode(
                df=human_background_tcr, is_embedding=True)
            
            self._get_pmhc_tcr_output(tcr_species="human",
                                      background_tcr_embedding=human_background_tcr_embedding)
            
            mouse_background_tcr_embedding, _ = self.encoder.encode(
                df=mouse_background_tcr, is_embedding=True)
            
            self._get_pmhc_tcr_output(tcr_species="mouse",
                                      background_tcr_embedding=mouse_background_tcr_embedding)
            
            if batch_finished_indicator[batch_size_ind]:
                
                self.pmhc_object.compute_percentile_ranks(background_size=check_size[check_size_ind])
                
                # If all percentile rankings are greater than the current threshold 
                # we stop the validation. 
                # The last threshold should be 0 (padded)
                if self.pmhc_object.safe_to_stop(rank_threshold=rank_threshold[check_size_ind]):
                    break
                
                # Else, we reset the ranks and percentile ranks 
                # And we update the indicies 
                if check_size_ind < len(check_size)-1:
                    # This means that we still have some more checks to do 
                    self.pmhc_object.reset_exo_ranks()
                    # update the indicies 
                    b_size_so_far = 0
                    check_size_ind += 1
                                
                                    
    def _amalgamate(self) -> Tuple[dict, dict, dict]:
        """Amalgamate all the partitions 

        This function will ``glue" all the partitions back together. This mainly used 
        for outputting our model results 
        """
        df_dict = {}
        tcr_dict = {}
        pmhc_dict = {}
        for missingness in self.user_df_dict:
            if len(self.user_df_dict[missingness]) == 0:
                df_dict[missingness] = {}
                tcr_dict[missingness] = {}
                pmhc_dict[missingness] = {}
                continue
            # DF
            df = [self.user_df_dict[missingness][split][partition]
                  for split in self.user_df_dict[missingness]
                  for partition in self.user_df_dict[missingness][split]]
            df = pd.concat(df, axis=0, ignore_index=True)
            df_dict[missingness] = df
            # TCRS
            tcrs = [self.user_tcr_embedding_dict[missingness][split][partition]
                    for split in self.user_tcr_embedding_dict[missingness]
                    for partition in self.user_tcr_embedding_dict[missingness][split]]
            tcrs = torch.cat(tcrs, dim=0)
            tcr_dict[missingness] = tcrs
            # pMHC
            # We propagate across missingness as we assume users use this function for
            # validation, visulation, etc
            pmhcs = [self.user_pmhc_embedding_dict[split]
                        for split in self.user_pmhc_embedding_dict if split != "partitions"]
            pmhcs = torch.cat(pmhcs, dim=0)
            pmhc_dict[missingness] = pmhcs

        return df_dict, tcr_dict, pmhc_dict

        
    def predict(self,
                compute_percentile_rank: bool = True,
                output_file_path: Optional[str] = None,
                rank_threshold: Union[float, list, np.ndarray] = 0.03,
                check_size: list = [1000, 10000, 100000],
                load_size: int = 1000000,
                minibatch_size: int = 50000) -> None:
        """Predict TCR-pMHC affinity

        Parameters
        ----------
        compute_percentile_rank: bool
            Whether or not to compute percentile ranks. If False, then only logits will be computed 
        output_file_path: Optional[str]
            The path the output file. ONLY .csv file is supported 
        rank_threshold: Union[float, list, np.ndarray]
            The rank percentile threshold greater than which further verification is NOT conducted 
        check_size: list 
            The numbers of background tcrs to be used to check against the user results 
        load_size: int 
            The number of background tcrs to be loaded by the dataloaders 
        minibatch_size: int
            The number of background tcrs to be sampled each time a dataloader is called

        """

        # Compute logit for each model in the ensemble 
        # This should also output the average logit over the ensemble
        print("Getting raw affinity scores...\n")
        self._compute_logit()
        print("Done.\n")
        if compute_percentile_rank:
            # For each model in the ensemble, we compute the percentile rank
            print("Computing percentile ranks for each model in the ensemble\n")

            self._compute_percentile_rank(rank_threshold=rank_threshold,
                                        check_size=check_size,
                                        load_size=load_size,
                                        minibatch_size=minibatch_size)
            
        if output_file_path is not None:
            filename, _ = os.path.splitext(output_file_path)
            df_dict, _, _ = self._amalgamate()
            for missingness in self.user_df_dict:
                if len(self.user_df_dict[missingness]) == 0:
                    continue
                df_dict[missingness].to_csv(filename+'_'+missingness+'.csv',
                                            index=False, header=True, sep=',')

    def validate(self,
                 true_labels: Union[pd.DataFrame, np.ndarray, list],
                 b: int = 0,
                 ndigits: int = 3) -> dict:
        """Compute aurocs and plot roc curves 

        Parameters
        ----------
        true_labels: Union[pd.DataFrame, np.ndarray, list]
            A one column pandas dataframe containing the true labels. We assume 1 is binding and 0 is not binding
        b: int
            The trial number to validate
        ndigits: int
            The number of digits in aucs to keep 

        Returns
        ---------
        dict: 
            A dictionary containing the aucs and their breakdowns 

        """

        if isinstance(true_labels, list) or isinstance(true_labels, np.ndarray):
            true_labels = np.array(true_labels)
        else:
            # We assume that true_labels is a one-column dataframe
            true_labels = true_labels.iloc[:, 0]
        truth = pd.DataFrame({'true_label': true_labels})
        df_dict, _, _ = self._amalgamate()

        # The result will be a nested dictionary
        # {"complete": {"overall": {"fpr": np.ndarray,
        #                           "tpr": np.ndarray,
        #                           "precision": np.ndarray,
        #                           "recall": np.ndarray,
        #                           "auroc": float,
        #                           "auprc": float},
        #               "human overall": {}},
        #  "alpha missing": {},
        #  "beta missing": {}}
        result_dicts = {key: None for key in self.user_df_dict}

        # For each missingness type, we also compute different auc types
        auc_to_compute = ["overall", "human overall", "mouse overall",
                          "human class i", "human class ii",
                          "mouse class i", "mouse class ii",
                          "class i", "class ii"]

        for missingness in self.user_df_dict:
            if len(self.user_df_dict[missingness]) == 0:
                continue
            # First get mhc classes
            mhc_class = df_dict[missingness]['mhc_class'].to_frame()
            # And the corresponding computed rank percentiles
            # NOTE that the closer the rank percentiles are to 0
            # the more likely the model thinks the pairs will bind
            # As a 1 in the true labels means binding
            # we need to flip the percentiles
            pred = (1-df_dict[missingness]
                    ['percentile_rank_'+str(b)]).to_frame()

            # Then we concatenate truth, predicted and mhc_class into one dataframe
            df = pd.concat([truth, pred.set_index(truth.index),
                           mhc_class.set_index(truth.index)], axis=1)

            # auc_dict is yet another nested dictionary
            # {"overall": {},
            #  "human overall": {},
            #  ...}

            auc_dict = {key: {} for key in auc_to_compute}

            for auc_type in auc_to_compute:
                if auc_type == "overall":
                    # If the column contains anyting
                    ind = df['mhc_class'].str.contains('.')
                elif auc_type == "human overall":
                    ind = df['mhc_class'].str.contains('human')
                elif auc_type == "mouse overall":
                    ind = df['mhc_class'].str.contains('mouse')
                else:
                    # NOTE that 'human class i' is a substring of 
                    # 'human class ii'
                    # We need $ at then end 
                    ind = df['mhc_class'].str.contains(auc_type+'$')

                if np.sum(ind) < 1:
                    continue

                fpr, tpr, _ = metrics.roc_curve(
                    y_true=df[ind]['true_label'], y_score=df[ind]['percentile_rank_'+str(b)])
                precision, recall, _ = metrics.precision_recall_curve(
                    y_true=df[ind]['true_label'], probas_pred=df[ind]['percentile_rank_'+str(b)])
                auroc = round(metrics.auc(fpr, tpr), ndigits=ndigits)
                auprc = round(metrics.auc(recall, precision), ndigits=ndigits)

                auc_dict[auc_type]["fpr"] = fpr
                auc_dict[auc_type]["tpr"] = tpr
                auc_dict[auc_type]["precision"] = precision
                auc_dict[auc_type]["recall"] = recall
                auc_dict[auc_type]["auroc"] = auroc
                auc_dict[auc_type]["auprc"] = auprc

            result_dicts[missingness] = auc_dict

        return result_dicts
