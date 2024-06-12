# Data IO
import os
import json
import pandas as pd
from copy import deepcopy

# Numeric manipulation
import numpy as np

# PyTorch
import torch

# To entertain users
from tqdm.auto import tqdm

# Typing
from typing import Optional, Tuple, Union

# pMTnet_Omni modules
try:
    from pMTnet_Omni.models.classifier import ensemble_pMHCTCR
except ImportError:
    from pMTnet_Omni.models.load_dummies import ensemble_pMHCTCR
from pMTnet_Omni.encoders.encoder_class import encoder_class
from pMTnet_Omni.background_tcr_loaders import background_species_tcr_dataset_class, background_species_tcr_dataloader_class
# from pMTnet_Omni_Document.data_curation import read_file
from pMTnet_Omni.utilities import batchify_check_size, setup_seed, partition_dataframe, amalgamate_partitions, check_data_dir_structure


class pMTnet_Omni_class:
    def __init__(self,
                 model_device: Optional[Union[str, torch.device]] = None,
                 data_dir: str = "./data/",
                 seed: Optional[int] = None,
                 extra_mhc_dict: Optional[Union[dict, str]] = None,
                 mutation_mode: str = "zero",
                 partition_size: int=500) -> None:
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
        partition_size: int 
            Size of each partition. Used to save GPU memory.

        """
        # Set model device
        if model_device is None:
            self.model_device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
        elif isinstance(model_device, str):
            self.model_device = torch.device(model_device)
        else:
            self.model_device = model_device
        
         # Columns related to pmhc 
        self.pmhc_cols = ['peptide', 
                        'mhca',
                        'mhcb',
                        'mhcaseq',
                        'mhcbseq',
                        "mhc_class",
                        'pmhc_species',
                        "mhca_use_seq", 
                        "mhcb_use_seq"]
        self.pmhc_embedding_cols = None
        # Columns related to tcrs 
        self.tcr_cols = ["va", "vb", "vaseq", "vbseq", "cdr3a", "cdr3b", "tcr_species"]
        self.tcr_embedding_cols = None
        
        # Set the mutation mode 
        self.mutation_mode = mutation_mode
        
        # Get paths to required data 
        encoder_path, classifier_path, model_info_path = check_data_dir_structure(data_dir)

        self.background_data_dir = os.path.join(data_dir, "background_tcrs/")
        self.background_pmhc_dir = os.path.join(data_dir, "background_pmhcs/")
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
        
        # User df & embeddings
        self.user_df = None
        self.user_tcr_embedding_df = None
        self.user_pmhc_embedding_df = None

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
                                                classifier_path=classifier_path,
                                                model_info_path=model_info_path)
        self.model_names = list(self.classifier_model.model_dict.keys())
        self.partition_size = partition_size

    def _read_user_df(self,
                      user_data: Union[str, pd.DataFrame],
                      **kwargs) -> None:
        """Read in user dataframe

        Parameters
        ----------
        user_data: str, pd.DataFrame
            The path to the user dataframe or the actual dataframe
        **kwargs
            Other key-value pairs for the pd.read_csv function

        """
        print("Reading user data frame...\n")
        if isinstance(user_data, str):
            # if the path is provided 
            # we use pd.read_csv to read the data 
            # NOTE: WE WILL NOT CHECK THE DATA 
            # THIS SHOULD BE DONE VIA data_curation
            self.user_df = pd.read_csv(user_data, **kwargs)
        elif isinstance(user_data, pd.DataFrame):
            self.user_df = deepcopy(user_data)
        else:
            raise TypeError("user_data should either the path to the file or a pd.DataFrame")
        # Make sure that all the NaN are converted to ""
        self.user_df = self.user_df.fillna("")
        self.user_df.drop_duplicates(subset=self.pmhc_cols+self.tcr_cols,
                                    inplace=True, ignore_index=True)
        # Add output columns 
        for model_id in self.model_names:
            self.user_df['logit_m_'+str(model_id)] = np.inf
        self.user_df['logit'] = np.inf 
        # Generate observation_id, pmhc_id and tcr_id for each observation 
        self.user_df['observation_id'] = self.user_df.index
        self.user_df["pmhc_id"] = self.user_df.groupby(self.pmhc_cols).ngroup()
        self.user_df["tcr_id"] = self.user_df.groupby(self.tcr_cols).ngroup()
        # Seperate the dataframe to avoid duplicates 
        self.user_pmhc_embedding_df = deepcopy(self.user_df[self.pmhc_cols + ['pmhc_id']])
        self.user_pmhc_embedding_df.drop_duplicates(inplace=True, ignore_index=True)
        self.user_tcr_embedding_df = deepcopy(self.user_df[self.tcr_cols + ['tcr_id']])
        self.user_tcr_embedding_df.drop_duplicates(inplace=True, ignore_index=True)
        
    def _encode_user_df(self) -> None:
        """Encode pMHC and TCRs provided in the user dataframe 

        """
        # We will encode pmhc first 
        
        pmhc_partition = partition_dataframe(df=self.user_pmhc_embedding_df,
                                            partition_size=self.partition_size)
        tcr_partition = partition_dataframe(df=self.user_tcr_embedding_df,
                                            partition_size=self.partition_size)
        print("Encoding pMHCs...\n")
        for partition in tqdm(pmhc_partition):
            _, pmhc_embedding = self.encoder.encode(df=pmhc_partition[partition],
                                                    is_embedding=False,
                                                    verbose=False)
            self.pmhc_embedding_cols = ['pmhc_embedding_'+str(k) for k in range(pmhc_embedding.shape[1])]
            temp_embedding_df = pd.DataFrame(pmhc_embedding.to('cpu'),
                                              columns=self.pmhc_embedding_cols)
            pmhc_partition[partition] = pd.concat([pmhc_partition[partition],
                                                   temp_embedding_df],
                                                  axis=1)
        print("Encoding TCRs...\n")
        for partition in tqdm(tcr_partition):
            tcr_embedding, _ = self.encoder.encode(df=tcr_partition[partition],
                                                    is_embedding=False,
                                                    verbose=False)
            self.tcr_embedding_cols = ['tcr_embedding_'+str(k) for k in range(tcr_embedding.shape[1])]
            temp_embedding_df = pd.DataFrame(tcr_embedding.to('cpu'),
                                              columns=self.tcr_embedding_cols)
            tcr_partition[partition] = pd.concat([tcr_partition[partition],
                                                   temp_embedding_df],
                                                  axis=1)

        self.user_pmhc_embedding_df = amalgamate_partitions(pmhc_partition)
        self.user_tcr_embedding_df = amalgamate_partitions(tcr_partition)


    def _get_user_df_output(self) -> None:
        """Predict using the dataframe

        Use the classifier model to produce logits of records in the dataframe

        """
        user_df_partition = partition_dataframe(df=self.user_df,
                                                partition_size=self.partition_size)
        print("Computing logits...\n")
        for partition in tqdm(user_df_partition):
            pmhc_embedding = user_df_partition[partition].merge(self.user_pmhc_embedding_df[self.pmhc_embedding_cols+['pmhc_id']],
                                                                how='left',
                                                                on='pmhc_id')[self.pmhc_embedding_cols]
            tcr_embedding = user_df_partition[partition].merge(self.user_tcr_embedding_df[self.tcr_embedding_cols+['tcr_id']],
                                                                how='left',
                                                                on='tcr_id')[self.tcr_embedding_cols]
            pmhc_embedding = torch.tensor(pmhc_embedding.values, dtype=torch.float32).to(self.model_device)
            tcr_embedding = torch.tensor(tcr_embedding.values, dtype=torch.float32).to(self.model_device)
            
            df_output = self.classifier_model.predict(
                            pmhc_embedding=pmhc_embedding,
                            tcr_embedding=tcr_embedding)
            for i, model_id in enumerate(self.model_names):
                user_df_partition[partition]['logit_m_'+str(model_id)] = df_output[:,i]
            user_df_partition[partition]['logit'] = np.mean(df_output, axis=1)
        
        self.user_df = amalgamate_partitions(user_df_partition)
        
    def _generate_background_dataloader(self,
                                        load_embedding: bool,
                                        replacement: bool,
                                        check_size: int,
                                        load_size: int,
                                        minibatch_size: int) -> Tuple[list, list, background_species_tcr_dataloader_class, background_species_tcr_dataloader_class]:
        """Generate background dataloaders
        This function though can be called explicitly by the users, is only intended to be used 
        internally. 

        Parameters 
        ----------
        load_embedding: bool
            Whether or not to load embeddings of the background tcrs 
        replacement: bool
            Whether or not to sample with replacement
        check_size: int 
            The number of background tcrs to be used to check against the user results 
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
                                                                        load_embedding=load_embedding,
                                                                        load_size=load_size)
        background_mouse_dataset = background_species_tcr_dataset_class(data_dir=self.background_data_dir,
                                                                        species="mouse",
                                                                        load_embedding=load_embedding,
                                                                        load_size=load_size)
        background_human_dataloader = background_species_tcr_dataloader_class(background_human_dataset,
                                                                              batch_size=batch_size,
                                                                              replacement=replacement)
        background_mouse_dataloader = background_species_tcr_dataloader_class(background_mouse_dataset,
                                                                              batch_size=batch_size,
                                                                              replacement=replacement)
        return batch_size, batch_finished_indicator, background_human_dataloader, background_mouse_dataloader

    def _compute_tcr_percentile_rank(self,
                                    check_size: int=1000000,
                                    load_size: int = 1000000,
                                    minibatch_size: int = 50000) -> None:
        """Compute the rank percentiles based on background TCRs

        Parameters
        ----------
        check_size: int 
            The numbers of background tcrs to be used to check against the user results 
        load_size: int 
            The number of background tcrs to be loaded by the dataloaders 
        minibatch_size: int
            The number of background tcrs to be sampled each time a dataloader is called
        
        """
        
        print("Loading background TCR datasets\n")
        batch_size, batch_finished_indicator, background_human_dataloader, background_mouse_dataloader = self._generate_background_dataloader(load_embedding=True,
                                                                                                                                                replacement=True,
                                                                                                                                                check_size=check_size,
                                                                                                                                                load_size=load_size,
                                                                                                                                                minibatch_size=minibatch_size)
        # For each unique pmhc, we will use the same background tcrs 
        all_pmhc_ids = np.unique(self.user_df['pmhc_id']) 
        df_list = []
        for pmhc_id in tqdm(all_pmhc_ids):
            # We first obtain the corresponding pmhc embedding
            ind = (self.user_pmhc_embedding_df['pmhc_id'] == pmhc_id)
            pmhc = self.user_pmhc_embedding_df.loc[ind, self.pmhc_embedding_cols]
            pmhc = torch.tensor(pmhc.values, dtype=torch.float32)
            cols = ["tcr_percentile_rank_m_" + str(m) for m in self.model_names] + ['observation_id']
            # Depending on the species of the TCR, we will use different data loaders 
            for species in ['human', 'mouse']:
                if species == "human":
                    data_loader = background_human_dataloader
                else:
                    data_loader = background_mouse_dataloader
                # Extract records with the pmhc and the desired species 
                df = self.user_df.loc[(self.user_df['pmhc_id'] == pmhc_id) & (self.user_df['tcr_species'] == species),
                                        ['logit_m_'+str(m) for m in self.model_names]+['observation_id']]
                background_result_list = []
                if df.shape[0] > 0:
                    # We will first get all logits on the background 
                    for batch_size_ind, tcrs in enumerate(data_loader):
                        b_size = batch_size[batch_size_ind]
                        background_tcr_embedding, _ = self.encoder.encode(df=tcrs, is_embedding=True)
                        duplicated_pmhc_embedding = pmhc.repeat(b_size, 1) 
                        background_result = self.classifier_model.predict(
                                    pmhc_embedding=duplicated_pmhc_embedding.to(self.model_device),
                                    tcr_embedding=background_tcr_embedding)
                        background_result_list.append(background_result)
                    background_result = np.concatenate(background_result_list, 
                                                            axis=0)
                    background_result = pd.DataFrame(background_result,
                                                    columns=['logit_m_'+str(m) for m in self.model_names])
                    background_result['observation_id']=-1
                    # To compute ranks (not percentile ranks), we first compute the internal 
                    # rank, namely, we rank the logits of user_df first 
                    # This tells us for each output, how many pairs internally are "better"
                    for m in self.model_names:
                        df['_interal_rank_m_'+str(m)] = df['logit_m_'+str(m)].rank(ascending=False, pct=False)
                        background_result['_interal_rank_m_'+str(m)] = -1
                    # Once we concatnate the results with the original dataframe 
                    # we can compute the ranks for all pairs 
                    df = pd.concat([df, background_result],
                                    axis=0, join='inner')
                    # Now to obtain the final rank (each output ranked only against the background)
                    # we simply subtract the internal rank from the overal rank
                    for m in self.model_names:
                        df["tcr_percentile_rank_m_" + str(m)] = df['logit_m_'+str(m)].rank(ascending=False, pct=False)
                        df["tcr_percentile_rank_m_" + str(m)] -= df['_interal_rank_m_'+str(m)]
                        df["tcr_percentile_rank_m_" + str(m)] += 1
                        df["tcr_percentile_rank_m_" + str(m)] /= (check_size + 1)
                    # Then we remove the background 
                    df = df.loc[df['observation_id']!=-1, cols].reset_index(drop=True)
                    df_list.append(df)
        df = pd.concat(df_list, axis=0)
        # Merge the rank results back to the original data 
        self.user_df = self.user_df.merge(df, how='left',
                                          on='observation_id')
        # Final rank is the average 
        self.user_df["tcr_percentile_rank"] = self.user_df[["tcr_percentile_rank_m_" + str(m) for m in self.model_names]].mean(axis=1)
        self.user_df.sort_values(by='observation_id',
                                 ignore_index=True)
    
    def _compute_pmhc_percentile_rank(self,
                                      minibatch_size: int = 50000) -> None:
        """Compute the rank percentiles based on background pmhcs 

        Parameters
        ----------
        minibatch_size: int
            The number of background pmhcs to be sampled each time 
            
        """
        # We only have less than 1m background pmhcs 
        # we will use all of them 
        all_background_pmhc = pd.read_csv(os.path.join(self.background_pmhc_dir, 
                                                       "pmhc_embedding.csv"))
        human_background_pmhc = all_background_pmhc[all_background_pmhc['pmhc_species']=="human"].reset_index(drop=True)
        mouse_background_pmhc = all_background_pmhc[all_background_pmhc['pmhc_species']=="mouse"].reset_index(drop=True)
        
        # For each unique pmhc, we will use the same background pmhcs   
        all_tcr_ids = np.unique(self.user_df['tcr_id']) 
        df_list = []
        for tcr_id in tqdm(all_tcr_ids):
            # Obtain the embedding of the tcr 
            ind = (self.user_tcr_embedding_df['tcr_id'] == tcr_id)
            tcr = self.user_tcr_embedding_df.loc[ind, self.tcr_embedding_cols]
            tcr = torch.tensor(tcr.values, dtype=torch.float32)
            cols = ["pmhc_percentile_rank_m_" + str(m) for m in self.model_names] + ['observation_id']
            # Depending on the species of pmhc, we will use different backgrounds 
            for species in ['human', 'mouse']:
                if species == "human":
                    background_pmhc = human_background_pmhc
                else:
                    background_pmhc = mouse_background_pmhc
                df = self.user_df.loc[(self.user_df['tcr_id'] == tcr_id) & (self.user_df['pmhc_species'] == species),
                                        ['logit_m_'+str(m) for m in self.model_names]+['observation_id']]
                background_result_list = []
                if df.shape[0] > 0:
                    background_pmhc_partitions = partition_dataframe(df=background_pmhc,
                                                                     partition_size=minibatch_size)
                    
                    for partition in background_pmhc_partitions:
                        b_size = background_pmhc_partitions[partition].shape[0]
                        background_pmhc_embedding = torch.tensor(background_pmhc_partitions[partition][self.pmhc_embedding_cols].values,
                                                                 dtype=torch.float32)
                        duplicated_tcr_embedding = tcr.repeat(b_size, 1) 
                        background_result = self.classifier_model.predict(
                                    pmhc_embedding=background_pmhc_embedding.to(self.model_device),
                                    tcr_embedding=duplicated_tcr_embedding.to(self.model_device))
                        background_result_list.append(background_result)
                    background_result = np.concatenate(background_result_list, 
                                                            axis=0)
                    background_result = pd.DataFrame(background_result,
                                                    columns=['logit_m_'+str(m) for m in self.model_names])
                    background_result['observation_id']=-1
                    # We first compute internal rank 
                    for m in self.model_names:
                        df['_interal_rank_m_'+str(m)] = df['logit_m_'+str(m)].rank(ascending=False, pct=False)
                        background_result['_interal_rank_m_'+str(m)] = -1
                    # Once we concatnate the results with the original dataframe 
                    # we can compute the ranks 
                    df = pd.concat([df, background_result],
                                    axis=0, join='inner')
                    for m in self.model_names:
                        df["pmhc_percentile_rank_m_" + str(m)] = df['logit_m_'+str(m)].rank(ascending=False, pct=False)
                        df["pmhc_percentile_rank_m_" + str(m)] -= df['_interal_rank_m_'+str(m)]
                        df["pmhc_percentile_rank_m_" + str(m)] += 1
                        df["pmhc_percentile_rank_m_" + str(m)] /= (background_pmhc.shape[0] + 1)
                    
                    df = df.loc[df['observation_id']!=-1, cols].reset_index(drop=True)
                    df_list.append(df)
        df = pd.concat(df_list, axis=0)
        self.user_df = self.user_df.merge(df, how='left',
                                          on='observation_id') 
        self.user_df["pmhc_percentile_rank"] = self.user_df[["pmhc_percentile_rank_m_" + str(m) for m in self.model_names]].mean(axis=1)
        self.user_df.sort_values(by='observation_id',
                                 ignore_index=True)
        
    def _compute_derived_percentile_rank(self) -> None:
        """Compute derived percentile ranks 
        
        Based on the ranks computed on tcrs and pmhcs, 
        we can further compute avg and max ranks. The computation is first 
        done per each model. Then we average the results within an ensemble 
        
        """
        for m in self.model_names:
            cols = ['tcr_percentile_rank_m_' + str(m), 'pmhc_percentile_rank_m_' + str(m)]
            self.user_df['avg_percentile_rank_m_'+str(m)] = self.user_df[cols].mean(axis=1)
            self.user_df['max_percentile_rank_m_'+str(m)] = self.user_df[cols].max(axis=1)
            self.user_df['min_percentile_rank_m_'+str(m)] = self.user_df[cols].min(axis=1)
        self.user_df['avg_percentile_rank'] = self.user_df[["avg_percentile_rank_m_" + str(m) for m in self.model_names]].mean(axis=1)
        self.user_df['max_percentile_rank'] = self.user_df[["max_percentile_rank_m_" + str(m) for m in self.model_names]].mean(axis=1)
        self.user_df['min_percentile_rank'] = self.user_df[["min_percentile_rank_m_" + str(m) for m in self.model_names]].mean(axis=1)
        self.user_df.sort_values(by='observation_id',
                                 ignore_index=True)

    def _clear_up_user_df(self) -> None:
        """Clear up the user dataframe 
        
        This function removes unnecessary columns from the user dataframe 
        """
        cols_to_drop = []
        for m in self.model_names:
            cols_to_drop += [c for c in self.user_df.columns if ("_m_"+str(m) in c)]
        
        cols_to_drop += ['observation_id', "tcr_id", "pmhc_id",
                        'mhca_use_seq', 'mhcb_use_seq',
                        'mhca', 'mhcb',
                        'mhcaseq', 'mhcbseq']
        
        self.user_df.drop(columns=cols_to_drop,
                          inplace=True)
        
    def read_user_df(self,
                     user_data: Union[str, pd.DataFrame],
                     **kwargs) -> None:
        """Read in user dataframe

        Parameters
        ----------
        user_data: str, pd.DataFrame
            The path to the user dataframe or the actual dataframe
        **kwargs
            Other key-value pairs for the pd.read_csv function
        
        """ 
        self._read_user_df(user_data=user_data,
                           **kwargs)
        self._encode_user_df()
        
    def predict(self,
                compute_percentile_rank: bool = True,
                output_file_path: Optional[str] = None,
                check_size: int = 1000000,
                load_size: int = 1000000,
                minibatch_size: int = 50000) -> None:
        """Predict TCR-pMHC affinity

        Parameters
        ----------
        compute_percentile_rank: bool
            Whether or not to compute percentile ranks. If False, then only logits will be computed 
        output_file_path: Optional[str]
            The path the output file. ONLY .csv file is supported 
        check_size: list 
            The numbers of background tcrs to be used to check against the user results 
        load_size: int 
            The number of background tcrs to be loaded by the dataloaders 
        minibatch_size: int
            The number of background tcrs to be sampled each time a dataloader is called

        """
        assert (output_file_path is None) or (os.path.exists(os.path.dirname(output_file_path))), "Path does not exist."
        # Compute logit for each model in the ensemble 
        # This should also output the average logit over the ensemble
        print("Getting raw affinity scores...\n")
        self._get_user_df_output()
        print("Done.\n")
        if compute_percentile_rank:
            # For each model in the ensemble, we compute the percentile rank
            print("Computing tcr percentile ranks\n")
            self._compute_tcr_percentile_rank(check_size=check_size,
                                            load_size=load_size,
                                            minibatch_size=minibatch_size)
            print("Computing pmhc percentile ranks\n")
            self._compute_pmhc_percentile_rank(minibatch_size=minibatch_size)
            print("Computing derived percentile ranks\n")
            self._compute_derived_percentile_rank()
        self._clear_up_user_df()
        
        if output_file_path is not None:
            filename, _ = os.path.splitext(output_file_path)
            self.user_df.to_csv(filename+'.csv',
                                index=False, header=True, sep=',')

