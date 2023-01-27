# Data IO
import os
import pandas as pd
from copy import deepcopy

# Numeric manipulation
import numpy as np

# PyTorch
import torch

# To entertain users
from tqdm import tqdm
# For recording model output log
from contextlib import redirect_stdout
# For saving roc curves in a pdf file
from matplotlib.backends.backend_pdf import PdfPages

# Typing
from typing import Optional, Tuple, Union

# pMTnet_Omni modules
from pMTnet_Omni.encoders.encoder_class import encoder_class
from pMTnet_Omni.classifier import pMHCTCR
from pMTnet_Omni.background_tcr_loaders import background_species_tcr_dataset_class, background_species_tcr_dataloader_class
from pMTnet_Omni.utilities import read_file, get_auroc, plot_roc_curve, batchify_check_size, setup_seed, partition_dataframe, check_data_dir_structure


class pMTnet_Omni_class:
    def __init__(self,
                 model_device: Optional[str] = None,
                 data_dir: str = "./data/",
                 seed: Optional[int] = None) -> None:
        """Initialize pMTnet_Omni_class object

        Parameters
        ---------
        model_device: str
            The device to use. Typically it would be cpu or cuda:0
        data_dir: str
            The directory containing data needed for model setups
        seed: Optional[int]
            The seed to be used 

        """
        # Set model device 
        if model_device is None:
            model_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else: 
            self.model_device = torch.device(model_device)

        encoder_path, classifier_path = check_data_dir_structure(data_dir)

        self.background_data_dir = os.path.join(data_dir, "background_tcrs/") 
        self.encoder_data_dir = os.path.join(data_dir, "data_for_encoders/")
        # Set seed
        if seed is not None:
            setup_seed(seed=seed)

        # User df & embeddings
        self.user_df_dict = {"complete": {},
                             "alpha missing": {},
                             "beta missing": {}}
        self.user_tcr_embedding_dict = {"complete": {},
                                        "alpha missing": {},
                                        "beta missing": {}}
        # As pmhc won't have missingnesses
        self.user_pmhc_embedding_dict = {"complete": {}}

        # Create the encoder
        
        vGdVAEacheckpoint_path = encoder_path[0]
        vGdVAEbcheckpoint_path = encoder_path[1]
        cdr3VAEacheckpoint_path = encoder_path[2]
        cdr3VAEbcheckpoint_path = encoder_path[3]
        pMHCcheckpoint_path = encoder_path[4]

        self.encoder = encoder_class(encoder_data_dir=self.encoder_data_dir,
                                     model_device=model_device,
                                     vGdVAEacheckpoint_path=vGdVAEacheckpoint_path,
                                     vGdVAEbcheckpoint_path=vGdVAEbcheckpoint_path,
                                     cdr3VAEacheckpoint_path=cdr3VAEacheckpoint_path,
                                     cdr3VAEbcheckpoint_path=cdr3VAEbcheckpoint_path,
                                     pMHCcheckpoint_path=pMHCcheckpoint_path)

        # Initialize th clssifier model
        self.classifier_model = pMHCTCR().to(model_device)
        classifier_checkpoint_path = classifier_path

        # Load the model
        print("Attempt to load the classifier model\n")
        if classifier_checkpoint_path is not None:
            classifier_checkpoint = torch.load(
                classifier_checkpoint_path, map_location=model_device)
            self.classifier_model.load_state_dict(classifier_checkpoint['net'])
            print("Success\n")
        else:
            print("Check point is not provided. Use random weights\n")

    def read_user_df(self,
                     user_data_path: str,
                     sep: str = ",",
                     header: int = 0,
                     partition_size=7000) -> None:
        """Read in user dataframe

        Parameters
        ----------
        user_data_path: str
            The path to the user dataframe 
        sep: str
            The sep argument used in pd.read_csv() function
        header: int
            The header used in pd.read_csv() function


        """
        user_df = read_file(file_path=user_data_path,
                            sep=sep,
                            header=header,
                            background_tcr_dir=self.encoder_data_dir,
                            mhc_path=os.path.join(self.encoder_data_dir, "valid_mhc.txt"))
        user_df['logit'] = np.inf
        self.user_df_dict['complete'] = partition_dataframe(
            df=user_df, partition_size=partition_size)

    def encode_user_df(self,
                       verbose: bool = True) -> None:
        """Encode pMHC and TCRs provided in the user dataframe 

        Parameters
        ---------
        verbose: bool
            Whether or not to print out model process 

        """
        for partition in self.user_df_dict['complete']:
            tcr_embedding, pmhc_embedding = self.encoder.encode(df=self.user_df_dict['complete'][partition],
                                                                is_embedding=False,
                                                                verbose=verbose)
            self.user_tcr_embedding_dict['complete'][partition] = tcr_embedding.to(
                'cpu')
            self.user_pmhc_embedding_dict['complete'][partition] = pmhc_embedding.to(
                'cpu')

    def create_incomplete_data(self) -> None:
        """Create incomplete data based on the user dataframe 
        This function is only supposed to be used during validation.
        The function will duplicate the original dataframe twice. 
        For the first duplicate, all tcr information on the alpha chain will be removed.
        For the second duplicate, all tcr information on the beta chain will be removed. 

        """
        for partition in self.user_df_dict['complete']:
            # Zero out TCR Alpha chain
            # Manipulate the dataframe
            # not necessary but might be useful when debugging
            df_alpha_missing = deepcopy(
                self.user_df_dict['complete'][partition])
            df_alpha_missing['va'] = ""
            df_alpha_missing['vaseq'] = ""
            df_alpha_missing['cdr3a'] = ""
            self.user_df_dict['alpha missing'][partition] = df_alpha_missing
            # Manipulate tcr embedding
            tcr_embedding_alpha_missing = self.user_tcr_embedding_dict['complete'][partition].detach(
            ).clone()
            # The embedding order is va (5) vb (5) cdr3a (30) cdr3b (30)
            tcr_embedding_alpha_missing[:, 0:5] = 0
            tcr_embedding_alpha_missing[:, 10:40] = 0
            self.user_tcr_embedding_dict['alpha missing'][partition] = tcr_embedding_alpha_missing

            # Zero out TCR Beta chain
            # Manipulate the dataframe
            # not necessary but might be useful when debugging
            df_beta_missing = deepcopy(
                self.user_df_dict['complete'][partition])
            df_beta_missing['vb'] = ""
            df_beta_missing['vbseq'] = ""
            df_beta_missing['cdr3b'] = ""
            self.user_df_dict['beta missing'][partition] = df_beta_missing
            # Manipulate tcr embedding
            tcr_embedding_beta_missing = self.user_tcr_embedding_dict['complete'][partition].detach(
            ).clone()
            # The embedding order is va (5) vb (5) cdr3a (30) cdr3b (30)
            tcr_embedding_beta_missing[:, 5:10] = 0
            tcr_embedding_beta_missing[:, 40:70] = 0
            self.user_tcr_embedding_dict['beta missing'][partition] = tcr_embedding_beta_missing

    def get_user_df_output(self) -> None:
        """Predict using the dataframe
        Use the classifier model to produce logits of records in the dataframe 

        """
        for missingness in self.user_df_dict:
            if len(self.user_df_dict[missingness]) == 0:
                continue
            for partition in self.user_df_dict[missingness]:
                df_output = self.classifier_model.predict(
                    torch.cat((self.user_pmhc_embedding_dict['complete'][partition],
                               self.user_tcr_embedding_dict[missingness][partition]), dim=1).to(self.model_device)
                )
                self.user_df_dict[missingness][partition]['logit'] = df_output

    def _generate_background_dataloader(self,
                                        load_embedding: bool,
                                        replacement: bool,
                                        check_size: list,
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

    def predict(self,
                rank_threshold: float = 0.03,
                B: int = 1,
                load_embedding: bool = True,
                replacement: bool = True,
                check_size: list = [1000, 10000, 100000],
                load_size: int = 1000000,
                minibatch_size: int = 50000,
                log_file_path: Optional[str] = None) -> None:
        """Compute the rank percentiles 

        Parameters
        ----------
        rank_threshold: float
            The rank percentile threshold greater than which further varification is NOT conducted 
        B: int 
            Number of trials 
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
        log_file_path: Optional[str]
            The file path to save the output log file

        """
        if log_file_path is None:
            log_file_path = os.devnull
            print("Log file will NOT be saved as no path is specified.\n") 
        elif os.path.splitext(log_file_path)[1] != ".txt":
            raise Exception("Only .txt file is used for the log file.\n")
        
        with open(log_file_path, 'w') as f:
            with redirect_stdout(f):
                # Make sure check size is in an ascending order
                check_size = np.sort(check_size)
                for b in range(B):
                    print("Starting trial "+str(b+1) +
                          " out of "+str(B)+" total tries\n")
                    print("Loading background TCR datasets\n")
                    batch_size, batch_finished_indicator, background_human_dataloader, background_mouse_dataloader = self._generate_background_dataloader(load_embedding=load_embedding,
                                                                                                                                                          replacement=replacement,
                                                                                                                                                          check_size=check_size,
                                                                                                                                                          load_size=load_size,
                                                                                                                                                          minibatch_size=minibatch_size)
                    for missingness in self.user_df_dict:
                        if len(self.user_df_dict[missingness]) == 0:
                            continue
                        for partition in self.user_df_dict[missingness]:
                            n_rows = self.user_df_dict[missingness][partition].shape[0]
                            self.user_df_dict[missingness][partition]["rank_" +
                                                                      str(b)] = np.inf
                            tcr_species_df = self.user_df_dict[missingness][partition]["tcr_species"].to_frame(
                            )

                            for row in tqdm(range(n_rows), position=0, leave=True):
                                # Depending on the species, we use their corresponding dataloader
                                if tcr_species_df.iloc[row, 0] == "human":
                                    data_loader = background_human_dataloader
                                else:
                                    data_loader = background_mouse_dataloader
                                print(
                                    self.user_df_dict[missingness][partition].iloc[[row], :])
                                check_size_ind = 0
                                user_output_position = 0
                                b_size_so_far = 0
                                for batch_size_ind, tcrs in enumerate(data_loader):
                                    b_size = batch_size[batch_size_ind]
                                    b_size_so_far += b_size
                                    print(
                                        "Verifying using "+str(check_size[check_size_ind]) + " background tcrs\n")
                                    print("\tLoading minibatch of " +
                                          str(tcrs.shape[0]) + " tcrs\n")
                                    #####################################
                                    # DEPENDING ON WHETHER THE USER WANTS TO LOAD EMBEDDING OR NOT
                                    # THIS PART OF THE CODE WILL CHANGE
                                    print("\tEncoding background tcrs\n")
                                    background_tcr_embedding, _ = self.encoder.encode(
                                        df=tcrs, is_embedding=load_embedding)
                                    duplicated_pmhc_embedding = self.user_pmhc_embedding_dict['complete'][partition][row, :].repeat(
                                        b_size, 1
                                    )
                                    #####################################
                                    print("\tPredicting background tcrs\n")
                                    background_result = self.classifier_model.predict(
                                        torch.cat((duplicated_pmhc_embedding.to(self.model_device), background_tcr_embedding), dim=1))
                                    # Compute how many backgrounds are "better" than user output
                                    # Therefore, the closer the value is to 0
                                    # the more likely that the model thinks the pair will bind
                                    current_position = np.sum(
                                        background_result > self.user_df_dict[missingness][partition]['logit'][row])
                                    print("\tRank in the current minibatch is " +
                                          str(current_position+1)+"th out of "+str(b_size+1)+"\n")
                                    user_output_position += current_position
                                    print("\tRank in the current batch so far is " +
                                          str(user_output_position+1)+"th out of "+str(b_size_so_far+1)+"\n")
                                    if batch_finished_indicator[batch_size_ind]:
                                        rank_percentile = user_output_position / \
                                            (check_size[check_size_ind]+1)
                                        print("\tCurrent batch is done. The final position is "+str(
                                            user_output_position+1)+" out of "+str(check_size[check_size_ind]+1)+"\n")
                                        if (rank_percentile > rank_threshold) or (b_size == np.max(batch_size)):
                                            # If the threshold is 0.03 and the rank_percentile
                                            # is greater than that, this means that more than
                                            # 3% of the background is 'better'
                                            self.user_df_dict[missingness][partition].at[row, 'rank_'+str(
                                                b)] = rank_percentile
                                            break
                                        user_output_position = 0
                                        b_size_so_far = 0
                                        check_size_ind += 1

    def amalgamate(self) -> Tuple[dict, dict, dict]:
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
            df = [self.user_df_dict[missingness][partition]
                  for partition in self.user_df_dict[missingness]]
            df = pd.concat(df, axis=0, ignore_index=True)
            df_dict[missingness] = df
            # TCRS
            tcrs = [self.user_tcr_embedding_dict[missingness][partition] for partition in self.user_tcr_embedding_dict[missingness]]
            tcrs = torch.cat(tcrs, dim=0)
            tcr_dict[missingness] = tcrs
            # pMHC
            # We propagate across missingness as we assume users use this function for 
            # validation, visulation, etc
            pmhcs = [self.user_pmhc_embedding_dict['complete'][partition] for partition in self.user_pmhc_embedding_dict['complete']]
            pmhcs = torch.cat(pmhcs, dim=0)
            pmhc_dict[missingness] = pmhcs

        return df_dict, tcr_dict, pmhc_dict

    def validate(self,
                 true_labels: Union[pd.DataFrame, np.ndarray, list],
                 roc_plot_path: str = "./roc_plots.pdf",
                 b: int = 0) -> dict:
        """Compute aurocs and plot roc curves 

        Parameters
        ----------
        true_labels: Union[pd.DataFrame, np.ndarray, list]
            A one column pandas dataframe containing the true labels. We assume 1 is binding and 0 is not binding
        roc_plot_path: str
            The file path to save the roc curves 

        Returns
        ---------
        dict: 
            A dictionary containing the aurocs and their breakdowns 

        """
        result_dfs = {key: None for key in self.user_df_dict}

        # For each missingness type, we also compute different auroc types
        auroc_to_compute = ["overall", "human overall", "mouse overall",
                            "human class i", "human class ii",
                            "mouse class i", "mouse class ii"]

        if isinstance(true_labels, list) or isinstance(true_labels, np.ndarray):
            true_labels = np.array(true_labels)
        else:
            # We assume that true_labels is a one-column dataframe
            true_labels = true_labels.iloc[:, 0]
        truth = pd.DataFrame({'true_label': true_labels})
        df_dict, _, _ = self.amalgamate()
        pp = PdfPages(roc_plot_path)
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
            pred = (1-df_dict[missingness]['rank_'+str(b)]).to_frame()

            # Then we concatenate truth, predicted and mhc_class into one dataframe
            df = pd.concat([truth, pred.set_index(truth.index),
                           mhc_class.set_index(truth.index)], axis=1)

            auc_dict = {key: [None] for key in auroc_to_compute}

            for auc_type in auroc_to_compute:
                if auc_type == "overall":
                    # If the column contains anyting
                    ind = df['mhc_class'].str.contains('.')
                elif auc_type == "human overall":
                    ind = df['mhc_class'].str.contains('human')
                elif auc_type == "mouse overall":
                    ind = df['mhc_class'].str.contains('mouse')
                else:
                    ind = df['mhc_class'].str.contains(auc_type)

                if np.sum(ind) < 1:
                    continue

                auc = get_auroc(
                    true_labels=df[ind]['true_label'], predictions=df[ind]['rank_'+str(b)])
                auc_dict[auc_type][0] = auc
                pp.savefig(plot_roc_curve(true_labels=df[ind]['true_label'],
                                          predictions=df[ind]['rank_'+str(b)],
                                          label=missingness+" "+auc_type+": "+str(auc)))

            # Convert the results to dataframe
            result_dfs[missingness] = pd.DataFrame(auc_dict)
        pp.close()
        return result_dfs
