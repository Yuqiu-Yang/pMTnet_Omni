pMTnet Omni Tutorial
====================
So I see you want to get into the nitty gritty details of our package. While our :doc:`../api_reference/index`
provides all the details of the functions and classes defined in the package, 
it could be a bit overwhelming. In this relatively detailed tutorial, we will 
provide an in-depth exploration of how the package works. 

.. note:: 
    We will assume that the path to the directory containing all the background 
    TCRs as well as other supporting data is `./data`

    And the user has already imported the class using the command:

    .. code-block:: python
        
        from pMTnet_Omni.pMTnet_Omni_class import pMTnet_Omni_class

Instantiate a **pMTnet_Omni_class** Object
--------------------------------------------
It's relatively straightforward to instantiate a **pMTnet_Omni_class** object.

* Default 
  
  .. code-block:: python 

    ###############################################
    # Use the default setting:
    # pMTnet_Omni will try cuda:0 if it's available 
    # Otherwise, it will use cpu
    pmtnet = pMTnet_Omni_class(data_dir="./data")
* CPU
  
  .. code-block:: python
    
    # CPU:
    pmtnet = pMTnet_Omni_class(model_device="cpu",
                               data_dir="./data")
* GPU 
  
  .. code-block:: python 
    
    # GPU:
    pmtnet = pMTnet_Omni_class(model_device="cuda:0",
                               data_dir="./data")

* Seed 
  
  .. code-block:: python 

    #To set seed to ensure reproducible results
    pmtnet = pMTnet_Omni_class(data_dir="./data",
                               seed=42)

The entire process will take several seconds as **pMTnet_Omni_class** will 
load the models and build necessary dictionaries for future usage. 

Import and Encode Data 
------------------------
As the `read_user_df` method calls the `read_csv` function in the `pandas` package,
the first three arguments: *user_data_path*, *sep*, and *header* are fairly standard. 
The final argument `partition_size`, however, worth some explanation. 

During our internal testing, we found that while a mediocre GPU is capable 
of handling several to hundreds of pairs, for dataset containing thousands of 
TCR-pMHC pairs, we usually will run out of memory. Partitioning the original 
data frame is a simple yet effective remedy. 

.. image:: ../images/partitions.png
    :width: 600

.. note:: 
    We will assume that the path to the user's data is `./df.csv`, and 
    it contains 5000 TCR-pMHC pairs 

By calling 

.. code-block:: python 

    pmtnet.read_user_df(user_data_path="./df.csv",
                        sep=",", header=0,
                        partition_size=2000)

**pMTnet_Omni_class** will partition the original data frame into three partitions with 
2,000, 2,000, and 1,000 pairs, respectively.

When this command 

.. code-block:: python 

    pmtnet.encode_user_df()

is called, and if the user is using GPU, **pMTnet_Omni_class** will transfer
each partition to GPU, compute the TCR embedding and the pMHC embedding, and 
transfer the results back to CPU. Therefore, only the model and one partition 
of the original data will live on GPU. 

Predict TCR-pMHC Affinity
---------------------------
Two steps are needed for our model to produce the final result.

The first step is quite straightforward:

.. code-block:: python

    pmtnet.get_user_df_output()

This will yield the *raw* affinity score estimated by our classifier. 

In the second step, we will compare each pair with the background TCRs. As 
we have millions of background TCRs, it would be time consuming to check 
each pair against the entire database. Hence, in our implementation, we borrowed 
an idea from the literature of clinical trials.

Each TCR-pMHC pair will undergo one or several ``trials" each with a 
sequence of `checks`. The procedure is conceptually simple: 

Within a ``trial", each TCR-pMHC pair will be first checked against a small 
subset of the background TCRs. If the resulting rank is high enough, we sample 
a larger subset of the background TCRs and check the given pair against them. 
The process repeats until either the pair falls out of the top rank list or 
it has been validated against enough background TCRs, at which point, the 
algorithm reports the final rank. 

.. image:: ../images/prediction.png
    :width: 600

With that in mind, the final affinity prediction can be carried out via 

.. code-block:: python 

    pmtnet.predict(rank_threshold=0.03, # We only consider pairs within top 3%
                   B=1, # Only one trial will be carried out 
                   load_embedding=True, # When loading the background TCRs, we will load their embeddings instead of the raw sequences 
                   replacement=True, # When sampling from background TCRs, we sample with replacement 
                   check_size=[10000, 100000, 1000000], # We will first check against 10,000 and then 100,000 and then 1,000,000 background TCRs
                   load_size=1000000, # Each trial will involve at most 1,000,000 background TCRs from each set
                   minibatch_size=50000, # Each minibatch will involve 50,000 background TCRs
                   log_file_path=None # No log file is needed 
                   )

Two parameters ``load_size" and ``minibatch_size" could be somewhat confusing. But they 
are implemented to further speed up the prediction process. 

**load_size** is implemented so that for each trial, only that many background TCRs will 
be potentially used. This is because the background TCRs datasets are relatively large, 
meaning that initializing the dataloaders will be time consuming. 

**minibatch_size** is how many background TCRs the dataloader will sample within a check_size.
For example, is the current check_size is 2,000 and the minibatch_size is 1,000. Then 
the dataloader will first load 1,000 TCRs, compute the rank, load another 1,000 TCRs, and 
update the rank. This will speed up the process as directly load, say 1,000,000 TCRs will 
be slow. 

Sample Output
--------------
To get the final output, we just need to ``glue" all the partitions back together

.. code-block:: python 

    # We just need the data dictionary
    # The other two are TCR and pMHC embedding dictionaries
    df_dict, _, _ = pmtnet.amalgamate()
    # The other two keys are only intended for validation 
    df_dict['complete'].to_csv("./df_results.csv", index=False)

.. image:: ../images/sample_output.png
    :width: 600
