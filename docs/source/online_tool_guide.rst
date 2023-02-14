Online Tool Guide 
============================
We will summarize the parameters used in the package.


Step 1: Instantiate a **pMTnet_Omni_class** Object
-----------------------------------------------------------

.. list-table:: Instantiation Parameters 
    :align: center
    :widths: 10 10 20 10 10 40 
    :header-rows: 1

    * - Parameter 
      - Format 
      - Acceptable Inputs 
      - Default
      - Exposed to Users 
      - Note
    * - model_device 
      - ``str``
      - "cpu", "gpu", None
      - None
      - Yes 
      - If ``None``, choose based on gpu availability
    * - seed 
      - ``int``
      - Any integer, None
      - None 
      - Yes 
      - If ``None``, no seed is used 
    * - data_dir  
      - ``str``
      - Any string 
      - './data'
      - No
      - This should be set to the data directory on the server. 
    * - extra_mhc_dict
      - ``str``
      - A string ends in `.pickle`
      - None
      - No
      - Users will upload this file, and we will set this for them. 



Step 2: Import Data
-----------------------------------------

.. list-table:: Data Importing Parameters 
    :align: center
    :widths: 10 10 20 10 10 40 
    :header-rows: 1

    * - Parameter 
      - Format 
      - Acceptable Inputs 
      - Default
      - Exposed to Users 
      - Note
    * - partition_size 
      - ``int``
      - Any integer
      - 500
      - Yes
      - If ``model_device`` is "cpu", it's not really used
    * - user_data_path
      - ``str``
      - A string to the data path
      - No default
      - No
      - Users will upload this file, and we will set this for them. 
    * - \*\*kwargs
      - ? 
      - Extra parameters: sep, header...
      - No default 
      - Yes? 
      - We can detect the file extension and set this for them?

Step 3:  Predict TCR-pMHC Affinity
--------------------------------------------

.. list-table:: Affinity Prediction Parameters 
    :align: center
    :widths: 10 10 20 10 10 40 
    :header-rows: 1

    * - Parameter 
      - Format 
      - Acceptable Inputs 
      - Default
      - Exposed to Users 
      - Note
    * - compute_percentile_rank 
      - ``bool``
      - True, False 
      - False 
      - Yes 
      - If False, only the raw affinity scores will be reported. 
    * - rank_threshold 
      - ``float``
      - Any number between 0 and 1
      - 0.03
      - Yes
      - The rank percentile threshold greater than which further verification is NOT conducted 
    * - B
      - ``int``
      - Any positive integer
      - 1
      - Yes
      - Number of trials 
    * - check_size 
      - ``list``
      - List of positive integers 
      - [1000, 10000, 100000]
      - Yes 
      - We should set a maximum
    * - load_size
      - ``int``
      - Any positive integer
      - 1,000,000
      - No?
      - Maybe we should always use the same number 
    * - minibatch_size
      - ``int``
      - Any positive integer
      - 50,000
      - No? 
      - Maybe we should always use the same number 
    * - verbose
      - ``bool``
      - True, False 
      - True
      - No
      - Maybe use something else to entertain the users 
    * - create_incomplete_data
      - ``bool``
      - True, False 
      - False 
      - No
      - This should always be False 
    * - output_file_path
      - ``str``
      - String to output file 
      - No default 
      - No
      - We will generate the output file online. Users can download 
    * - load_embedding
      - ``bool``
      - True, False 
      - True
      - No
      - This should always be True   
    * - replacement
      - ``bool``
      - True, False 
      - True
      - No
      - This should always be True   
    * - log_file_path
      - ``str``
      - String to log file path 
      - None 
      - No
      - We will generate the output file online. Users can download 