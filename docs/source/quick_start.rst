Quick Start Guide
==============
So I hear your week has been hectic. 
No worries. In this tutorial, we will walk through the basic usage of 
**pMTnet Omni** with minimum configuration. 
If you are truly swamped, we recommend `our online tool <http://lce-test.biohpc.swmed.edu/pmtnet>`_.

.. note:: 
    Before proceeding, make sure you have all the dependencies and the package 
    installed in your environment. 

    Please also make sure that you have downloaded the supporting data.  

    If not, please check out :doc:`installation_guide`. 

    We will assume that you already have the environment set up. 

.. note::
    For this tutorial, we will assume that the file path to the data directory **"./data"**

    We will assume that the file path to the user data file is **"./df.csv"**

.. note::
    Make sure your data file is structured like the following:

    .. image:: ./images/sample_df.png
        :width: 600

    For a more detailed instruction on the data format, please check out :doc:`tutorial/input_file`. 


CLI (Command Line Interface)
-----------------
By using CLI, you only need one line of code. 

.. code:: bash 

    python -m pMTnet_Omni --data_dir "./data" --user_data_path "./df.csv" --output_file_path "./df_results.csv"


Interactive Python 
------------------
Via interactive Python, you will need more steps to get your final results. 
However, you get to inspect the results of each step of the model output.

Initialize the pMTnet_Omni_class 
~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python 

    # Import pMTnet Omni clas 
    from pMTnet_Omni.pMTnet_Omni_class import pMTnet_Omni_class

    # Initialize the class 
    pmtnet = pMTnet_Omni_class(data_dir="./data")

User Data Preprocessing 
~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python 
    
    # Read user data 
    pmtnet.read_user_df(user_data_path="./df.csv")

    # Create embedding for user data 
    pmtnet.encode_user_df()

Model Prediction 
~~~~~~~~~~~~~~~~
.. code-block:: python 

    # Get model raw output 
    pmtnet.get_user_df_output()

    # Final predcition 
    pmtnet.predict()

Save Prediction Results
~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python 

    # Combine all the partitions 
    df_dict, _, _ = pmtnet.amalgamate()

    # Save the result 
    df_dict['complete'].to_csv("./df_results.csv", index=False)








