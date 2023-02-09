Input Check 
=================
Being one of the meticulous (paranoid) ones, I myself 
when using packages, sometime prefer to peek under the hood 
just to make sure that the program is doing what I expect it 
to do. With so many instructions on :doc:`/input_format/index`,
various nomenclatures in this field, and not to mention, human 
errors, wouldn't it be nice if there are some simple ways to 
tell you "Rest assured. Everything looks fine :)"

.. note:: 
    For this purpose, along with the main program, we also 
    published 5 datasets which you can find under 
    the ``./validation_data`` folder to help you check 
    the input manually. 

    In this section, however, we will let you in how **pMTnet Omni**
    curates data so that you will have a better idea of what 
    is acceptable and what is not. 

    For the purpose of demonstration, we will forge a simple 
    dataset throughout this section. 

.. code-block:: python 

    import pandas as pd 

    # We construct a simple dataset 
    df = pd.DataFrame({'v_a': ['TRAV19:01', 'TRAV7.3*04'],
                        'cdr3_a': ['CALSXRYXGATNKLIF', 'CAVSDRGSALGRLHF'],
                        'V_B': ['TRBV9:O1', 'TRBV1:01'],
                        'cdr3_b': ['CASSPYXSSGXNVLTF', 'CSAAXGGSAXTLYF'],
                        'mhc': ['S*99:01', 'H2-Kb'],
                        'peptide': ['RINATLETK', 'VIQYFASI'],
                        'pMHC_SPECIES': ['Human', 'mouse'],
                        'TCR_SPECIES': ['human', 'Mouse']})

As you should see, there are several problems with this dataset. 

Column Names 
    The names contain ``_`` and the cases are not consistent 

VA VB
    A different convention is adopted than ours. For example, 
    ``.`` is used in place of ``-``, and ``:`` is used in place 
    of ``*``. There is a typo: ``TRVB9:O1``. 

Amino Acid Sequences 
    There is an unknown AA\: ``X``

MHC 
    There is a typo (actually two typos): ``S*99:01``

Species 
    The cases are inconsistent 


Column Names, VA, VB
--------------------------
First thing first, we have to make sure that we have all the 
columns we need. The ``check_column_names`` function contained 
in the ``utilities`` module provides this functionality. 

As a side product, we will also check 
the names of the V segments on the Alpha chain and their 
counterparts on the Beta chain. 

.. code-block:: python 

    import pandas as pd 
    # Import utilities functions 
    from pMTnet_Omni.utilities import check_column_names,\
                                      get_mhc_class,\
                                      check_data_sanity
    
    # Use the datasets in the ./validation_data 
    df = check_column_names(df=df,
                            background_tcrs_dir='./validation_data/')

Once you print out the new dataframe, you should see 

* All column names have been corrected to comply with our standard 
* Several new columns have been added, including: ``vaseq``, ``vbseq``, and ``mhcseq``
* Attempts have been make to change to naming convention of ``va`` and ``vb``   
* For all the V segments provided except for ``TRBV9:O1``, we have found their corresponding AA sequences

.. note:: 
    An ``NaN`` indicates that we are not able to find this particular
    record in our database. 

After identifying the typo, we change ``TRBV9*O1`` to ``TRBV9*01`` in the 
original dataframe. Now if you re-execute the ``check_column_names`` 
function, the ``NaN`` should go away.

MHC Classes 
------------------------
For our main algorithm to execute correctly, we need to separate 
the information on the Alpha chain and the Beta chain using what 
is provided in the ``mhc`` column. This can be done by calling 
the ``get_mhc_class`` function. 

.. warning:: 
    This function assumes that the input is the output of 
    the ``check_column_names`` function 

.. code-block:: python 

    mhc_df = get_mhc_class(df)

When you run this piece of code, it should raise an exception saying 
that "ValueError: Class of S*99:01 can not be determined". Therefore, 
in the original dataframe, change the value to ``A*99:01``, which 
still contains a typo. If we re-run the sequence, we see that this time 
the ``get_mhc_class`` guessed that ``A*99:01`` is a ``human class i`` HLA. 

MHC and AA Sequences
------------------------
However, ``A*99:01`` is not a known HLA. And we still have the 
unknown AA: ``X`` in our dataset. The ``check_data_sanity`` function 
is implemented as an attempt to address this problem. 

.. code-block:: python 

    # First we concatenate the two daraframes 
    # and drop columns that are no longer needed 
    df = pd.concat([df, mhc_class], axis=1, ignore_index=False)
    df = df.drop(['mhc', 'mhcseq'], axis=1)
    # Check data sanity 
    df, df_antigen_dropped, df_mhc_alpha_dropped, df_mhc_beta_dropped = \
        check_data_sanity(df=df,
                          mhc_path='./validation_data/valid_mhc.txt')

There are two things to notice here. First, as ``A*99:01`` can not be found in our database and no corresponding sequence 
is provided, the entire record has been dropped. Second, all the ``X``'s have 
now been replaced by ``_``. 

