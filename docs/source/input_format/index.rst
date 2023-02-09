Input File Format
=======================
Our algorithm is really powerful. However, it still has not possessed human intelligence yet.
This unfortunately means that the input files need to conform with some certain format.

File Format 
---------------
Although the default format that **pMTnet Omni** expects is `.csv`, since 
under the hood, the `read_user_df` method calls the `read_csv` function from 
the `pandas` package, **pMTnet Omni** accepts inputs of major file formats, as 
long as the corresponding `sep` argument is supplied. 

.. note:: 
    By default, **pMTnet Omni** assumes that the first row is the header.

.. list-table:: Acceptable File Format 
    :widths: 50 50
    :align: center 
    :header-rows: 1

    * - File Format 
      - `sep`
    * - .csv  
      - ``,``
    * - .txt
      - User defined. Usually ``,`` or ``\t``
    * - .tsv
      - ``\t``

Column Names
---------------
As subsequent functions of **pMTnet Omni** manipulates
the input dataframe based on its column names, harmonizing 
column names is necessary. Therefore, when reading the user input,
**pMTnet Omni** will first attempt to find the following column names.

.. note:: 
    Details on the data format will be explained in :ref:`Data Format` section. 

.. list-table:: Column Name 
    :widths: 15 50 5 30
    :align: center
    :header-rows: 1

    * - Name 
      - Meaning 
      - Mandatory 
      - Note 
    * - va
      - The name of the Alpha chain for the V segment
      - No
      - At least one of ``va`` and ``vaseq`` needs to be supplied 
    * - vaseq
      - The actual sequence of amino acids of ``va``
      - No
      - At least one of ``va`` and ``vaseq`` needs to be supplied 
    * - vb
      - The name of the Beta chain for the V segment
      - No
      - At least one of ``vb`` and ``vbseq`` needs to be supplied 
    * - vbseq
      - The actual sequence of amino acids of ``vb``
      - No
      - At least one of ``vb`` and ``vbseq`` needs to be supplied 
    * - cdr3a
      - The sequence of amino acids for the CDR3 region on the Alpha chain
      - Yes 
      -  
    * - cdr3b
      - The sequence of amino acids for the CDR3 region on the Beta chain
      - Yes 
      - 
    * - peptide
      - The sequence of amino acids presented by the MHC
      - Yes 
      - 
    * - mhc
      - The name(s) of the MHC
      - Yes 
      - 
    * - mhcseq
      - The sequence(s) of amino acids of the corresponding ``mhc`` 
      - No
      - If the ``mhc`` supplied can not be found in the ESM library, then this has to be supplied 
    * - tcr_species
      - Species of the TCR
      - Yes 
      - ``human`` or ``mouse``
    * - pmhc_species
      - Species for the peptide-MHC
      - Yes 
      - ``human`` or ``mouse``

If **pMTnet Omni** can not match these names exactly, it will try
modifying the found column names in the user input to match the names. 
Specifically, it will change the names to lower cases, strip all white spaces, 
and remove special characters like ``*, _, +, -``. See the following for some 
acceptable input examples as well as some unacceptable input examples.

.. list-table:: Example Input
    :widths: 50 50
    :header-rows: 1
    :align: center

    * - Input 
      - Acceptable
    * - V_A seq
      - Yes
    * - pMHC SPECIES
      - Yes
    * - V Alpha
      - No
    * - Epitope 
      - No

.. _Data Format: 

Data Format 
---------------
We understand that the nomenclatures are not always consistent 
in the field of biology. Albeit being conceptually viable, 
it would be too overwhelming for us to take all commonly-used naming 
conventions into consideration. 

In this section, the required format for all fields in the user 
input will be elaborated upon. 

.. toctree::
   :maxdepth: 1
   :caption: Amino Acids Sequences

   amino_acids_seq

.. toctree:: 
   :maxdepth: 1
   :caption: VA, VB
   
   va_vb
   
.. toctree::
   :maxdepth: 1
   :caption: MHC
   
   mhc


