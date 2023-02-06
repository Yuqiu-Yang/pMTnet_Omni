Amino Acids Sequences Format
=============================
Columns that **pMTnet Omni** expect to contain sequences of 
amino acids are 

.. list-table:: Columns of Amino Acids (AA) Sequences
   :align: center 
   :widths: 30 70 
   :header-rows: 1

   * - Name 
     - Meaning 
   * - vaseq 
     - The AA sequence the Alpha chain for the V segment
   * - vbseq
     - The AA sequence the Beta chain for the V segment
   * - cdr3a
     - The sequence of amino acids for the CDR3 region on the Alpha chain
   * - cdr3b
     - The sequence of amino acids for the CDR3 region on the Beta chain
   * - peptide 
     - The sequence of amino acids presented by the MHC
   * - mhcseq
     - The sequence(s) of amino acids of the corresponding MHC

Overall 
-----------
The **one-letter code** of the following 20 Amino Acids are 
accepted by **pMTnet Omni**. 

.. image:: ../images/CDR_letters.png
   :align: center
   :width: 600 

.. warning:: 
   If a letter in a provided sequence is not of the 20 amino acids
   (this includes cases such as actual letters\: ``X``, 
   white spaces\: :literal:`\ `, special characters\: ``_, +, /`` etc.),  
   it will first be replaced by ``_``. When converting to a matrix of 
   Atchley Factors, those ``_`` will be interpreted as ``0,0,0,0,0``.

vaseq vbseq 
---------------
When making affinity predictions, **pMTnet Omni** actually uses values contained in 
these two columns instead of their names contained in ``va`` and ``vb`` columns.

.. note:: 
    If only ``va`` (resp. ``vb``) is provided, we will perform a 
    look-up using our `reference database` provided `here <http://lce-test.biohpc.swmed.edu/pmtnet>`_. 
    However, 


Therefore, it's vital to make sure that your input sequences 


cdr3a cdr3b 
---------------
.. warning:: 
    CDR3s usually start from C and end with F. 
    We are aware of different definitions of CDR3s that result 
    in slightly different start and end boundaries of CDR3s. 
    Please use the definition that is consistent with ours

peptide 
----------------
.. warning:: 
   Any AA sequence in the ``peptide`` column that contains 
   **more than** 30 amino acids will be dropped.

mhcseq 
----------------






