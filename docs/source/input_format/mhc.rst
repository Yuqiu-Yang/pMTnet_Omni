MHC Format
===================
We single out the requirements for MHC format used in 
the ``mhc`` column as we are aware that there are multiple 
nomenclatures in this field. 
In general, the format expected by our algorithm is similar 
to what you will find on the *Immune Epitope Database And
Analysis Resource* website (`IEDB`_). For the requirements on 
the MHC sequences in the ``mhcseq`` column, please refer to 
:doc:`amino_acids_seq`. 

.. _IEDB: https://www.iedb.org/

Overall
-------------
Our algorithm uses information provided in the ``mhc`` column 
in conjunction with the ``pmhc_species`` column to determine the 
MHC classes as well as other information. Therefore, before you 
get started:

.. warning:: 
    Make sure that the information provided in the ``pmhc_species`` column
    is compatible with the corresponding information in the 
    ``mhc`` column. Otherwise, the program will raise an exception.

Prefix 
--------------------
Our program is quite flexible on the input format of the names 
of MHCs. It is completely acceptable if you want to prepend 
``HLA-`` to a human HLA (or ``H2-`` a mouse MHC) or ignore that 
prefix. It's even fine if you have whitespaces\: :literal:`\ ` some where in the 
string or use ``H-2-`` instead of ``H2-``. 

.. list-table:: Sample Input 
    :align: center
    :widths: 90 10
    :header-rows: 1

    * - Input Examples 
      - Acceptable
    * - HLA-A*01:01; HLA A*01:01; HLA- A*01:01; HLAA*01:01; A*01:01
      - Yes 
    * - Human A*01:01; MHC-A*01:01
      - No
    * - H2-Db; H-2-Db; H2 Db; Db; H2Db
      - Yes 
    * - Mouse-Db; MHC-Db
      - No 

Details
-------------
We have already computed the ESM embeddings of around 20,000 MHCs. Since 
they are stored using key-value pairs, if a value provided in the ``mhc``
column can not find an exact match, our algorithm will assume that it is 
a "new" MHC and invoke the ESM2 algorithm to encode the 
corresponding sequences. This could significantly slow down the entire process.
Or the program will halt and throw an error if the sequences are not provided. 

.. admonition:: mhc requirements

    Human Class I 
        One name for the HLA would suffice. Our program will use input value 
        as the name for the Alpha chain and impute the Beta chain using
        ``human_microglobulin``.

    Human Class II HLA that starts with DP or DQ
        Names for **both** chains should be provided. The format 
        we assume is ``MHC Alpha`` followed by a forward 
        slash ``/``, which is then followed by ``MHC Beta``.

    Human Class II HLA that starts with DR
        There are two possible scenarios that we take into account. 
        If both the user provided information on **both** chains, then 
        the inference method follows that of the HLA DP and DQ. On the 
        other hand, if only the information on Beta chain is supplied,
        then our program use the input value as the name for the Beta 
        chain and impute the Alpha chain as ``DRA*01:01``.

    Mouse Class I
        One name for the MHC would suffice. Our program will use input value 
        as the name for the Alpha chain and impute the Beta chain using
        ``mouse_microglobulin``.

    Mouse Class II 
        One name for the MHC would suffice. Our program will use the 
        input value appended with ``_alpha`` as the name for the 
        Alpha chain and the input value appended with ``_beta`` 
        as the name for the Beta chain 

.. list-table:: Sample Input 
   :align: center 
   :widths: 50 50
   :header-rows: 1

   * - Class
     - mhc 
   * - Human Class I
     - A*01:01
   * - Human Class II: Only DRB
     - DRB1*01:01
   * - Human Class II: DRA and DRB
     - DRA*01:01/DRB1*01:01
   * - Human Class II: DP 
     - DPA1*04:02/DPB1*01:01
   * - Human Class II: DQ
     - DQA1*06:04/DQB1*02:07
   * - Mouse Class I 
     - H-2-Db
   * - Mouse Class II 
     - H-2-IAk

.. note:: 
  If you are still not sure whether or not the information 
  you supplied conforms with our standard, we also 
  provided some rudimentary functionalities to help you.
  Please refer to :doc:`/tutorial/input_check` where we guide 
  you through the process.  



