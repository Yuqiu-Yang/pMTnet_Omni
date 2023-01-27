Input File Formatting
===================
Our algorithm is really powerful. However, it still has not possessed human intelligence yet.
This unfortunately means that the input files need to conform with some certain format.

File Format 
---------------
Although the default format that **pMTnet Omni** expects is `.csv`, since 
under the hood, the `read_user_df` method calls the `read_csv` function from 
the `pandas` package, **pMTnet Omni** accepts inputs of major file formats:

* .csv 
* .txt
* .tsv

You just need to supply the corresponding `sep` argument. 

Column Names
---------------
When reading the user input, **pMTnet Omni** will attempt to modify the found 
column names in the user input to contain the following 
(The names in **bold** are mandatory):

* va
* vaseq
* vb
* vbseq
* **cdr3a**
* **cdr3b**            
* **peptide**
* **mhca**
* **mhcb**
* **tcr_species**
* **pmhc_species**

It's completely fine if the column names in your file have different cases, 
contain white spaces, and possess special characters like `*, _, +, -`. 

For example 

* V_A seq (Fine)
* pMHC SPECIES (Fine)
* V Alpha (**Not Fine**)

For columns `va` and `vaseq` (resp. `vb` and `vbseq`), at least one of 
them has to be present. We will perform a look up if only one of them is given. 

Sequence Pairs
-----------------
Once we sort out the column names, **pMTnet Omni** will also perform some rudimentary 
data sanity check. 

#. All incomplete data will be dropped 
#. Make sure the sequences in the `peptide` column contain **less than** 30 amino acids
#. Check all amino acids present in the sequences in `vaseq`, `vbseq`, `cdr3a`, `cdr3b`, and `peptide` are legitimate amino acids
#. All MHCs have their corresponding ESM embeddings

