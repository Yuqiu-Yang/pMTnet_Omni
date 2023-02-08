VA, VB Format 
====================
Although when ``vaseq`` and ``vbseq`` are provided (for their format 
requirements, please refer to :doc:`amino_acids_seq`), 
these two columns ``va`` and ``vb`` are not used by our 
subsequent algorithm, we expect that it would be more 
convenient for the users to supply the names. 

In this section, we provide some general guidelines 
on the nomenclatures used in **pMTnet Omni**. Before 
you get started: 

.. warning:: 
    Make sure that the information provided in the ``tcr_species`` column
    is compatible with the corresponding information in the 
    ``va`` and ``vb`` column. 

IMGT Nomenclature
-------------------------
During our data curation, we found that there are 
at least three nomenclatures for TCRs. We choose 
to use the nomenclature adopted by 
*The International Immunogenetics Information System* 
(`IMGT`_). 

.. _IMGT: https://www.imgt.org/

.. warning:: 
    Make sure that your nomenclature conforms with 
    the IMGT nomenclature. For correspondence among 
    different nomenclatures, you can refer to 
    `this website`_.

.. _this website: https://www.imgt.org/IMGTrepertoire/LocusGenes/#J

That being said, we still feel it could be helpful 
to provide a checklist to overcome some 
common "issues" we encountered:

.. |check| raw:: html

    <input checked=""  type="checkbox">

.. |check_| raw:: html

    <input checked=""  disabled="" type="checkbox">

.. |uncheck| raw:: html

    <input type="checkbox">

.. |uncheck_| raw:: html

    <input disabled="" type="checkbox">



|uncheck| ``va`` should start with ``TRAV`` 

|uncheck| ``vb`` should start with ``TRBV`` 

|uncheck| No multiple TCRs in a string 

|uncheck| Replace all ``.`` with ``-``

|uncheck| Replace all ``:`` with ``*``

|uncheck| Strip off all whitespaces\: :literal:`\ `

|uncheck| Change names like ``TRAV01-01`` to ``TRAV1-1``

|uncheck| If there is not allele, still append ``*01`` to the string

Although we perform basic data curation while reading the user input, 
it's nearly impossible for us to cover all corner cases. Therefore, we 
strongly recommend you to check your input format before preceding to 
using the main algorithm.

.. list-table:: Sample Input 
   :align: center 
   :widths: 50 50
   :header-rows: 1

   * - va
     - vb
   * - TRAV19*01
     - TRBV9*01
   * - TRAV7-3*04
     - TRBV1*01
  
.. note:: 
  If you are still not sure whether or not the information 
  you supplied conforms with our standard, we also 
  provided some rudimentary functionalities to help you.
  Please refer to :doc:`/tutorial/input_check` where we guide 
  you through the process. 