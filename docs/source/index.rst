.. pMTnet_Omni documentation master file, created by
   sphinx-quickstart on Wed Dec 14 11:36:34 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ../../assets/logo.png
   :width: 600

**pMTnet Omni**: your one-stop TCR-pMHC affinity prediction algorithm
===========================================================================

**pMTnet Omni** is a deep learning algorithm for affinity prediction based on 
TCR Va, Vb, CDR3a, CDR3b sequences, peptide sequence, and MHC allele types.
The predictions can be made for human and mouse alleles, 
and for both CD8 T cells/MHC class I and CD4 T cells/MHC class II.

Here is a quick overview of the structure of the model:

.. image:: ./images/overview.png
   :width: 600
   :align: center

For a more detailed exploration of our model, please refer to `our paper <https://www.google.com>`_:

*pMTnet Omni Paper citation here*

Our GitHub repo can be found `here <https://github.com/Yuqiu-Yang/pMTnet_Omni>`_

The supporting data as well as the online tool are hosted on `DBAI <http://lce-test.biohpc.swmed.edu/pmtnet>`_

Dependencies
--------------

* python>=3.9
* numpy==1.22.4
* pandas==1.5.2
* matplotlib==3.6.2
* scikit-learn==1.0.2
* tqdm==4.64.1
* torch==1.13.1
* fair-esm==2.0.0

Quick (but sloppy) Installation
--------------------------------

.. note:: 
   This will install *pMTnet_Omni* to your `base` environment. 
   For a more in-depth instruction, check out our :doc:`installation_guide`. 

.. code:: bash
   
   pip install pMTnet_Omni


User Guide / Tutorial
=====================
.. toctree::
   :maxdepth: 1
   :caption: Get Started

   installation_guide
   quick_start

.. toctree::
   :maxdepth: 3
   :caption: Input Format

   input_format/index

.. toctree::
   :maxdepth: 1
   :caption: Guided Tutorial

   tutorial/input_check
   tutorial/details

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
