.. pMTnet_Omni documentation master file, created by
   sphinx-quickstart on Wed Dec 14 11:36:34 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ../../assets/logo.png
   :width: 600
   :align: center

**pMTnet Omni**: pan-MHC and cross-Species Prediction of T Cell Receptor-Antigen Binding
============================================================================================

.. warning:: 
   This document is only for internal reference. 

**pMTnet Omni** is a deep learning algorithm for affinity prediction based on 
TCR Va, Vb, CDR3a, CDR3b sequences, peptide sequence, and MHC allele types.
The predictions can be made for human and mouse alleles, 
and for both CD8 T cells/MHC class I and CD4 T cells/MHC class II.

Here is a quick overview of the structure of the model:

.. image:: ./images/overview.png
   :width: 600
   :align: center

For a more detailed exploration of our model, please refer to `our paper <https://doi.org/10.1101/2023.12.01.569599>`_:

.. code-block:: text

    @article {Han2023.12.01.569599,
      author = {Yi Han and Yuqiu Yang and Yanhua Tian and Farjana J. Fattah and Mitchell S. von Itzstein and Minying Zhang and Xiongbin Kang and Donghan M. Yang and Jialiang Liu and Yaming Xue and Chaoying Liang and Indu Raman and Chengsong Zhu and Olivia Xiao and Yifei Hu and Jonathan E. Dowell and Jade Homsi and Sawsan Rashdan and Shengjie Yang and Mary E. Gwin and David Hsiehchen and Yvonne Gloria-McCutchen and Ke Pan and Fangjiang Wu and Don Gibbons and Xinlei Wang and Cassian Yee and Junzhou Huang and Alexandre Reuben and Chao Cheng and Jianjun Zhang and David E. Gerber and Tao Wang},
      title = {pan-MHC and cross-Species Prediction of T Cell Receptor-Antigen Binding},
      elocation-id = {2023.12.01.569599},
      year = {2023},
      doi = {10.1101/2023.12.01.569599},
      publisher = {Cold Spring Harbor Laboratory},
      URL = {https://www.biorxiv.org/content/early/2023/12/12/2023.12.01.569599},
      eprint = {https://www.biorxiv.org/content/early/2023/12/12/2023.12.01.569599.full.pdf},
      journal = {bioRxiv}
   }

Our GitHub repo can be found `here <https://github.com/Yuqiu-Yang/pMTnet_Omni>`_

The online tool is hosted on `DBAI <https://dbai.biohpc.swmed.edu/pmtnet/>`_

Please follow `this link <https://365utsouthwestern-my.sharepoint.com/:u:/r/personal/yuqiu_yang_utsouthwestern_edu/Documents/pMTnet_Omni_datafile/data.tgz?csf=1&web=1&e=YdNTsn>`_ 
to download the supporting data. 

Dependencies
--------------

* python>=3.9, <3.11
* numpy==1.22.4
* pandas==1.5.2
* matplotlib==3.6.2
* scikit-learn==1.0.2
* tqdm==4.64.1
* torch==1.13.1

Quick (but sloppy) Installation
--------------------------------

.. warning:: 
   We will NOT publish this package on PyPI. Therefore, this won't 
   work. We left it here mainly for demonstration purposes. 

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
   online_tool_guide

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
