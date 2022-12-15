.. pMTnet_Omni documentation master file, created by
   sphinx-quickstart on Wed Dec 14 11:36:34 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ../../assets/logo.png
   :width: 600

pMTnet Omni: your one-stop pMHC-TCR binding prediction algorithm
=======================================
.. note::
   The logo image is from Dr. Tao Wang's twitter

**pMTnet Omni** is a deep learning algorithm for predicting the binding 
properties of any pair of pMHC (peptide-major histocompatibility complex) 
and TCR (T-cells receptor)

The details of the models can be found in `our paper <https://www.google.com>`_

The software can be found `on github <https://www.google.com>`_

We also host the algorithm `here <https://www.google.com>`_

.. note::
   This project and this document are under active development

**Installation**
.. note::
   This will install the package in your `base` environment which in general
   is a bad practice. 
   
For a more in-depth instruction, check out our :doc:`installation_guide`. 
Specifically, :ref:`this section <installation guide>`.

Conda install

.. code-block:: console
   conda install 

PyPI install

.. code-block:: console
   pip istall 


.. toctree::
   :maxdepth: 1
   :caption: Get Started

   installation_guide
   quick_start

.. toctree::
   :maxdepth: 1
   :caption: Guided Tutorial

   tutorial/input_file
   tutorial/load_models

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api_reference/index



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
