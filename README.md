# pMTnet Omni: your one-stop TCR-pMHC affinity prediction algorithm :microscope:

![Logo](/assets/logo.png)

![forthebadge](/assets/tcr-pmhc.svg)
![forthebadge](/assets/deep-learning.svg)

[![Documentation Status](https://readthedocs.org/projects/tao-wang-pmtnet-omni/badge/?version=latest)](https://tao-wang-pmtnet-omni.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/Yuqiu-Yang/pMTnet_Omni/branch/main/graph/badge.svg?token=L59TPMM3VN)](https://codecov.io/gh/Yuqiu-Yang/pMTnet_Omni)

<b>pMTnet Omni</b> is a deep learning algorithm for affinity prediction based on TCR Va, Vb, CDR3a, CDR3b sequences, peptide sequence, and MHC allele types. The predictions can be made for human and mouse alleles, and for both CD8 T cells/MHC class I and CD4 T cells/MHC class II.

Please refer to our paper for more details: [pMTnet Omni paper link here](www.google.com)

We host the supporting data as well the online tool [here](http://lce-test.biohpc.swmed.edu/pmtnet), where you can find all the members of the pMTnet 
family, including [pMTnet V1](https://github.com/tianshilu/pMTnet). 

We have also built a detailed [online documentation](https://tao-wang-pmtnet-omni.readthedocs.io/en/latest/) where we walk you through every single step from installation to final affinity prediction.

## Model Overview 
![Model Overview](/docs/source/images/overview.png)

## Dependencies 
- numpy==1.22.4
- pandas==1.5.2
- matplotlib==3.6.2
- scikit-learn==1.0.2
- tqdm==4.64.1
- torch==1.13.1

## Enviroment Setup
```shell
conda env create -f pMTnet_Omni_env.yml
```

## Installation (Coming right up (not run now))
<b>PyPi</b>
```shell
conda activate pMTnet_Omni
pip install pMTnet_Omni
```
<b>Conda</b>
```shell
conda activate pMTnet_Omni
conda install pMTnet_Omni
```

## Quick Start Guide 
1. Go to [our website](http://lce-test.biohpc.swmed.edu/pmtnet) to download the supporting data to your local directory, say <i>./data</i>. 

<b>Caution: </b> the data file is around 30Gb. Make sure you have enough space on your hard drive. 

2. Prepare your dataset so that it looks somewhat like the following:
![Sample df](/docs/source/images/sample_df.png)

3. Say your dataset is under <i>./df.csv</i>. In your terminal, run 
```shell
python -m pMTnet_Omni --data_dir "./data" --user_data_path "./df.csv" --output_file_path "./df_results.csv"
```

4. An example output would look like this:
![Sample output](/docs/source/images/sample_output.png)

For a more in-depth exploration including the usage of interactive Python, check out our [online documentation](https://tao-wang-pmtnet-omni.readthedocs.io/en/latest/). 


