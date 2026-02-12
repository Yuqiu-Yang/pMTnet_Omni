# pMTnet Omni: pan-MHC and cross-Species Prediction of T Cell Receptor-Antigen Binding :microscope:

![Logo](/assets/pmtnet_logo.png)

![forthebadge](/assets/tcr-pmhc.svg)
![forthebadge](/assets/deep-learning.svg)

| Package | Documentation | Code Coverage |
| --- | --- | --- |
| pMTnet Omni | | [![codecov](https://codecov.io/gh/Yuqiu-Yang/pMTnet_Omni/branch/main/graph/badge.svg?token=L59TPMM3VN)](https://codecov.io/gh/Yuqiu-Yang/pMTnet_Omni) |
| pMTnet Omni Document | [![Documentation Status](https://readthedocs.org/projects/pmtnet-omni-document/badge/?version=latest)](https://pmtnet-omni-document.readthedocs.io/en/latest/?badge=latest) | [![codecov](https://codecov.io/gh/Yuqiu-Yang/pMTnet_Omni_Document/branch/main/graph/badge.svg?token=BR1vyICN3q)](https://codecov.io/gh/Yuqiu-Yang/pMTnet_Omni_Document) |

Profiling the binding of T cell receptors (TCRs) of T cells to antigenic peptides presented by MHC proteins is one of the most important unsolved problems in modern immunology. We developed <b>pMTnet-omni</b>, an Artificial Intelligence (AI) system based on hybrid protein sequence and structure information, to predict the pairing of TCRs of &alpha;&beta; T cells with peptide-MHC complexes (pMHCs). <b>pMTnet-omni</b> is capable of handling peptides presented by both class I and II pMHCs, and capable of handling both human and mouse TCR-pMHC pairs, through information sharing enabled this hybrid design. 

Please refer to [our paper](https://doi.org/10.1101/2023.12.01.569599) for more details.

## CITATION :book:
We have uploaded our article to bioRxiv. To cite
```
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
```

## Model Overview 
![Model Overview](/docs/source/images/overview.png)


## DBAI :computer:
To try out pMTnet Omni, we recommend our online tool hosted on [DBAI](https://dbai.biohpc.swmed.edu/pmtnet/), where you can find all the members of the pMTnet 
family, including [pMTnet V1](https://github.com/tianshilu/pMTnet). 

## DIY :muscle: 
### Dependencies 
- numpy==1.22.4
- pandas==1.5.2
- matplotlib==3.6.2
- scikit-learn==1.0.2
- tqdm==4.64.1
- torch==1.13.1
- pMTnet_Omni_Document

### Enviroment Setup
> [!WARNING]
As pMTnet Omni relies on pytorch=1.13.1, please make sure that your Python version is <=3.10. 

**_NOTE:_** We have tested the package on Python=3.9 and Python=3.10. If you want to use a lower version Python, proceed with caution. 

```shell 
conda create -n pMTnet_Omni python=3.10
pip install -r requirements.txt 
```
or
```shell
conda env create -f pMTnet_Omni_env.yml
```

### INPUT FORMAT :inbox_tray:
To fully unleash the power of pMTnet Omni, you have to curate your data to conform to our acceptable input format. 

Prepare your dataset so that it looks somewhat like the following:
![Sample df](/docs/source/images/sample_df.png)

**_NOTE:_** We highly recommend using our supporting package [pMTnet_Omni_Document](https://github.com/Yuqiu-Yang/pMTnet_Omni_Document) to prepare your data. 

We have also built a detailed [online documentation](https://pmtnet-omni-document.readthedocs.io/en/latest/) where we walk you through every single step for data preparation. It also includes a step-by-step instruction on input format. 

### Quick Start Guide 
1. Use [this link](https://365utsouthwestern-my.sharepoint.com/:u:/r/personal/yuqiu_yang_utsouthwestern_edu/Documents/pMTnet_Omni_datafile/data.tgz?csf=1&web=1&e=YdNTsn) to download the supporting data to your local directory, say <i>./data</i>. 

<b>Caution: </b> the data file is around 30Gb. Make sure you have enough space on your hard drive. 

2. Prepare your dataset so that it looks somewhat like the following:
![Sample df](/docs/source/images/sample_df.png)

3. Use our supporting package [pMTnet_Omni_Document](https://github.com/Yuqiu-Yang/pMTnet_Omni_Document) which should have been installed to the virtual environment to curate your data. 
Say your dataset is 
at `./df.csv`. After activating the virtual environment, you can proceed in two ways: 

For interactive Python, you can try the following: 
```shell
from pMTnet_Omni_Document.data_curation import read_file 

# Read the file 
df, mhc_seq_dict = read_file(file_path="./df.csv",
							save_results=True,
							sep=",")

```
For CLI, execute 
```shell
python -m pMTnet_Omni_Document --file_path ./df.csv --output_folder_path ./
```

In the output, you will see two files:

``./df_curated.csv`` will contain all the curated data. You will 
also see some extra columns in this file. 

.. note:: 
    
    If you see that ``mhca_use_seq`` and/or ``mhcb_use_seq`` columns 
    all have ``False``, then the json file will simply contain an empty 
    dictionary. 

``./mhc_seq_dict.json`` is a *json* file of a dictionary.
The keys are various MHC sequences and the values are their corresponding 
ESM embeddings.

4. Say your curated dataset is at `./df_curated.csv` and the extra MHC dictionary is saved at `./extra_mhc_dict.json`. For interactive Python, you can run 
```shell
# Load packages 
import pandas as pd 
from pMTnet_Omni.pMTnet_Omni_class import pMTnet_Omni_class

# Read dataframe and initialize pmtnet-omni
df=pd.read_csv("./df_curated.csv")
pmtnet = pMTnet_Omni_class(extra_mhc_dict = "./extra_mhc_dict.json")
pmtnet.read_user_df(df)

# Binding affinity prediction 
pmtnet.predict(output_file_path = "./pmtnet_results.csv",
					check_size = [1000, 10000, 100000, 1000000],
					rank_threshold = [0.15, 0.1, 0.08, 0.03])

```
For CLI, in your terminal, run 
```shell
python -m pMTnet_Omni --data_dir ./data --user_data ./df_curated.csv --extra_mhc_dict_path ./extra_mhc_dict.json --compute_percentile_rank --output_file_path ./df_results.csv
```

In general, pMTnet_Omni generates prediction for one TCR-pMHC within a couple of seconds on H100.

5. An example output would look like this:
![Sample output](/docs/source/images/sample_output.png)
 
