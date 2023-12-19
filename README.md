# pMTnet Omni: pan-MHC and cross-Species Prediction of T Cell Receptor-Antigen Binding :microscope:

![Logo](/assets/pmtnet_logo.png)

![forthebadge](/assets/tcr-pmhc.svg)
![forthebadge](/assets/deep-learning.svg)

| Package | Documentation | Code Coverage |
| --- | --- | --- |
| pMTnet Omni | | [![codecov](https://codecov.io/gh/Yuqiu-Yang/pMTnet_Omni/branch/main/graph/badge.svg?token=L59TPMM3VN)](https://codecov.io/gh/Yuqiu-Yang/pMTnet_Omni) |
| pMTnet Omni Document | [![Documentation Status](https://readthedocs.org/projects/pmtnet-omni-document/badge/?version=latest)](https://pmtnet-omni-document.readthedocs.io/en/latest/?badge=latest) | [![codecov](https://codecov.io/gh/Yuqiu-Yang/pMTnet_Omni_Document/branch/main/graph/badge.svg?token=BR1vyICN3q)](https://codecov.io/gh/Yuqiu-Yang/pMTnet_Omni_Document) |

Profiling the binding of T cell receptors (TCRs) of T cells to antigenic peptides presented by MHC proteins is one of the most important unsolved problems in modern immunology. We developed <b>pMTnet-omni</b>, an Artificial Intelligence (AI) system based on hybrid protein sequence and structure information, to predict the pairing of TCRs of &alpha;&beta; T cells with peptide-MHC complexes (pMHCs). <b>pMTnet-omni</b> is capable of handling peptides presented by both class I and II pMHCs, and capable of handling both human and mouse TCR-pMHC pairs, through information sharing enabled this hybrid design. 

Please refer to our paper for more details: [pMTnet Omni paper link here](https://doi.org/10.1101/2023.12.01.569599)

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
conda create -n pMTnet_Omni_env python=3.10
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

**_NOTE:_** We DO NOT have an online document for this repo. However, we do have HTML files for the documentation ready to go.

### Model Overview 
![Model Overview](/docs/source/images/overview.png)

### Quick Start Guide 
1. Use [this link](https://365utsouthwestern-my.sharepoint.com/:u:/r/personal/yuqiu_yang_utsouthwestern_edu/Documents/pMTnet_Omni_datafile/data.tgz?csf=1&web=1&e=YdNTsn) to download the supporting data to your local directory, say <i>./data</i>. 

<b>Caution: </b> the data file is around 30Gb. Make sure you have enough space on your hard drive. 

2. Prepare your dataset so that it looks somewhat like the following:
![Sample df](/docs/source/images/sample_df.png)

3. Use our supporting package [pMTnet_Omni_Document](https://github.com/Yuqiu-Yang/pMTnet_Omni_Document) to curate your data. 

4. Say your curated dataset is at `./df.csv` and the extra MHC dictionary is saved at `./extra_mhc_dict.json`. For interactive Python, you can run 
```shell
# Load packages 
import pandas as pd 
from pMTnet_Omni.pMTnet_Omni_class import pMTnet_Omni_class

# Read dataframe and initialize pmtnet-omni
df=pd.read_csv("./df.csv")
pmtnet = pMTnet_Omni_class()
pmtnet.read_user_df(df)

# Binding affinity prediction 
pmtnet.predict(output_file_path = "./pmtnet_results.csv",
					check_size = [1000, 10000, 100000, 1000000],
					rank_threshold = [0.15, 0.1, 0.08, 0.03])

```
For CLI, in your terminal, run 
```shell
python -m pMTnet_Omni --data_dir ./data --user_data_path ./df.csv --extra_mhc_dict_path ./extra_mhc_dict.json --output_file_path ./df_results.csv
```

5. An example output would look like this:
![Sample output](/docs/source/images/sample_output.png)
 
