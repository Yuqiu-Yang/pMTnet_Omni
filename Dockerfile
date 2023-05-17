FROM continuumio/miniconda3

WORKDIR /usr/src/pMTnet_Omni

COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN conda install jupyter

RUN mkdir -p /usr/src/user_input/
RUN mkdir -p /usr/src/user_output/
RUN mkdir -p /usr/src/pMTnet_Omni_data/