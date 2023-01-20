from setuptools import setup 

VERSION = "0.0.1"


setup(
    name="pMTnet_Omni",
    version=VERSION,
    description="",
    author="Yi Han, Yuqiu Yang, Tao Wang",
    author_email="Tao.Wang@UTSouthwestern.edu",
    python_requires=">=3.9",
    install_requires=[
        'numpy==1.22.4',
        'pandas==1.5.2',
        'matplotlib==3.6.2',
        'scikit-learn==1.0.2',
        'tqdm==4.64.1',
        'torch==1.13.1',
        'pytest==7.1.2',
        'coverage==6.3.2',
        'pytest-cov==3.0.0',
        'pytest-mock==3.10.0'
    ],
)

