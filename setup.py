from setuptools import setup 
import sys 
import os 
import shutil
import distutils.cmd


VERSION = "0.0.3"

class PypiCommand(distutils.cmd.Command):
    
    description = "Build and upload for PyPI."
    user_options = []
    
    def initialize_options(self):
        pass
    
    
    def finalize_options(self):
        pass
    
    
    def run(self):
        try:
            shutil.rmtree("dist/")
        except FileNotFoundError:
            pass
        
        wheel_file = "pMTnet_Omni-{}-py3-none-any.whl".format(VERSION)
        tar_file = "pMTnet_Omni-{}.tar.gz".format(VERSION)
        
        os.system("{} setup.py sdist bdist_wheel".format(sys.executable))
        os.system("twine upload dist/{} dist/{}".format(wheel_file, tar_file))
    
    
# class CondaCommand(distutils.cmd.Command):
    
#     description = "Build and upload for conda."
#     user_options = []
    
#     def initialize_options(self):
#         pass
    
    
#     def finalize_options(self):
#         pass
    
    
#     def run(self):
#         try:
#             shutil.rmtree("dist_conda/")
#         except FileNotFoundError:
#             pass
#         os.system("conda build . --output-folder dist_conda/")
#         os.system("anaconda upload ./dist_conda/noarch/pMTnet_Omni-{}-py_0.tar.bz2".format(VERSION))


setup(
    name="pMTnet_Omni",
    version=VERSION,
    description="Your one-stop TCR-pMHC affinity prediction algorithm",
    author="Yi Han, Yuqiu Yang, Tao Wang",
    author_email="yi.han@utsouthwestern.edu, yuqiuy@smu.edu, Tao.Wang@UTSouthwestern.edu",
    long_description_content_type="text/markdown",
    long_description=open("README.md").read(),
    packages=["pMTnet_Omni", "pMTnet_Omni.encoders"],
    python_requires=">=3.9",
    install_requires=[
        'numpy==1.22.4',
        'pandas==1.5.2',
        'matplotlib==3.6.2',
        'scikit-learn==1.0.2',
        'tqdm==4.64.1',
        'torch==1.13.1'
    ],
    test_requires=[
        'pytest==7.1.2',
        'coverage==6.3.2',
        'pytest-cov==3.0.0',
        'pytest-mock==3.10.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
    ],
    cmdclass={
        "pypi": PypiCommand,
        # "conda": CondaCommand
    }
)

