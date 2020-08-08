BCGNet: Deep Learning Toolbox for BCG Artifact Reduction
========================================================
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![License](https://img.shields.io/github/license/SASVDERDBGTYS/BCGNet)](https://opensource.org/licenses/Apache-2.0)
[![doi](https://img.shields.io/badge/doi-10.1109%2FTBME.2020.3004548-blue)](https://ieeexplore.ieee.org/document/9124646)


BCGNet is a deep learning toolbox based on Keras and TensorFlow for ballistocardiogram (BCG) artifact reduction. More detail about our model and methods can be found [here](https://ieeexplore.ieee.org/document/9124646).<br>

# Table of Contents

- [Installation Guide](#installation-guide)
- [Jupyter Demo](#jupyter-demo)
- [Usage](#usage)
- [Reference](#reference)

# Installation Guide
## Additional Dependencies (Optional)
Before setting up the Python environment, there might be several additional software dependencies that need to 
be installed if the user wishes to use the GPU version of TensorFlow for training, which is must faster than the CPU
version. Please note GPU version of TensorFlow is available only for compatible GPUs from NVIDIA.

### 1. NVIDIA Driver and CUDA
For the Tensorflow version 2.3.0 we used for testing this package, NVIDIA driver version has to be at least 418.39 and 
CUDA version has to be 10.1. Our build was tested using NVIDIA driver version 435.21 and CUDA 10.1.

To install NVIDIA driver, download the desired version for the particular OS and GPU on your local unit under the 'Advanced Driver Search' [here](https://www.nvidia.com/Download/index.aspx).<br> 

To install CUDA, detailed instructions can be found [here](https://developer.nvidia.com/cuda-toolkit).<br>

Note: after installing, NVIDIA driver and CUDA version can be checked by executing `nvidia-smi` in your terminal.

### 2. CuDNN
Once CUDA is successfully installed, the user can proceed to install CuDNN. Detailed instructions for installing CuDNN
can be found [here](https://docs.nvidia.com/deeplearning/sdk/pdf/cuDNN-Installation-Guide.pdf).<br>

Note: For Linux users, note that it's critical that `LD_LIBRARY_PATH`, which maps the CuDNN path, is added to 
`.bashrc` and sourced.

### 3. miniconda/anaconda
It's recommended to install either anaconda or the lightweight miniconda on your machine for managing the Python 
environment. To install anaconda/miniconda, detailed instructions can be found [here](https://docs.anaconda.com/anaconda/install/).<br>

## Setting up Python Dependencies

The recommended way is to set up a new conda environment using the `requirement.yml` file we provide. Alternatively,
the user can choose to install all the packages using `pip`.

First clone/download our package into any directory you wish, then set up the dependencies via the following ways.

### 1. Using yml file (Recommended)
To set up the environment using the yml file, simply execute the following commands, and a new conda environment with 
name `bcgnet_TF230`
```
cd $PATH_TO_OUR_PACKAGE

conda env create -f requirement.yml
```
    
### 2. Using pip
Alternatively, the user can choose to install all the Python dependencies using `pip` via the following commands
```
cd $PATH_TO_OUR_PACKAGE

pip install requirement.txt
```

The user then needs to install `jupyterlab` and the detailed instructions can be found [here](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html).<br>

Note: to check the TensorFlow version, execute
```
python -c 'import tensorflow as tf; print(tf.__version__)'
``` 
# Jupyter Demo
Once the package is downloaded and all dependencies are installed, the user can learn sample usage
in `demo.ipynb`, which uses two runs of data from a single subject. To do that, the user type the following in
their terminal
```
conda activate bcgnet_TF230
jupyter lab
```
and then open up the demo file in the browser window. 

# Usage
The directory structure is as follows
```
BCGNet
|-config
|   __init__
|   default_config.yaml
|   load_config
|-dataset
|   __init__
|   dataset_utils
|   default_dataset
|-example_data
|-models
|   __init__
|   default_models
|   gru_arch_000
|   model_utils
|-session
|   __init__
|   data_generator
|   default_session
|-utils
|   __init__
|   context_management
|-.gitignore
|-__init__
|-demo.ipynb
|-README.md
|-requirement.txt
|-requirement.yml
```

In particular `\config` contains the configuration file `default_config.yaml` as well as the python functions that 
handles loading of configuration files in `load_config.py`. 

`\dataset` contains functions that handles single runs of data, which are encapsulated in dataset objects defined in
`default_datasets.py` while some utility functions are in `dataset_utils.py`.

`\example_data` contains two runs of raw data and the same runs of data processed by the optimal basis set (OBS) 
method.

`\models` contains the neural network models that are used. The model from our paper is defined in 
`default_models.py` and `gru_arch_000` provides an example for user on how to define a custom model. Please note that
each model must be contained in a separate file with the name of the file same as the name of the model. 
`model_utils.py` handles loading of the models.

`\session` contains functions that handles the entire training session, defined in `default_session.py` while 
`data_generator.py` defines the data generator for training the models. 

`\utils` contains utility functions that are used for the entire project.

# Reference
If you use this code in your project, please cite:
```
McIntosh, J. R., Yao, J., Hong, L., Faller, J., & Sajda, P. (2020). Ballistocardiogram artifact reduction 
in simultaneous EEG-fMRI using deep learning. IEEE Transactions on Biomedical Engineering.
```
Direct link: https://ieeexplore.ieee.org/document/9124646

Bibtex format:
```
@article{mcintosh2020ballistocardiogram,
  title={Ballistocardiogram artifact reduction in simultaneous EEG-fMRI using deep learning},
  author={McIntosh, James R and Yao, Jiaang and Hong, Linbi and Faller, Josef and Sajda, Paul},
  journal={IEEE Transactions on Biomedical Engineering},
  year={2020},
  publisher={IEEE}
}
```
