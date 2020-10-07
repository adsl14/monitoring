# Monitoring
-------------

<img src="images/logo.jpg" width="50%">

Deep learning network for monitoring areas using sentinel-2 for analisys and sentinel-1 for analisys and training.

## Install Earth Engine API and Tensorflow in Windows 10

### 1. Software

* Anaconda 4.8.5
* NVIDIA CUDA Toolkit 10.0
* NVIDIA cuDNN v7.6.5
* Python 3.7.9
* Earth Engine API 0.1.237
* Keras 2.3.1
* Tensorflow 1.15.0
* Pandas 1.1.3
* Sklearn 0.23.2
* Matplotlib 3.3.2
* Pywin32 227
* Git 2.23.0.windows.1

### 2. Install CUDA

#### 2.1 Download CUDA Toolkit 10.0 and install it in C hard disk: [CUDA](https://developer.nvidia.com/cuda-10.0-download-archive)

### 3. Install cuDNN

#### 3.1 Download cuDNN v7.6.5 (November 5th, 2019), for CUDA 10.0: [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)

#### 3.2 Unzip the file downloaded, copy the folders from the zip (include, lib, bin) and replace them with the same folders that are located where CUDA 10.0 were installed.

### 4. Install Anaconda in Windows

#### 4.1 Anaconda page: [Anaconda](https://www.anaconda.com/products/individual)

### 5. Install Earth Engine API, Tensorflow and Keras.

#### 5.1 Open the 'Anaconda prompt'

#### 5.2 Write the following commands in order to install the packages:
1. Create the virtual environment with python 3.7.9 version: `conda create --name keras-gpu python=3.7.9`
2. Activate the virtual environment: `conda activate keras-gpu`
3. Install earthengine: `pip install earthengine-api`
4. Install keras: `pip install keras==2.3.1`
5. Install tensorflow: `pip install tensorflow-gpu==1.15`
6. Install pandas: `pip install pandas`
7. Install sklearn: `pip install sklearn`
8. Install matplotlib: `pip install matplotlib`
9. Install Linux commands in Windows: `conda install m2-base`
10. Install Python extensions for Microsoft Windows: `conda install pywin32`
11. Install Github: `conda install git`

If you have problems when running the code, check if the package's version are the same from above.