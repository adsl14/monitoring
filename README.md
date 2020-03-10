# Monitoring
-------------

Machine learning network for monitoring areas.

## Install Earth Engine API and Tensorflow in Windows 10

### 1. Software

* Earth Engine API
* NVIDIA CUDA Toolkit 10.0
* NVIDIA cuDNN v7.6.5
* Python 3.7
* Tensorflow 1.15.0

### 2. Install CUDA

#### 2.1 Download CUDA 10.0 and install it in C hard disk: [CUDA](https://developer.nvidia.com/cuda-downloads)

### 3. Install cuDNN

#### 3.1 Download cuDNN v7.6.5 (November 5th, 2019), for CUDA 10.0 (you need to register first): [cuDNN](https://developer.nvidia.com/rdp/form/cudnn-download-survey)

#### 3.2 Unzip the file downloaded, copy the folders from the zip (include, lib, bin) and replace them with the same folders that are located where CUDA 10.0 were installed.

### 4. Install Python 3.7

#### 4.1 Install Anaconda: [Anaconda]:(https://www.anaconda.com/distribution/)

### 5. Install Earth Engine API and Tensorflow

#### 5.1 Open the 'Anaconda prompt'

#### 5.2 Write the following in the command line in order to install earthengine and then tensorflow:
1. Activate the virtual environment: `activate keras-gpu`
2. Install earthengine: `pip install earthengine-api`
2. Install tensorflow: `pip install tensorflow-gpu==1.15`
