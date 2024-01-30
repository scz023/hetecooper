# Installation

* [__System/Hardware Requirements__](#requirements)
* [__Installation__](#installation)
    * [__1. Dependency Installation__](#1-dependency-installation)
    * [__2. Install Pytorch__](#2-pytorch-installation-18)
    * [__3. Install Spconv__](#3-spconv-121-requred)




---
## System/Hardware Requirements
To get started, the following requirements should be fulfilled.
* __System requirements.__ OpenCOOD is tested under Ubuntu 18.04
* __Adequate GPU.__ A minimum of 6GB gpu is recommended.
* __Disk Space.__ Estimate 100GB of space is recommended for data downoading.
* __Python__ Python3.7 is required.


---
## Installation
### 1. Dependency Installation
First, download Hetecooper github to your local folder if you haven't done it yet.
```sh
git clone https://github.com/scz1/hetecooper.git
cd hetecooper
```
Next we create a conda environment and install the requirements.

```sh
conda env create -f environment.yml
conda activate hetecooper
python setup.py develop
```

If conda install failed,  install through pip
```sh
pip install -r requirements.txt
```

### 2. Pytorch Installation (pytorch==1.10.1, cuda==11.3)

Go to https://pytorch.org/ to install pytorch cuda version.

### 3. Spconv (1.2.1 or 2.x)
OpenCOOD support both spconv 1.2.1 and 2.x to generate voxel features. 

To install spconv 1.2.1, please follow the guide in https://github.com/traveller59/spconv/tree/v1.2.1.

To install spconv 2.x, please run the following commands (if you are using cuda 11.3):
```python
pip install spconv-cu113
```
#### Tips for installing spconv 1.2.1:
1. make sure your cmake version >= 3.13.2
2. CUDNN and CUDA runtime library (use `nvcc --version` to check) needs to be installed on your machine.



### 4. Bbx IOU cuda version compile
Install bbx nms calculation cuda version
  
  ```bash
  python opencood/utils/setup.py build_ext --inplace
  ```


### 5. Dependencies for FPV-RCNN (optional)
Install the dependencies for fpv-rcnn.
  
  ```bash
  python opencood/pcdet_utils/setup.py build_ext --inplace
  ```



### 6. Tools for acclerate graph build and subgraph map
Install the dependencies for fpv-rcnn.
  
  ```bash
  python opencood/graph_utils/setup.py build_ext --inplace
  ```

### 7. Deep Graph Library
Install graph tools.
<br/>
If you have installed dgl-cuXX package, please uninstall it first.
  ```bash
  pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html
  pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
  ```

<!-- ### 7. torch_geometric tools

```bash
conda install pyg=2.2 -c pyg -c conda-forge
pip install pyg-lib -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
# torch_sparse install
pip install https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_sparse-0.6.13-cp37-cp37m-linux_x86_64.whl
# torch_scatter install
pip install https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
pip install pytorch-lightning yacs torch_geometric==2.0.4 torchmetrics==0.11.4
pip install performer-pytorch
pip install tensorboardX

``` -->

### 8. clean conda cache
```bash
conda clean --all
```