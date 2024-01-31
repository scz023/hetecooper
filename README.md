# Hetecooper: A Framework for Heterogeneous Agent Cooperation

Our paper will be available on Arxiv

<br/>

## Train model 

### 1. Train model under the condition of model isomorphism
```python
python opencood/tools/train_homo.py \
--hypes_yaml ${CONFIG_FILE} \
[--model_dir  ${CHECKPOINT_FOLDER}] \
[--fusion_method ${intermediate_with_comm}]
```
Example:
```python
python opencood/tools/train_homo.py \
--hypes_yaml opencood/hypes_yaml/opv2v/opv2v_point_pillar_graphformer.yaml \
--fusion_method intermediate_with_comm \
```

### 2. Adapt parameters of fusion module and mapping module for collaborative heterogeneous models
```python
python opencood/tools/train_fuse.py python opencood/tools/train_homo.py \
--model_dir ${EGO_CHECKPOINT_FOLDER} \
--model_dir_cooper $ {COLLABORATIVE_CHECKPOINT_FOLDER}
```

## Test model 
### 1. Collaborative agents uses isomorphism model
```python
python opencood/tools/inference_homo.py \
--model_dir ${CHECKPOINT_FOLDER} \
--fusion_method ${FUSION_STRATEGY}
```

### 2. Collaborative agents uses heterogeneous model
```python
python opencood/tools/inference_hete.py \
--model_dir ${CHECKPOINT_FOLDER} \
--model_dir_cooper $ {COLLABORATIVE_CHECKPOINT_FOLDER} \
--fusion_method ${FUSION_STRATEGY}
```

<br/>
<br/>

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
### Tips for installing spconv 1.2.1:
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

### 8. clean conda cache
```bash
conda clean --all
```

