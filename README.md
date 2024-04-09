# epos-opt

Original code for vanilla EPOS from [EPOS: Estimating 6D Pose of Objects with Symmetries](https://github.com/thodan/epos)
## Prerequisites
- CUDA >= 11.6
- glog headers (`sudo apt-get install libgoogle-glog-dev`)
## Installation

### 1. Clone the environment and include submodules:

```
git clone --recursive https://github.com/pansap99/epos-opt.git
```

### 2. Set up conda environment
- Change the prefix in `environment.yml` accordingly
- Install environment with
```
conda env create -f environment.yml
```

### 3. Build progressive-x

```
cd ./external/progressive-x
mkdir build; cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

### 4. Install visualization package (Optional)

Download the pose-vis wheel file from [here](https://ntuagr-my.sharepoint.com/:f:/g/personal/psapoutzoglou_ntua_gr/EoR2e85O8xpDnRlNf9IFOb0B1N5fc_fjAgRqKB4v_KVEYA?e=7cqhW9) and install it:
```
pip install pose_vis-1.0-py3-none-any.whl
```

### 5. Setup environment variables
- Run `mkdir -p $CONDA_PREFIX/etc/conda/activate.d`
- Create file ```$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh``` with the following content:

```
#!/bin/sh

export REPO_PATH=/path/to/cloned/repo # Folder for the EPOS repository.
export STORE_PATH=/path/to/cloned/repo/store  # Folder for TFRecord files and trained models.
export BOP_PATH=/path/to/cloned/repo/datasets  # Folder for BOP datasets (bop.felk.cvut.cz/datasets).

export TF_DATA_PATH=$STORE_PATH/tf_data  # Folder with TFRecord files.
export TF_MODELS_PATH=$STORE_PATH/tf_models  # Folder with trained EPOS models.

export PYTHONPATH=$REPO_PATH:$PYTHONPATH
export PYTHONPATH=$REPO_PATH/external/bop_renderer/build:$PYTHONPATH
export PYTHONPATH=$REPO_PATH/external/bop_toolkit:$PYTHONPATH
export PYTHONPATH=$REPO_PATH/external/progressive-x/build:$PYTHONPATH
export PYTHONPATH=$REPO_PATH/external/slim:$PYTHONPATH

export LD_LIBRARY_PATH=$REPO_PATH/external/llvm/lib:$LD_LIBRARY_PATH
```
- Re-activate conda environment to set the parameters

### 6. Download and setup the directories

- Download models from this [folder](https://ntuagr-my.sharepoint.com/:f:/g/personal/psapoutzoglou_ntua_gr/EnRqn_GBhJpKj_DOiuSLYlMBqtT8M2_HYY2hDAvcyyYdng?e=3wRcPN) and place them under the ```$STORE_PATH``` directory
- Donwload any dataset from the ```datasets``` [folder](https://ntuagr-my.sharepoint.com/:f:/g/personal/psapoutzoglou_ntua_gr/ElH4q1jy60pApZIKXSS33PYBO34GMvJOVg_x81g58ZzPbA?e=f3G6TX) and place it under ```$BOP_PATH$``` directory.
- Adjust the ```./config.yml``` file accordingly.

## Usage 

### Inference with the ONNX runtime

```
python infer.py --imagePath='path/to/images' --config=./config.yml  --objID=1
```

### Evaluation

```
python eval.py --gtPoses='../../datasets/carObj1/test_primesense/000001/scene_gt.json' --estPoses='./eval/est_poses.json'
```

### Visualization

```
python vis.py  --objID=1  --images='../../datasets/carObj1/test_primesense/000001/rgb'  --poses='./eval/est_poses.json'  --confs='./eval/confs.txt'
```
## Dockers
The repo contains Dockerfiles for building Docker images containing all the required components to run epos-opt for two architectures (x86, arm).
## Prerequisites
- Install the NVIDIA container toolkit as documented [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- If not already done, setup the directories as described in **Installation - step 6**.

## Instructions
- Change to the `docker` directory
- If you wish to build the images, run the `build_all_<arch>.sh` script, where `<arch>`=`x86` or `arm`. Give the desired image tag (e.g. `latest`) as argument. The image used as base for `base` image is deprecated, and therefore unavailable to use for building the images from scratch.  
- In `run_container.sh`, change the `STORE_PATH`, `BOP_PATH`, `EVAL_RES`, `CONFIG_FILE` variables accordingly (please use absolute paths). The folders defined by `$STORE_PATH`, `$BOP_PATH` , `$EVAL_RES` and the file defined by `$CONFIG_FILE` will be mounted on the container on runtime from the host as bind mounts so the contents can be accessed from both the host and the container. The paths in the `config.yml` file should refer to directories *in the container* (typically beginning with `/home/epos-opt/`). 
- Run `run_container.sh` by specifying the desired Docker image, tag and architecture. For example, for running the arm image/-s, run `./run_container.sh epos-arm-opt latest arm`
- Within the container, `cd /home/epos-opt/scripts`
- From here follow the instructions in **Usage** (Visualization is not supported!)
- To avoid the long build times, you can pull the images from Dockerhub with `docker pull` (`felice2023/base:latest`, `felice2023/epos-x86-opt:latest, felice2023/base-arm:latest`, `felice2023/epos-arm-opt:latest`). Please rename them with `docker image tag` before using them (remove the `felice2023`, e.g. change `felice2023/base:latest` to `base:latest`)

## Troubleshooting:
  - `Could NOT find CUDA: Found unsuitable version "", but required is exact
  version "11.6" (found /usr)` when building ProgressiveX outside Docker: try specifying the CUDA toolkit location in cmake configuration 
  (`-D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda`)
  - `CMake Error: The source "/home/epos-opt/external/progressive-x/CMakeLists.txt" does not match the source "<path>/epos-opt/external/progressive-x/CMakeLists.txt" used to generate cache.  Re-run cmake with a different source directory.`: Delete CMakeCache.txt
