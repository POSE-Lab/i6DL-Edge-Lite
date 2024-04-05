# epos-opt

Original code for vanillia EPOS from [EPOS: Estimating 6D Pose of Objects with Symmetries](https://github.com/thodan/epos)
## Installation

### 1. Clone the environment and include submodules:

```
git clone --recursive https://github.com/pansap99/epos-opt.git
```

### 2. Set up conda environment

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

Create file ```~/anaconda3/envs/lala/etc/conda/activate.d/env_vars.sh``` with the following content:

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


