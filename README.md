# i6DL-Edge-Lite

ROS-independent version of [i6DL-Edge](https://github.com/POSE-Lab/i6DL-Edge). The code was tested on Ubuntu 20.04, with GCC/G++ 9.4.0. The module can run:
- On **x86_64** architectures (typical desktop PCs), either in a **conda** environment (section [Installation](#install)) or a **Docker** environment (section [Dockers](#Dockers))
- on **ARM/aarch64** architectures, in a Docker environment.

## Prerequisites
- CUDA >= 11.6
- glog headers (`sudo apt-get install libgoogle-glog-dev`)
- libopencv-dev (`sudo apt-get install libopencv-dev`)
- libeigen3-dev (`sudo apt-get install libeigen3-dev`)

## <a name="install"></a> Installation

Steps 2-5 only apply when running the module in a **conda** environment.

### 1. Clone the environment and include submodules:

```
git clone --recursive https://github.com/POSE-Lab/i6DL-Edge-Lite.git
```

### 2. Set up conda environment
- Change to `base` environment.
- Install the i6DL-Edge-Lite environment with
```
conda env create --prefix $CONDA_PREFIX/envs/eposOpt -f environment.yml
```
`$CONDA_PREFIX` is the environment variable pointing to the Anaconda installation path. 

- Activate the environment and proceed with the rest of the steps.

### 3. Build progressive-x

```
cd ./external/progressive-x
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
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
export PYTHONPATH=$REPO_PATH/external/bop_toolkit:$PYTHONPATH
export PYTHONPATH=$REPO_PATH/external/progressive-x/build:$PYTHONPATH
export LD_LIBRARY_PATH=$REPO_PATH/external/llvm/lib:$LD_LIBRARY_PATH
```
- Re-activate conda environment to set the environment variables defined in the previous step.

### <a name="step6"></a> 6. Download and setup the directories

- Download any trained model from this [folder](https://ntuagr-my.sharepoint.com/:f:/g/personal/psapoutzoglou_ntua_gr/EnRqn_GBhJpKj_DOiuSLYlMBqtT8M2_HYY2hDAvcyyYdng?e=3wRcPN), unzip it and place it under the ```$STORE_PATH``` directory
- Download the [IndustryShapes dataset](https://www.doi.org/10.5281/zenodo.14616197) and place it under the ```$BOP_PATH``` directory. 
- Make a copy of the ```./config.yml``` file in ```scripts``` named e.g. `config_mine.yml` and adjust it accordingly (check the template `config.yml` for details).

## <a name="usage"></a> Usage 

Run the inference, evaluation, visualization scripts from within the `scripts` folder.

### Inference

Inference on a test dataset is supported 1) for the ONNX inference engine, using the trained model we provide (see [Installation - step 6](#step6)) 2) for the TensorRT inference engine. For the latter, see section [TensorRT inference](#tensorrt).

For either method, run the inference as follows: 

```
python infer.py --imagePath='/path/to/test_images' --config=/path/to/config_file  --objID=<object ID>
```
e.g.

```
python infer.py --imagePath=../../datasets/carObj1/test_primesense/000001/rgb/ --config=./config_mine.yml  --objID=1
```

### Evaluation

```
python eval.py --config /path/to/config_file --gtPoses='/path/to/bop_dataset/scene_gt.json' --estPoses='/path/to/evaluation_results/est_poses.json'
```
e.g. 

```
python eval.py --config ./config_mine.yml --gtPoses='../../datasets/carObj1/test_primesense/000001/scene_gt.json' --estPoses='./eval/est_poses.json'
```

### Visualization of estimated poses

```
python vis.py  --objID=<object ID>  --images='/path/to/test_images'  --poses='./path/to/evaluation_results/est_poses.json'  --confs='./path/to/evaluation_results/confs.txt'
```
e.g. 

```
python vis.py  --objID=1  --images='../../datasets/carObj1/test_primesense/000001/rgb'  --poses='./eval/est_poses.json'  --confs='./eval/confs.txt'
```
## <a name="tensorrt"></a> [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html) inference
A TensorRT model, or engine (also called a plan) is optimized in a way that is heavily dependent on the underlying hardware. As a result, a TensorRT model is generally not portable across different GPU architectures. For this reason, we do not provide built TensorRT models. Instead, one should build the model (engine) themselves from the provided ONNX model using `trtexec`. 

### FP16 and FP32 engine
Run 

```
/usr/src/tensorrt/bin/trtexec --onnx=/path/to/onnx/model --saveEngine=/path/to/output/engine --<precision> 

```
where `precision` = `fp16` or `fp32`

### INT8 engine
To create an INT8 engine, the pre-trained model (ONNX in this case) needs to be calibrated on a subset of the training data. After calibration, a cache file is generated, which will be used to generate the INT8 engine.

- Run the calibration script:
```
python calibrator.py --calib_dataset_loc /path/to/bop_dataset/train_folder --saveCache /output/cache/filename (calibration file) --onnx /path/to/onnx/model --img_size height width channels --num_samples num_samples --batch_size batch_size 
```
Where

`img_size`: image size of calibration images

`num_samples`: Number of samples that will be randomly selected for every object (default: 300)

`batch size`: Number of samples that will be processed in every iteration (batch size) (default: 64)

e.g.

```
python calibrator.py --calib_dataset_loc /home/i6DL-Edge-Lite/store/train_primesense --saveCache /home/i6DL-Edge-Lite/store/crf12345AndLab123MI3/crf12345AndLab123MI3_640_int8.cache --onnx /home/i6DL-Edge-Lite/store/crf12345AndLab123MI3/crf12345AndLab123MI3_640.onnx --img_size 480 640 3 --num_samples 100 --batch_size 4
```

- Build the engine:
```
/usr/src/tensorrt/bin/trtexec --onnx=/path/to/onnx/model --saveEngine=path/to/output/engine --int8 --calib=/path/to/calib/cache
```
### Inference
In the YAML configuration file, change the `method` field to `trt`, and the `trt` field to the path of the TensorRT engine you created. Run inference as described in Usage.

## <a name="Dockers"></a> Dockers
The repo contains Dockerfiles for building Docker images containing all the required components to run i6DL-Edge-Lite for two architectures (x86_64, arm/aarch64). For inference with TensorRT and Dockers, it is recommended to build the TensorRT models **inside** the container, as the host environment will likely differ from the Docker environment (see section [TensorRT inference](#tensorrt)).

### Prerequisites

**Install dependencies:**

```
sudo apt-get update
sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
```

**Add Docker’s official GPG key:**

```
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```

**Set up the Docker repository:**

```
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```
**Install Docker:**

```
  sudo apt-get update
  sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

**Verify installation:**
```
  sudo docker run hello-world
```
**Enable GPU/CUDA support**
- Install the appropriate NVIDIA drivers for your system from [the official page](https://www.nvidia.com/en-us/drivers/). Supported driver versions are >= 418.81.07
- Install the NVIDIA container toolkit as documented [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (the Apt installation was tested)


### Instructions
1. If not already done, setup the directories as described in **[Installation - step 6](https://github.com/POSE-Lab/i6DL-Edge-Lite/?tab=readme-ov-file#6-download-and-setup-the-directories)**.
2. Change to the `docker` directory
3. Build the images: run the `build_all.sh` script. Give as arguments the desired image tag (e.g. `latest`) and the CPU architecture (i.e. "x86" or "arm") for which you wish to build the Docker images (e.g `./build_all.sh latest x86`). 
4. Run `run_container.sh` with the following key-value arguments: 
```
  IMAGE: A valid docker image name
  TAG: Docker image tag
  STORE_PATH: Absolute path in the host containing trained models and other files. Maps to /home/i6DL-Edge-Lite/store in the container.
  BOP_PATH: Absolute path in the host for BOP datasets. Maps to /home/i6DL-Edge-Lite/store/bop_datasets in the container.
  EVAL_RES: Absolute path to folder for storing evaluation results after the container's deletion. Maps to /home/i6DL-Edge-Lite/scripts/eval in the container.

```
  For this particular script **only**, the order of the arguments is irrelevant. The folders defined by `$STORE_PATH`, `$BOP_PATH` , `$EVAL_RES`, and the `scripts` folder will be mounted on the container on runtime from the host as bind mounts so the contents can be accessed from both the host and the container. 

5. Within the container, `cd /home/i6DL-Edge-Lite/scripts`
6. Change the YAML configuration file so that any paths refer to directories *in the container* (typically beginning with `/home/i6DL-Edge-Lite/`). 
7. From here follow the instructions in [Usage](#usage) (Visualization is not supported!)

## Troubleshooting:
  - `Could NOT find CUDA: Found unsuitable version "", but required is exact
  version "11.6" (found /usr)` when building ProgressiveX outside Docker: try specifying the CUDA toolkit location in cmake configuration (`-D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda`)

  - `ImportError: $CONDA_PREFIX/lib/lib/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found (required by /lib/libgdal.so.30)` when running inference: try specifying the location of the required version of libstdc++.so.6 by creating a symbolic link:
  (`ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 $CONDA_PREFIX/lib/libstdc++.so.6`)

  - `CMake Error: The source "<container path>/external/progressive-x/CMakeLists.txt" does not match the source "<host path>/external/progressive-x/CMakeLists.txt" used to generate cache.  Re-run cmake with a different source directory.` when building the Docker images (specifically `i6dl-edge-lite-<arch>`): This may occur if you have built Progressive-X outside the Docker container first. Delete CMakeCache.txt in external/progressive-x/build on the host and re-run `build_all.sh`. 

  - `error: 'clamp' is not a member of 'std'` when building Progressive-X: Confirm that the GCC and g++ compilers support the C++ standard 17 by running `gcc -v --help 2> /dev/null | sed -n '/^ *-std=\([^<][^ ]\+\).*/ {s//\1/p}'`. Then delete the `build` folder and run CMake again as `cmake .. -DCMAKE_BUILD_TYPE=Release -D PYBIND11_CPP_STANDARD=-std=c++17`
