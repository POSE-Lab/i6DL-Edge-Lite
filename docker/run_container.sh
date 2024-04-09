if [ $# -lt 3 ]; then
  echo -e "3 arguments should be provided.\nA valid docker image name, image tag, target architecture (x86 or arm)\n"
  exit 1
fi

IMAGE=$1
TAG=$2
ARCH=$3

STORE_PATH="/home/nvidia/nvme/athena/epos-opt/store" # Path in the host containing trained models and other files
BOP_PATH="/home/nvidia/nvme/athena/epos-opt/store/bop_datasets" # Path in the host for BOP datasets
CONFIG_FILE="/home/nvidia/nvme/athena/epos-opt/scripts/config_mine.yml" # will be mounted on the container so it can be modified from the host. Use absolute path.
EVAL_RES="/home/nvidia/nvme/athena/epos-opt/eval" # Folder to store evaluation results after the container's deletion. 
cd $ARCH 
xhost +

nvidia-docker run -it --privileged \
-v "$STORE_PATH:/home/epos-opt/store" \
-v "$CONFIG_FILE:/home/epos-opt/scripts/config.yml" \
-v "$BOP_PATH:/home/epos-opt/store/bop_datasets" \
-v "$EVAL_RES:/home/epos-opt/scripts/eval" \
--net=host \
--env="TF_DATA_PATH=$STORE_PATH/tf_data" \
--env="TF_MODELS_PATH=$STORE_PATH/tf_models" \
--env="BOP_PATH=/home/epos-opt/store/bop_datasets" \
--env="DISPLAY=$DISPLAY" \
  --init  \
    --env="NVIDIA_DRIVER_CAPABILITIES=all" \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix" \
    ${IMAGE}:${TAG} \
    bash
