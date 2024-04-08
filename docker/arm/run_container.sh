if [ $# -lt 2 ]; then
  echo -e "At least 2 arguments should be provided.\nMandatory: a valid docker image name, image tag\noptional: environment parameters file"
  exit 1
fi

IMAGE=$1
TAG=$2

USER_CONTAINER=root
WS_CONTAINER="/$USER_CONTAINER/catkin_ws/"
WS_HOST="$(pwd)/catkin_ws"
SANDBOX_HOST="$(pwd)/sandbox/"
SANDBOX_CONTAINER="/$USER_CONTAINER/sandbox/"
STORE_PATH="/home/nvidia/nvme/athena/epos-opt/store" # in the host
BOP_PATH="/home/epos-opt/store/bop_datasets" # in the container, leave it as default
CONFIG_FILE="/home/nvidia/nvme/athena/epos-opt/scripts/config.yml" # will be mounted on the container so it can be modified from the host
# -v $WS_HOST:$WS_CONTAINER:rw \
#export ROS_MASTER_URI=http://147.102.107.214:11311/
#export ROS_IP=147.102.107.141
#export ROS_HOSTNAME=147.102.107.141
xhost +
#cd main-arm-new
if [ -z "$3" ]; then
echo "no environment file chosen"
  nvidia-docker run -it --privileged \
  -v "$SANDBOX_HOST:$SANDBOX_CONTAINER:rw" \
  -v "$WS_HOST:$WS_CONTAINER:rw" \
  -v "$HOME/.bashrc:/$USER_CONTAINER/.bashrc:rw" \
  -v "$STORE_PATH:/home/epos-opt/store" \
  -v "$CONFIG_FILE:/home/epos-opt/scripts/config.yml" \
  --net=host \
  --env="TF_DATA_PATH=$STORE_PATH/tf_data" \
  --env="TF_MODELS_PATH=$STORE_PATH/tf_models" \
  --env="BOP_PATH=$BOP_PATH" \
  --env="DISPLAY=$DISPLAY" \
    --init  \
      --env="NVIDIA_DRIVER_CAPABILITIES=all" \
      --env="DISPLAY=$DISPLAY" \
      --env="QT_X11_NO_MITSHM=1" \
      --volume="/tmp/.X11-unix:/tmp/.X11-unix" \
      ${IMAGE}:${TAG} \
      bash

else 
  ENV_FILE=$3
  nvidia-docker run -it --privileged \
  -v "$SANDBOX_HOST:$SANDBOX_CONTAINER:rw" \
  -v "$WS_HOST:$WS_CONTAINER:rw" \
  -v "$HOME/.bashrc:/$USER_CONTAINER/.bashrc:rw" \
  --env-file $ENV_FILE \
  --net=host \
  --env="DISPLAY=$DISPLAY" \
    --init  \
      --env="NVIDIA_DRIVER_CAPABILITIES=all" \
      --env="DISPLAY=$DISPLAY" \
      --env="QT_X11_NO_MITSHM=1" \
      --volume="/tmp/.X11-unix:/tmp/.X11-unix" \
      ${IMAGE}:${TAG} \
      bash
fi 
