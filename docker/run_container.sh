if [ $# -lt 5 ]; then
  echo -e "5 arguments should be provided in the form of key=value:\n
  IMAGE: A valid docker image name\n
  TAG: Docker image tag\n
  STORE_PATH: Absolute path in the host containing trained models and other files. Maps to /home/i6DL-Edge-Lite/store in the container\n
  BOP_PATH: Absolute path in the host for BOP datasets. Maps to /home/i6DL-Edge-Lite/store/bop_datasets in the container.\n
  EVAL_RES: Absolute path to folder for storing evaluation results after the container's deletion. Maps to /home/i6DL-Edge-Lite/scripts/eval in the container."
  exit 1
fi

for ARGUMENT in "$@" # iterate over arguments (given as key=value pairs)
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=) # cut off argument at '=' delimiter to keep the key

    KEY_LENGTH=${#KEY}
    VALUE="${ARGUMENT:$KEY_LENGTH+1}" # cut off $KEY_LENGTH+1 characters from $ARGUMENT to keep the $VALUE
    export "$KEY"="$VALUE"
done

SCRIPTS_PATH="$(pwd)/../scripts"
echo $IMAGE
echo $TAG
echo $STORE_PATH
echo $BOP_PATH
echo $EVAL_RES
echo $SCRIPTS_PATH
xhost +

nvidia-docker run -it --privileged \
-v "$STORE_PATH:/home/i6DL-Edge-Lite/store" \
-v "$SCRIPTS_PATH:/home/i6DL-Edge-Lite/scripts" \
-v "$BOP_PATH:/home/i6DL-Edge-Lite/store/bop_datasets" \
-v "$EVAL_RES:/home/i6DL-Edge-Lite/scripts/eval" \
--net=host \
--env="TF_DATA_PATH=$STORE_PATH/tf_data" \
--env="TF_MODELS_PATH=$STORE_PATH/tf_models" \
--env="BOP_PATH=/home/i6DL-Edge-Lite/store/bop_datasets" \
--env="DISPLAY=$DISPLAY" \
  --init  \
    --env="NVIDIA_DRIVER_CAPABILITIES=all" \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix" \
    ${IMAGE}:${TAG} \
    bash
