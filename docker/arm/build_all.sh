# utility script for building the ARM images
TAG=$1
#DOCKER_ID=$1
cd base-arm && docker build -t base-arm:${TAG} . && cd ..
cd epos-arm-opt && docker build -t epos-arm-opt:${TAG} .
