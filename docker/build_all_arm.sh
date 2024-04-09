# utility script for building the ARM images
# build context is the repo 
TAG=$1
#DOCKER_ID=$1
docker build -t base-arm:$TAG -f ./arm/base-arm/Dockerfile .. && \ 
docker build -t epos-arm-opt:$TAG -f ./arm/epos-arm-opt/Dockerfile ..
