# utility script for building the ARM images
# build context is the repo 
TAG=$1
#DOCKER_ID=$1
docker build -t base-x86:$TAG -f ./x86/base-x86/Dockerfile .. && \ 
docker build -t epos-x86-opt:$TAG -f ./x86/epos-x86-opt/Dockerfile ..
