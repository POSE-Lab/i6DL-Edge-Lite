# Utility script for building the Docker images
# Build context is the repo

if [ $# -ne 2 ]; then
  echo -e "Please provide a valid image tag and target architecture ("x86" or "arm")"
  exit 1
fi

TAG=$1
ARCH=$2 
docker build -t base-$ARCH:$TAG --build-arg="CPU_CORES=$(nproc)" -f ./$ARCH/base-$ARCH/Dockerfile .. && \
docker build -t i6dl-edge-lite-$ARCH:$TAG --build-arg="TAG=$TAG" --build-arg="CPU_CORES=$(nproc)" -f ./$ARCH/i6dl-edge-lite-$ARCH/Dockerfile ..

