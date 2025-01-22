ARG TAG=latest
FROM base-x86:${TAG}
WORKDIR /home
ARG CPU_CORES=8
## Set up EPOS
# Install necessary Python modules
COPY ./docker/x86/i6dl-edge-lite-x86/requirements.txt ./requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Update 21/01/2025: change ONNX Runtime installation way because 
# the default CUDA version for onnxruntime-gpu in pypi is 12.x since 1.19.0.
RUN pip install onnxruntime-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
RUN apt-get install sudo

RUN sudo git clone https://github.com/libigl/libigl-python-bindings.git &&\
cd libigl-python-bindings &&\ 
python3 -m pip install libigl

ENV REPO_PATH_SRC=.
ENV REPO_PATH=/home/i6DL-Edge-Lite
#RUN sudo mkdir -p ${REPO_PATH_DST}

COPY ${REPO_PATH_SRC}/epos_lib ${REPO_PATH}/epos_lib
COPY ${REPO_PATH_SRC}/external ${REPO_PATH}/external
RUN echo "Using ${CPU_CORES} CPU cores"

ENV PYTHONPATH=$REPO_PATH:$PYTHONPATH
ENV PYTHONPATH=$REPO_PATH/external/bop_toolkit:$PYTHONPATH
ENV PYTHONPATH=$REPO_PATH/external/progressive-x/build:$PYTHONPATH
ENV PYTHONPATH=$REPO_PATH/external/slim:$PYTHONPATH

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$REPO_PATH/external/llvm/lib

RUN sudo mkdir -p $REPO_PATH/external/progressive-x/build &&\
cd $REPO_PATH/external/progressive-x/build &&\
sudo cmake .. -DCMAKE_BUILD_TYPE=Release &&\
sudo make -j${CPU_CORES}

RUN pip install matplotlib rich tqdm plyfile
RUN apt update -y && apt install -y python-is-python3

RUN pip install pycuda --verbose
ENV TRT_VERSION=8.5.2.2
RUN apt-get update && apt-get -y install --fix-broken libnvinfer8=8.5.2-1+cuda11.8 \
libnvinfer-bin=8.5.2-1+cuda11.8 \
libnvinfer-plugin8=8.5.2-1+cuda11.8 \
libnvparsers8=8.5.2-1+cuda11.8 \
libnvonnxparsers8=8.5.2-1+cuda11.8 \
python3-libnvinfer=8.5.2-1+cuda11.8 \
libnvinfer-dev=8.5.2-1+cuda11.8 \
libnvinfer-plugin-dev=8.5.2-1+cuda11.8 \
libnvparsers-dev=8.5.2-1+cuda11.8 \
libnvonnxparsers-dev=8.5.2-1+cuda11.8 \
python3-libnvinfer-dev=8.5.2-1+cuda11.8
#apt-get install libnvinfer-bin=${version} libnvinfer-dev=${version}