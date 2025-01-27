ARG TAG=latest
FROM base-arm:${TAG}
WORKDIR /home
ARG CPU_CORES=8
## Set up EPOS
# Install pip wheel for ONNX for Jetson
RUN wget https://nvidia.box.com/shared/static/v59xkrnvederwewo2f1jtv6yurl92xso.whl -O onnxruntime_gpu-1.12.1-cp38-cp38-linux_aarch64.whl &&\
pip3 install --upgrade pip && pip3 install onnxruntime_gpu-1.12.1-cp38-cp38-linux_aarch64.whl

# Install necessary Python modules
COPY ./docker/arm/i6dl-edge-lite-arm/requirements.txt ./requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt


# # Build IGL from source
RUN git clone https://github.com/libigl/libigl-python-bindings.git  

# RUN python3 setup.py build --debug develop

RUN cd libigl-python-bindings && python3 -m pip install ./


RUN apt update -y && apt install -y python-is-python3
ENV REPO_PATH_SRC=.
ENV REPO_PATH=/home/i6DL-Edge-Lite
COPY ${REPO_PATH_SRC}/epos_lib ${REPO_PATH}/epos_lib
COPY ${REPO_PATH_SRC}/external ${REPO_PATH}/external
ENV PYTHONPATH=$REPO_PATH:$PYTHONPATH
ENV PYTHONPATH=$REPO_PATH/external/bop_toolkit:$PYTHONPATH
ENV PYTHONPATH=$REPO_PATH/external/progressive-x/build:$PYTHONPATH
ENV PYTHONPATH=$REPO_PATH/external/slim:$PYTHONPATH

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$REPO_PATH/external/llvm/lib

RUN echo "Using ${CPU_CORES} CPU cores"

RUN mkdir -p $REPO_PATH/external/progressive-x/build &&\
cd $REPO_PATH/external/progressive-x/build &&\
cmake .. -DCMAKE_BUILD_TYPE=Release &&\
make -j${CPU_CORES}

RUN pip install matplotlib rich tqdm plyfile
RUN apt-get install cuda-nvcc-11-4
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV PATH=/usr/local/cuda-11.4/bin:$PATH
RUN pip3 install pycuda --verbose
RUN echo "deb https://repo.download.nvidia.com/jetson/common r35.2 main" >> /etc/apt/sources.list && \
apt-get update && apt-get -y install python3-libnvinfer-dev libnvinfer-bin
