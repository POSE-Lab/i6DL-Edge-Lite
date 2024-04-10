FROM base-x86:latest
WORKDIR /home

## Set up EPOS
# Install necessary Python modules
COPY ./docker/x86/epos-x86-opt/requirements.txt ./requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt
RUN apt-get install sudo

RUN sudo git clone https://github.com/libigl/libigl-python-bindings.git &&\
cd libigl-python-bindings &&\ 
python3 -m pip install libigl

# Directory in the *final* image where the repo will be stored
ENV REPO_PATH_SRC=.
ENV REPO_PATH=/home/epos-opt
#RUN sudo mkdir -p ${REPO_PATH_DST}

# Copy repo contents from the intermediate image to the final 
COPY ${REPO_PATH_SRC} ${REPO_PATH}

ENV PYTHONPATH=$REPO_PATH:$PYTHONPATH
ENV PYTHONPATH=$REPO_PATH/external/bop_renderer/build:$PYTHONPATH
ENV PYTHONPATH=$REPO_PATH/external/bop_toolkit:$PYTHONPATH
ENV PYTHONPATH=$REPO_PATH/external/progressive-x/build:$PYTHONPATH
ENV PYTHONPATH=$REPO_PATH/external/slim:$PYTHONPATH

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$REPO_PATH_DST/external/llvm/lib

# The installation locations of OSMesa and LLVM:
# ENV OSMESA_PREFIX=$REPO_PATH_DST/external/osmesa
# ENV LLVM_PREFIX=$REPO_PATH_DST/external/llvm

# RUN mkdir -p $OSMESA_PREFIX &&\
# mkdir -p $LLVM_PREFIX &&\
# cd $REPO_PATH_DST/external/bop_renderer/osmesa-install &&\
# mkdir -p ./build &&\
# cd ./build &&\
# bash ../osmesa-install.sh

# RUN cd $REPO_PATH_DST/external/bop_renderer &&\
# mkdir -p ./build &&\
# cd ./build &&\
# cmake .. -DCMAKE_BUILD_TYPE=Release &&\
# make -j8

RUN sudo mkdir -p $REPO_PATH/external/progressive-x/build &&\
cd $REPO_PATH/external/progressive-x/build &&\
sudo cmake .. -DCMAKE_BUILD_TYPE=Release &&\
sudo make -j8

RUN pip install matplotlib rich tqdm plyfile
RUN apt update -y && apt install -y python-is-python3