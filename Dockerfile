FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
MAINTAINER chriamue@gmail.com

RUN apt-get update && apt-get install -y \
      build-essential \
      cmake \
      git \
      rpm \
      libatlas-base-dev \
      libatlas-dev \
      libboost-all-dev \
      libgflags-dev \
      libgoogle-glog-dev \
      libhdf5-dev \
      libleveldb-dev \
      liblmdb-dev \
      libopencv-dev \
      libprotobuf-dev \
      libsnappy-dev \
      lsb-release \
      protobuf-compiler \
      python-dev \
      python-numpy \
      python-pip \
      python-setuptools \
      python-scipy \
      sudo \
      wget \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip

RUN git clone https://github.com/BVLC/caffe.git && cd caffe \
    && cat python/requirements.txt | xargs -n1 pip install \
    && mkdir build && cd build && cmake .. && make -j"$(nproc)" all \
    && make install && cd .. && rm -rf caffe
RUN cp -R /caffe/build/include/caffe/proto /caffe/include/caffe

RUN git clone https://github.com/chriamue/openpose \
    && cd openpose && git pull && git checkout cmake && git pull origin cmake \
    && rm -rf 3rdparty/caffe && mkdir -p build && cd build \
    && cmake -DCaffe_DIR=/caffe/build/install .. && make -j"$(nproc)" \
    && make pack && dpkg -i *.deb && cd / && rm -rf openpose

# ADD . /dressup
# RUN cd dressup && mkdir build && cd build && cmake .. && make

RUN cd / && git clone https://github.com/chriamue/dressup.git && cd dressup && mkdir build && cd build && cmake .. && make

WORKDIR /dressup/build

CMD ./dressup
