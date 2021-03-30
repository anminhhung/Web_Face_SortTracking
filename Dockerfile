FROM ubuntu:18.04 

RUN apt-get update

RUN apt-get install -y git \
    software-properties-common \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install python3.6 -y && \
    apt install python3-distutils -y && \
    apt install python3.6-dev -y && \
    apt install build-essential -y && \
    apt-get install python3-pip -y && \
    apt update && apt install -y libsm6 libxext6 && \
    apt-get install -y libxrender-dev && \ 
    apt install libgl1-mesa-glx -y
    #ImportError: libGL.so.1: cannot open shared object file: No such file or directory

RUN python3 -m pip install -U pip &&\
    # fix bug can not install skbuild
    python3 -m pip install -U setuptools

# install opencv c++ 
RUN apt install pkg-config
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt-get install -y libopencv-dev 

RUN apt-get install -y libopencv-dev libcaffe-cpu-dev libboost-all-dev \
    libgflags-dev libgtest-dev libc++-dev clang libgoogle-glog-dev \
    libprotobuf-dev protobuf-compiler libopenblas-dev

WORKDIR /Face_Detection

# install lib
COPY requirements.txt .
RUN pip3 install -r requirements.txt 

# make PCN
ADD src src
RUN cd src/PCN && \
    make && \
    make install 

WORKDIR /Face_Detection
COPY . .

EXPOSE 5100

CMD ["/bin/bash", "entrypoint.sh"]