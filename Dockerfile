FROM ubuntu:20.04 as cpu_build

# Install dependencies
RUN apt-get update && \
    apt-get --yes install git sudo
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

# Add Tini
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]

# Clone the repo and install the toolkit
RUN git clone --depth 1 --recurse-submodules -j8 https://github.com/opendr-eu/opendr -b install_scripts
WORKDIR "/opendr"
RUN ./bin/install.sh

# Create script for starting Jupyter Notebook
RUN /bin/bash -c "source ./bin/activate.sh; pip3 install jupyter"
RUN echo "#!/bin/bash\n source ./bin/activate.sh\n ./venv/bin/jupyter notebook --port=8888 --no-browser --ip 0.0.0.0 --allow-root" > start.sh
RUN chmod +x start.sh

# Start Jupyter Notebook inside OpenDR
CMD ["./start.sh"]

FROM cpu_build as gpu_build
WORKDIR "/"

ENV NVARCH x86_64
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.2 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441"
ENV NV_CUDA_CUDART_VERSION 10.2.89-1

ENV NV_ML_REPO_ENABLED 1
ENV NV_ML_REPO_URL https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/${NVARCH}
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/${NVARCH}/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list && \
    if [ ! -z ${NV_ML_REPO_ENABLED} ]; then echo "deb ${NV_ML_REPO_URL} /" > /etc/apt/sources.list.d/nvidia-ml.list; fi && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 10.2.89

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-10-2=${NV_CUDA_CUDART_VERSION} \
    cuda-compat-10-2 \
    && ln -s cuda-10.2 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility


ENV NV_CUDA_LIB_VERSION 10.2.89-1
ENV NV_NVTX_VERSION 10.2.89-1
ENV NV_LIBNPP_VERSION 10.2.89-1
ENV NV_LIBCUSPARSE_VERSION 10.2.89-1


ENV NV_LIBCUBLAS_PACKAGE_NAME libcublas10

ENV NV_LIBCUBLAS_VERSION 10.2.2.89-1
ENV NV_LIBCUBLAS_PACKAGE ${NV_LIBCUBLAS_PACKAGE_NAME}=${NV_LIBCUBLAS_VERSION}


ENV NV_LIBNCCL_PACKAGE_NAME "libnccl2"
ENV NV_LIBNCCL_PACKAGE_VERSION 2.11.4-1
ENV NCCL_VERSION 2.11.4
ENV NV_LIBNCCL_PACKAGE ${NV_LIBNCCL_PACKAGE_NAME}=${NV_LIBNCCL_PACKAGE_VERSION}+cuda10.2

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-10-2=${NV_CUDA_LIB_VERSION} \
    cuda-npp-10-2=${NV_LIBNPP_VERSION} \
    cuda-nvtx-10-2=${NV_NVTX_VERSION} \
    cuda-cusparse-10-2=${NV_LIBCUSPARSE_VERSION} \
    ${NV_LIBCUBLAS_PACKAGE} \
    ${NV_LIBNCCL_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

# Keep apt from auto upgrading the cublas and nccl packages. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold ${NV_LIBNCCL_PACKAGE_NAME} ${NV_LIBCUBLAS_PACKAGE_NAME}

#libreadline-dev  libffi-dev 
# Install CUDA 10.2
RUN apt-get --yes install gcc-9 g++-9 wget && \
    # update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 9 --slave /usr/bin/g++ g++ /usr/bin/g++-8 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 8 --slave /usr/bin/g++ g++ /usr/bin/g++-9 && \
    apt-get --yes install zlib1g-dev libbz2-dev libssl-dev libsqlite3-dev && \
    wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run && \
    wget http://people.cs.uchicago.edu/~kauffman/nvidia/cudnn/cudnn-10.2-linux-x64-v8.2.0.53.tgz && \
    sudo apt-get --yes install libxml2 && \
    sh cuda_10.2.89_440.33.01_linux.run --silent --toolkit --override  --librarypath=/usr/local/cuda-10.2 && \
    ln -s /usr/local/cuda-10.2/ /usr/local/cuda 

RUN tar -xzvf cudnn-10.2-linux-x64-v8.2.0.53.tgz
RUN cp cuda/include/cudnn.h /usr/local/cuda/include
RUN cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
RUN chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
RUN bash -c 'echo "/usr/local/cuda-10.2/lib64" > /etc/ld.so.conf.d/nvidia.conf'
RUN ln -sf /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudnn_adv_train.so.8.2.0 /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudnn_adv_train.so.8
RUN ln -sf /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8.2.0 /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8
RUN ln -sf /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8.0.0 /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8
RUN ln -sf /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8.2.0 /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8
RUN ln -sf /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudnn_ops_train.so.8.2.0 /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudnn_ops_train.so.8
RUN ln -sf /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8.2.0 /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8
RUN ln -sf /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudnn.so.8.2.0 /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudnn.so.8
RUN ldconfig

WORKDIR "/opendr"