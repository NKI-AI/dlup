ARG CUDA="11.2"
ARG CUDNN="8"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu20.04
ARG CUDA
ARG CUDNN
ARG PYTORCH="1.9"
ARG PYTHON="3.8"
ARG UNAME="dlup"


ENV CUDA_PATH /usr/local/cuda
ENV CUDA_ROOT /usr/local/cuda/bin
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64

RUN apt-get -qq update
# libsm6 and libxext6 are needed for cv2
RUN apt-get update && apt-get install -y libxext6 libsm6 libxrender1 build-essential sudo \
    autoconf automake libtool pkg-config libtiff-dev libopenjp2-7-dev libglib2.0-dev libcairo2-dev \
    libxml++2.6-dev libsqlite3-dev libgdk-pixbuf2.0-dev \
    libgl1-mesa-glx git wget rsync tmux nano dcmtk fftw3-dev liblapacke-dev libpng-dev libopenblas-dev jq
RUN  rm -rf /var/lib/apt/lists/*


# Standard openslide cannot read 3dhistech data.
WORKDIR /tmp
RUN git clone https://github.com/NKI-AI/openslide.git
WORKDIR /tmp/openslide
RUN autoreconf -i
RUN ./configure
RUN make
RUN sudo make install
RUN ldconfig


# Make a user
# Rename /home to /users to prevent issues with singularity
RUN mkdir /users
RUN echo $UNAME
RUN adduser --disabled-password --gecos '' --home /users/$UNAME $UNAME
RUN adduser $UNAME sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER $UNAME

WORKDIR /tmp

RUN wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b
ENV PATH "/users/$UNAME/miniconda3/bin:$PATH:$CUDA_ROOT"

# Setup python packages
RUN conda update -n base conda -yq
RUN conda install python=${PYTHON}
RUN conda install pyyaml mkl mkl-include setuptools cmake cffi typing boost
RUN conda install cudatoolkit=${CUDA} torchvision -c pytorch
RUN if [ "nightly$PYTORCH" = "nightly" ] ; then echo "Installing pytorch nightly" && \
    conda install pytorch -c pytorch-nightly; else conda install pytorch=${PYTORCH} -c pytorch ; fi
RUN conda install tqdm jupyter matplotlib scikit-image pandas joblib -yq
RUN python -m pip install openslide-python opencv-python numpy==1.20 tifftools -q

USER root
RUN mkdir /$UNAME
USER $UNAME

# Copy files into the docker
COPY [".", "/$UNAME"]
USER root
WORKDIR /$UNAME
RUN python setup.py install
RUN chown -R $UNAME:$UNAME /$UNAME

USER $UNAME

# Verify installation
RUN python -c 'import openslide'
RUN python -c 'import dlup'

# Provide an open entrypoint for the docker
ENTRYPOINT $0 $@
