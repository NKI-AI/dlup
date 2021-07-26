FROM ubuntu:18.04
ARG PYTHON="3.8"
ARG UNAME="dlup"
ARG BUILD_WORKERS="4"

RUN apt-get -qq update
RUN apt-get update && apt-get install -y libxrender1 build-essential sudo \
    autoconf automake libtool pkg-config libtiff-dev libopenjp2-7-dev libglib2.0-dev \
    libxml++2.6-dev libsqlite3-dev libgdk-pixbuf2.0-dev libgl1-mesa-glx git wget rsync \
    fftw3-dev liblapacke-dev libpng-dev libopenblas-dev libxext-dev jq sudo \
    libfreetype6 libfreetype6-dev \
    # Purge pixman and cairo to be sure
    && apt-get remove libpixman-1-dev libcairo2-dev \
    && apt-get purge libpixman-1-dev libcairo2-dev \
    && apt-get autoremove && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pixman 0.40, as Ubuntu repository holds a version with a bug which can cause difficulties reading thumbnails
RUN cd /tmp \
    && wget https://www.cairographics.org/releases/pixman-0.40.0.tar.gz \
    && tar xvf pixman-0.40.0.tar.gz && rm pixman-0.40.0.tar.gz && cd pixman-0.40.0 \
    && ./configure && make -j$BUILD_WORKERS && make install \
    && cd /tmp && rm -rf pixman-0.40.0

# Install cairo 1.16
RUN cd /tmp \
    && wget https://www.cairographics.org/releases/cairo-1.16.0.tar.xz \
    && tar xvf cairo-1.16.0.tar.xz && rm cairo-1.16.0.tar.xz && cd cairo-1.16.0 \
    && ./configure && make -j$BUILD_WORKERS && make install \
    && cd /tmp && rm -rf cairo-1.16.0

# Install OpenSlide for NKI-AI repository.
RUN git clone https://github.com/NKI-AI/openslide.git /tmp/openslide \
    && cd /tmp/openslide \
    && autoreconf -i \
    && ./configure && make -j$BUILD_WORKERS && make install && ldconfig \
    && cd /tmp && rm -rf openslide

# Make a user
# Rename /home to /users to prevent issues with singularity
RUN mkdir /users && echo $UNAME \
    && adduser --disabled-password --gecos '' --home /users/$UNAME $UNAME \
    && adduser $UNAME sudo \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \

RUN mkdir /$UNAME
USER $UNAME

RUN cd /tmp && wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH "/users/$UNAME/miniconda3/bin:$PATH:$CUDA_ROOT"

# Setup python packages
RUN conda update -n base conda -yq \
    && conda install python=${PYTHON} \
    && conda install astunparse ninja setuptools cmake future requests dataclasses \
    && conda install pyyaml mkl mkl-include setuptools cmake cffi typing boost \
    && conda install tqdm jupyter matplotlib scikit-image pandas joblib -yq \
    && conda install typing_extensions \
    && conda clean -ya \
    && python -m pip install numpy==1.20 tifftools -q \
    # Install openslide-python from NKI-AI
    && python -m pip install git+https://github.com/NKI-AI/openslide-python.git

# Install jupyter config to be able to run in the docker environment
RUN jupyter notebook --generate-config
ENV CONFIG_PATH "/users/$UNAME/.jupyter/jupyter_notebook_config.py"
COPY "docker/jupyter_notebook_config.py" ${CONFIG_PATH}

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
