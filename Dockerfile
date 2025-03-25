# ---- Base stage ----
# Install Python and OpenMPI.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS base

ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update && \
	apt-get install -y --no-install-recommends python3.11 python3.11-dev wget curl openssh-client && \
	rm -rf /var/lib/apt/lists/* && \
	update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
	curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
	python get-pip.py && \
	rm get-pip.py

WORKDIR /opt
ENV MPI_HOME=/opt/openmpi
ENV PATH=$MPI_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH
RUN wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.8.tar.gz && \
	tar -xzf openmpi-4.1.8.tar.gz && \
	rm openmpi-4.1.8.tar.gz && \
	cd openmpi-4.1.8 && \
	./configure --prefix=$MPI_HOME --with-cuda=$CUDA_HOME && \
	make -j4 install

# ---- Build stage for Docker ----
# Setting a user that is not root.
FROM base AS docker
RUN groupadd -g 1001 matmul && \
	useradd -u 1001 -g matmul tony

ENV HOME=/home/tony
WORKDIR $HOME
RUN cp /root/.bashrc . && \
	cp /root/.profile . && \
	chown -R tony:matmul $HOME && \
	mkdir .local app && \
	chown -R tony:matmul .local app

WORKDIR $HOME/app
USER tony
ENV PATH=$HOME/.local/bin:$PATH

COPY --chown=tony:matmul requirements.txt .
RUN python -m pip install --no-cache-dir --no-binary=mpi4py -r requirements.txt 

COPY --chown=tony:matmul . .
RUN python -m pip install -e .

# ---- Build stage for Singularity ----
# Not setting a user is recommended for compatibility
# with Singularity, since the container won't run as root.
FROM base AS singularity
ENV HOME=/shared-folder
WORKDIR $HOME
RUN cp /root/.bashrc . && \
	cp /root/.profile . && \
	chmod -R a+rwx $HOME

WORKDIR $HOME/app
COPY --chmod=777 requirements.txt .
RUN python -m pip install --no-cache-dir --no-binary=mpi4py -r requirements.txt

COPY --chmod=777 . .
RUN python -m pip install -e .

