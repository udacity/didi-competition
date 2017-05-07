FROM osrf/ros:kinetic-desktop-full

# Support for nvidia-docker.
# nvidia-docker volume mounts appropriate libs at /usr/local/nvidia in container
LABEL com.nvidia.volumes.needed="nvidia_driver"
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# The basics
RUN apt-get update && apt-get install -q -y \
        wget \
        pkg-config \
        git-core \
        python \
        python-dev \
        libpcap-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Velodyne driver
RUN . /opt/ros/kinetic/setup.sh && \
    mkdir -p /workspace/src && \
    cd /workspace/src && \
    git clone -q https://github.com/ros-drivers/velodyne.git && \
    catkin_init_workspace && \
    cd .. && \
    catkin_make install

# Install Pip and Python modules
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py \
    && \ 
    pip --no-cache-dir install \
        scipy \
        numpy \
        matplotlib \
        pandas \
        ipykernel \
        jupyter \
        pyyaml \
        shapely \
    && \
    python -m ipykernel.kernelspec

COPY ./viz_entrypoint.sh /
ENTRYPOINT ["/viz_entrypoint.sh"]
CMD ["/bin/bash"]
