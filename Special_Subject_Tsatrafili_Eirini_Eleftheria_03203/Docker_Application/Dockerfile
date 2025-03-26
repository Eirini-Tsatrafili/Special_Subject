# Base image with Python and OpenCV support
FROM ubuntu:20.04

# Set environment variables to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for Intel RealSense, OpenCV, GStreamer, and Jupyter Notebook
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    git \
    cmake \
    build-essential \
    libusb-1.0-0-dev \
    pkg-config \
    libssl-dev \
    v4l-utils \
    libx11-dev \
    libgtk-3-dev \
    libglfw3-dev \
    libx11-xcb-dev \
    libxcb-dri3-dev \
    libxcb-present-dev \
    libxcb-render0-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libopencv-dev \
    freeglut3-dev \
    mesa-common-dev \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gtk-3.0 \
    libgirepository1.0-dev \
    libgstreamer1.0-dev \
    python3-gst-1.0 \
    python3-opencv \
    libgstrtspserver-1.0-dev \
    gstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-rtsp \
    gstreamer1.0-python3-plugin-loader \
    meson \
    python3-venv \
    ninja-build \
    flex \
    bison \
    ntp \
    udev \
    usbutils \
    && apt-get clean

# Install OpenCV and other Python dependencies
RUN pip3 install opencv-python opencv-contrib-python

# Set up RealSense dependencies and installation
RUN git clone https://github.com/IntelRealSense/librealsense.git /librealsense && \
    cd /librealsense && \
    git checkout development && \
    mkdir build && \
    cd build && \
    cmake .. -DBUILD_EXAMPLES=true -DFORCE_LIBUVC=true && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Build and install Python bindings for RealSense SDK
RUN cd /librealsense/build && \
    cmake .. -DBUILD_PYTHON_BINDINGS=ON -DPYTHON_EXECUTABLE=$(which python3) && \
    make -j$(nproc) && \
    make install

# Create a virtual environment and install Python dependencies
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install \
    torch \
    torchvision \
    ultralytics \
    numpy \
    pyrealsense2 \
    PyGObject \
    notebook

# Set environment variables for Python virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Expose ports for Jupyter Notebook
EXPOSE 8888

# Set the default working directory to /app
WORKDIR /app

# Copy the necessary application files into the container
COPY . /app

# Install the necessary Python dependencies for the project
RUN /opt/venv/bin/pip install -r /app/requirements.txt

# Create the entrypoint for running the code
CMD ["bash", "-c", "jupyter-notebook --no-browser --ip=0.0.0.0 --port=8888"]
