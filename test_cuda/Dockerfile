FROM nvcr.io/nvidia/pytorch:22.12-py3

ARG PYTHON_VERSION=3.8
ARG OPENCV_VERSION=4.7.0

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Lisbon

#ENV PYTHONPATH="/usr/lib/python${PYTHON_VERSION}/site-packages:/usr/local/lib/python${PYTHON_VERSION}/site-packages"

#RUN /usr/bin/python -m pip install --upgrade pip

RUN pip uninstall opencv -y && \
    pip install \
        wandb \
        facenet-pytorch \
        numpy \
        jupyterlab \
        ipywidgets \
        && \
    apt-get -y update -qq --fix-missing && \
    apt-get -y install --no-install-recommends \
        unzip \
        cmake \
        ffmpeg \
        libtbb2 \
        gfortran \
        apt-utils \
        pkg-config \
        checkinstall \
        qt5-default \
        build-essential \
        libopenblas-base \
        libopenblas-dev \
        liblapack-dev \
        libatlas-base-dev \
        #libgtk-3-dev \
        #libavcodec58 \
        libavcodec-dev \
        #libavformat58 \
        libavformat-dev \
        libavutil-dev \
        #libswscale5 \
        libswscale-dev \
        libjpeg8-dev \
        libpng-dev \
        libtiff5-dev \
        #libdc1394-22 \
        libdc1394-22-dev \
        libxine2-dev \
        libv4l-dev \
        libgstreamer1.0 \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-0 \
        libgstreamer-plugins-base1.0-dev \
        libglew-dev \
        libpostproc-dev \
        libeigen3-dev \
        libtbb-dev \
        zlib1g-dev \
        libsm6 \
        libxext6 \
        libxrender1 \
    && \
# # Install OpenCV
#     wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip --progress=bar:force:noscroll && \
#     unzip $OPENCV_VERSION && \
#     mv opencv-4.7.0 /opencv && \
#     wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip -O opencv_contrib.zip --progress=bar:force:noscroll && \
#     unzip opencv_contrib.zip && \
#     mv opencv_contrib-$OPENCV_VERSION /opencv_contrib && \
# # Prepare build
#     mkdir /opencv/build && \
#     cd /opencv/build && \
#     cmake \
#       -D CMAKE_BUILD_TYPE=RELEASE \
#       -D BUILD_PYTHON_SUPPORT=ON \
#       -D BUILD_DOCS=ON \
#       -D BUILD_PERF_TESTS=OFF \
#       -D BUILD_TESTS=OFF \
#       -D CMAKE_INSTALL_PREFIX=/usr/local \
#       -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
#       -D BUILD_opencv_python3=$( [ ${PYTHON_VERSION%%.*} -ge 3 ] && echo "ON" || echo "OFF" ) \
#       -D PYTHON${PYTHON_VERSION%%.*}_EXECUTABLE=$(which python${PYTHON_VERSION}) \
#       -D PYTHON_DEFAULT_EXECUTABLE=$(which python${PYTHON_VERSION}) \
#       -D BUILD_EXAMPLES=OFF \
#       -D WITH_IPP=OFF \
#       -D WITH_FFMPEG=ON \
#       -D WITH_GSTREAMER=ON \
#       -D WITH_V4L=ON \
#       -D WITH_LIBV4L=ON \
#       -D WITH_TBB=ON \
#       -D WITH_QT=ON \
#       -D WITH_OPENGL=ON \
#       -D WITH_CUDA=ON \
#       -D WITH_LAPACK=ON \
#       -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
#       -D CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
#       -D CUDA_ARCH_BIN="5.3 6.1 7.0 7.5" \
#       -D CUDA_ARCH_PTX="" \
#       -D WITH_CUBLAS=ON \
#       -D WITH_NVCUVID=ON \
#       -D ENABLE_FAST_MATH=1 \
#       -D CUDA_FAST_MATH=1 \
#       -D ENABLE_PRECOMPILED_HEADERS=OFF \
#       .. \
#     && \
# # Build, Test and Install
#     cd /opencv/build && \
#     make -j$(nproc) && \
#     make install && \
#     ldconfig && \
# cleaning
   apt-get -y remove \
        unzip \
        cmake \
        gfortran \
        apt-utils \
        pkg-config \
        checkinstall \
        build-essential \
        libopenblas-dev \
        liblapack-dev \
        libatlas-base-dev \
        #libgtk-3-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswscale-dev \
        libjpeg8-dev \
        libpng-dev \
        libtiff5-dev \
        libdc1394-22-dev \
        libxine2-dev \
        libv4l-dev \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libglew-dev \
        libpostproc-dev \
        libeigen3-dev \
        libtbb-dev \
        zlib1g-dev \
    && \
    apt-get autoremove -y && \
    apt-get clean 
    # Call default command.
    #ffmpeg -version && \
    #ldd `which ffmpeg` && \
    #python --version && \
    #python -c "import cv2 ; print(cv2.__version__)"

RUN pip install dload
# Set the working directory
# Copy your code into the container
COPY /. /test_cuda

# Set the working directory to the app directory
WORKDIR /test_cuda