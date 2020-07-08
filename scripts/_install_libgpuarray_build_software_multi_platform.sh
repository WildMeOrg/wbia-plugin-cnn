#!/bin/bash
################################
# Build OpenCV

set -ex

export VERSION="04c2892"

export WORKSPACE=$PWD

export VIRTUAL_ENV="/opt/libgpuarray"

# Checkout source, only if it isn't already laying around
if [ ! -d ${WORKSPACE} ]; then
    git clone https://github.com/Theano/libgpuarray.git ${WORKSPACE}/libgpuarray
    cd ${WORKSPACE}/libgpuarray
    git checkout ${VERSION}
    rm -rf ${WORKSPACE}/libgpuarray/build
    mkdir -p ${WORKSPACE}/libgpuarray/build
    cd ${WORKSPACE}/libgpuarray/build
    cmake \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=${VIRTUAL_ENV} \
        ..
    make -j9
    sudo make install
    sudo update_dyld_shared_cache
    cd ..
    python setup.py build_ext -L ${VIRTUAL_ENV}/lib -I ${VIRTUAL_ENV}/include
    pip install -e .
fi

# Return to the workspace
cd ${WORKSPACE}
