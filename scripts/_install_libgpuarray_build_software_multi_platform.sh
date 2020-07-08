#!/bin/bash
################################
# Build OpenCV

set -ex

export VERSION="04c2892"

export WORKSPACE=$PWD

export VIRTUAL_ENV="/opt/libgpuarray"

# Checkout source, only if it isn't already laying around
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
make install || sudo make install
ldconfig || sudo update_dyld_shared_cache || echo "Could not update dynamic libraries"
cd ..
python setup.py build_ext -L ${VIRTUAL_ENV}/lib -I ${VIRTUAL_ENV}/include
pip install -e .

# Return to the workspace
cd ${WORKSPACE}
