#!/bin/bash

# this script tries to build the entire app into an executable in app/dist

# to build the app, you need to have:
# - python 3.10+
# - virtualenv for python
# - subtle_app_utilities_bdist folder with utilities build wheel files
# - default model files in app/models folder

# check system type
unameOut="$(uname -s)"
case "${unameOut}" in
  Linux*)     machine=Linux;;
  Darwin*)    machine=Mac;;
  CYGWIN*)    machine=Cygwin;;
  MINGW*)     machine=MinGw;;
  *)          machine="UNKNOWN:${unameOut}"
esac


#if [ -z "$PYTHON" ]; then
export PYTHON=python3.10
#fi
#if [ -z "$PIP" ]; then
export PIP="python3.10 -m pip"
#fi

BUILD_DIR=build
if [ -d ${BUILD_DIR} ]; then
  echo ">>> build dir (${BUILD_DIR}) already exists; deleting"
  rm -rf ${BUILD_DIR}
fi
mkdir -p ${BUILD_DIR}
echo "created build dir: $BUILD_DIR"
if [ ! -d ${BUILD_DIR} ]; then
  echo ">>> build dir (${BUILD_DIR}) does not exist ... "
  exit 1
fi

echo ">>> installing libraries..."
if command -v yum &> /dev/null
then
    yum install -y libSM libXrender libXext libXtst libXi libXdmcp libbsd libglvnd
fi
if command -v apt-get &> /dev/null
then
    apt-get install -y libsm6 libxrender1 libfontconfig1 libxtst6 libxi6 libglvnd
fi

echo ">>> installing pyinstaller..."
$PIP install "pyinstaller==5.9.0" > /dev/null
$PIP install --upgrade "setuptools>=45.0.0"

echo ">>> installing dependencies..."
if [ ! -d "subtle_app_utilities_bdist" ]; then
  echo ">>> Missing subtle_app_utilities_bdist!"
  echo ">>> Clone repo and build dependencies first!"
  exit 1
fi

# install relevant libraries
echo ">>> installing libraries..."
if command -v yum &> /dev/null
then
    yum install -y libSM libXrender libXext libXtst libXi libXdmcp libbsd mesa-libGL
fi
if command -v apt-get &> /dev/null
then
    apt-get install -y libsm6 libxrender1 libfontconfig1 libxtst6 libxi6 libgl1
fi

$PIP install --find-links=subtle_app_utilities_bdist -r app/requirements.txt


echo ">>> packaging SubtleGAD app..."
cd app && \
  $PYTHON -m PyInstaller infer.spec > /dev/null
cd ../
# collect libcuda.so.1 if exists
# only tested in nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
mkdir -p ${BUILD_DIR}/libs
cp /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudart.so.11.0 ${BUILD_DIR}/libs

cp /usr/local/lib/python3.10/site-packages/torch/lib/libcudnn.so.8 ${BUILD_DIR}/libs/libcudnn.so.8
cp /usr/local/lib/python3.10/site-packages/torch/lib/libcudnn_cnn_infer.so.8 ${BUILD_DIR}/libs
cp /usr/local/lib/python3.10/site-packages/torch/lib/libcudnn_ops_infer.so.8 ${BUILD_DIR}/libs
cp /usr/local/lib/python3.10/site-packages/torch/lib/libcudnn_ops_train.so.8 ${BUILD_DIR}/libs

cp /usr/local/cuda-11.2/compat/libcuda.so.460.* ${BUILD_DIR}/libs
cp /usr/local/cuda-11.2/compat/libnvidia-ptxjitcompiler.so.460.* ${BUILD_DIR}/libs
cp /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcublas.so.11 ${BUILD_DIR}/libs
cp /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcublasLt.so.11 ${BUILD_DIR}/libs
cp /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcufft.so.10 ${BUILD_DIR}/libs
cp /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcurand.so.10 ${BUILD_DIR}/libs
cp /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcusolver.so.11 ${BUILD_DIR}/libs
cp /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcusparse.so.11 ${BUILD_DIR}/libs
cp /usr/local/cuda-11.2/targets/x86_64-linux/lib/libnvrtc.so.11.2 ${BUILD_DIR}/libs
cp /usr/local/cuda-11.2/targets/x86_64-linux/lib/libnvrtc-builtins.so.11.2 ${BUILD_DIR}/libs

cp /usr/lib64/libGLX_mesa.so.0 ${BUILD_DIR}/libs
cp /usr/lib64/libxcb.so.1 ${BUILD_DIR}/libs
#cp /usr/lib64/libGL.so.1 ${BUILD_DIR}/libs

mkdir -p ${BUILD_DIR}/bin
cp /usr/local/cuda-11.2/bin/ptxas ${BUILD_DIR}/bin/ptxas


echo ">>> printing identified dependencies"
for entry in "${BUILD_DIR}/libs"/*
do
  echo "$entry"
done

# encrypt model files and remove the plain text files
$PYTHON -m subtle.util.encrypt_models --delete-original manifest.json app/models/
