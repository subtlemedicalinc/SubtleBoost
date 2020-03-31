#!/bin/bash

# this script tries to build the entire app into an executable in app/dist

# to build the app, you need to have:
# - python 3.5+
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


if [ -z "$PYTHON" ]; then
  export PYTHON=python3
fi
if [ -z "$PIP" ]; then
  export PIP=pip
fi

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

echo ">>> installing pyinstaller..."
$PIP install "pyinstaller>=3.4,<3.5" > /dev/null
$PIP install --upgrade "setuptools>=45.0.0"

echo ">>> installing dependencies..."
if [ ! -d "subtle_app_utilities_bdist" ]; then
  echo ">>> Missing subtle_app_utilities_bdist!"
  echo ">>> Clone repo and build dependencies first!"
  exit 1
fi

$PIP install --find-links=subtle_app_utilities_bdist -r app/requirements.txt
$PIP install --no-deps deepbrain

echo ">>> installing SimpleElastix"

CUR_DIR=$PWD
mkdir -p elastix
cd elastix
wget -N https://com-subtlemedical-dev-public.s3.amazonaws.com/elastix/elastix-build.tar.gz
tar -zxvf elastix-build.tar.gz
cd build
sed -i "s@\/home\/build@$(pwd)@g" SimpleITK-build/Wrapping/Python/Packaging/setup.py
cd SimpleITK-build/Wrapping/Python
$PIP uninstall -y SimpleITK > /dev/null
$PYTHON Packaging/setup.py install
cd $CUR_DIR
rm -rf elastix
$PYTHON -c "import SimpleITK as sitk; sitk.ElastixImageFilter(); print('SimpleElastix successfully installed');"

echo ">>> installing tensorflow..."
TF_VERSION="1.12.0"
# manually remove tensorflow that may have been installed as dependencies
$PIP uninstall -y tensorflow > /dev/null
if [ ${machine} = "Mac" ]; then
  echo ">>> choosing tensorflow for Mac..."
  $PIP install -U "tensorflow==$TF_VERSION" > /dev/null
elif [ ${machine} = "Linux" ]; then
  echo ">>> choosing tensorflow-gpu for Linux..."
  $PIP install -U "tensorflow-gpu==$TF_VERSION" > /dev/null
else
  echo ">>> Unsupported system ${machine}"
  exit 1
fi

echo ">>> packaging SubtleGAD app..."
cd app && \
  pyinstaller infer.spec > /dev/null && \
  cd ..
# collect libcuda.so.1 if exists
# only tested in nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
mkdir -p ${BUILD_DIR}/libs
if [ -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
  cp /usr/lib/x86_64-linux-gnu/libcuda.so.1 ${BUILD_DIR}/libs/libcuda.so.1
fi
if [ -f /usr/lib64/libcuda.so.1 ]; then
  cp /usr/lib64/libcuda.so.1 ${BUILD_DIR}/libs/libcuda.so.1
fi
if [ -f /usr/lib/x86_64-linux-gnu/libcublas.so.9.0 ]; then
  cp /usr/lib/x86_64-linux-gnu/libcublas.so.9.0  ${BUILD_DIR}/libs/libcublas.so.9.0
fi
if [ -f /usr/local/cuda/lib64/libcublas.so.9.0 ]; then
  cp /usr/local/cuda/lib64/libcublas.so.9.0  ${BUILD_DIR}/libs/libcublas.so.9.0
fi

# export LD_LIBRARY_PATH for convert_models_to_trt to be able to use TensorRT
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${BUILD_DIR}/libs"

# encrypt model files and remove the plain text files
$PYTHON -m subtle.util.encrypt_models --delete-original manifest.json app/models
