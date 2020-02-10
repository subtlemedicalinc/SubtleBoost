#!/bin/sh
#
# Subtle Medical's entry App Script
#
# This script is the main entry point the Subtle Platform uses to execute
# Subtle Apps. Each app can modify the contents of this script.
# This script gets executed with 4 arguments,
#
#   1 - Absolute path to input directory
#   2 - Absolute path to output directory
#   3 - Absolute path to config.yml
#   4 - (optional) Absolute path to license.json
#

APP="SubtleGAD"
INPUT_DIR=$1
OUTPUT_DIR=$2
CONFIG=$3
LICENSE=$4

echo "Starting $APP..."
echo "Input Dir: $INPUT_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "Config File: $CONFIG"
echo "License File: $LICENSE"

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$SCRIPTPATH/libs_trt"
if [ -z "$(ldconfig -p | grep libcuda.so.1)" ]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$SCRIPTPATH/libs"
fi

SCRIPT_DIR="$(pwd)/data/apps/$APP"

if [ -d $SCRIPT_DIR ]; then
    cd $SCRIPT_DIR
fi

if [ -z ${LICENSE} ]; then
    LICENSE='empty'
fi

./infer $INPUT_DIR $OUTPUT_DIR --config $CONFIG --license $LICENSE 2>&1
EXIT_CODE=$?

echo "Done!"
exit $EXIT_CODE
