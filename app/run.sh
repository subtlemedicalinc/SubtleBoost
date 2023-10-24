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

APP="SubtleGad"
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

echo "SCRIPT: $SCRIPT"
echo "SCRIPTPATH: $SCRIPTPATH"

export LD_LIBRARY_PATH="$SCRIPTPATH/libs"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$SCRIPTPATH/SubtleMR/libs_trt"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$SCRIPTPATH/SubtleMR/libs"
export TF_FORCE_GPU_ALLOW_GROWTH=true


SCRIPT_DIR="$(pwd)/data/apps/$APP"

if [ -d $SCRIPT_DIR ]; then
    cd $SCRIPT_DIR
fi

if [ -z ${LICENSE} ]; then
    LICENSE='empty'
fi

if [ -d "$INPUT_DIR/input_mr" ]; then
    rm -rf $INPUT_DIR/input_mr
fi
mkdir -p  $INPUT_DIR/input_mr

PYCMD=$(cat <<EOF
import yaml

with open('./config.yml', 'r') as file:
    config_keys =yaml.safe_load(file)
config_keys['jobs'][0]['exec_config'].update(series_desc_suffix = "")
config_keys['series'][-1].update(reg_match = "")
config_keys['series'][-1].update(reg_exclude= "")
    
with open('./configcopy.yml', 'w') as file:
    yaml.dump(config_keys, file)
EOF
)

if [ -d "$SCRIPTPATH/SubtleMR/" ]; then

    cd $SCRIPTPATH/SubtleMR/
    chmod +x ./infer/infer
    bash run.sh $INPUT_DIR $INPUT_DIR/input_mr $SCRIPTPATH/SubtleMR/configcopy.yml $SCRIPTPATH/SubtleMR/licenseMR.json 2>&1
    EXIT_CODE_MR=$?

    if  [ "$EXIT_CODE_MR" -ne "0" ]; then
        mv  $INPUT_DIR/input_mr $OUTPUT_DIR
        echo $EXIT_CODE_MR
        exit $EXIT_CODE_MR
    fi

    if [ -d "$INPUT_DIR/input_boost" ]; then
        rm -rf $INPUT_DIR/input_boost
    fi

    mkdir -p $INPUT_DIR/input_boost
    cd ..
    chmod +x ./infer/infer
    ./infer/infer $INPUT_DIR/input_mr $INPUT_DIR/input_boost --config $CONFIG --license $LICENSE 2>&1
    EXIT_CODE_BOOST=$?

    if  [ "$EXIT_CODE_BOOST" -ne "0" ]; then
        mv  $INPUT_DIR/input_boost $OUTPUT_DIR
        echo $EXIT_CODE_BOOST
        exit $EXIT_CODE_BOOST
    fi


    cd ./SubtleMR
    chmod +x ./infer/infer
    bash run.sh $INPUT_DIR/input_boost $OUTPUT_DIR $SCRIPTPATH/SubtleMR/configcopy.yml $SCRIPTPATH/SubtleMR/licenseMR.json 2>&1
    EXIT_CODE_MR2=$?

    if  [ "$EXIT_CODE_MR2" -ne "0" ]; then
        echo $EXIT_CODE_MR2
        exit $EXIT_CODE_MR2
    fi

else
    chmod +x ./infer/infer
    ./infer/infer $INPUT_DIR $OUTPUT_DIR --config $CONFIG --license $LICENSE 2>&1
    EXIT_CODE=$?
fi

rm -rf $INPUT_DIR/input_boost
rm -rf $INPUT_DIR/input_mr

echo "Done!"
echo $EXIT_CODE
exit $EXIT_CODE
