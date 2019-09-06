export DATA_RAW=/home/srivathsa/projects/studies/gad/tiantan/data
export DATA_DIR=/home/srivathsa/projects/studies/gad/tiantan/preprocess/data
export DATA_LIST=/home/srivathsa/projects/SubtleGad/data_lists/data_train_tiantan_sri.txt
export DATA_LIST_TEST=/home/srivathsa/projects/SubtleGad/data_lists/data_test_tiantan_sri.txt
export GPU=2
export LEARN_RESIDUAL=0
export SLICES_PER_INPUT=7
export NUM_CHANNEL_FIRST=32
export checkpoint_file=mres_rstudy_split.checkpoint
export LOG_DIR=/home/srivathsa/projects/studies/gad/tiantan/inference
export CHECKPOINT_DIR=/home/srivathsa/projects/studies/gad/tiantan/train/checkpoints
export HIST_DIR=/home/srivathsa/projects/studies/gad/tiantan/train/history
export TB_DIR=/home/srivathsa/projects/studies/gad/tiantan/train/tb
export ZOOM=0
export DESCRIPTION=mres_rstudy_split
export SERIES_NUM=1002
export PREDICT_DIR=/home/srivathsa/projects/studies/gad/tiantan/data
export INFERENCE_MPR=1
export INFERENCE_MPR_AVG=mean
export RESIZE=240
export NUM_ROTATIONS=5
export USE_RESPATH=1
./scripts/inference.sh
