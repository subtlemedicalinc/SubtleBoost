export DATA_RAW=/home/srivathsa/projects/studies/gad/tiantan
export DATA_DIR=/home/srivathsa/projects/studies/gad/tiantan_pp/data
export DATA_LIST=data_lists/data_train_tiantan_sri.txt
export LEARN_RESIDUAL=0
export SLICES_PER_INPUT=7
export NUM_CHANNEL_FIRST=32
export LOG_DIR=/home/srivathsa/projects/studies/gad/tiantan_pp/train/logs
export CHECKPOINT_DIR=/home/srivathsa/projects/studies/gad/tiantan_pp/train/checkpoints
export HIST_DIR=/home/srivathsa/projects/studies/gad/tiantan_pp/train/history
export TB_DIR=/home/srivathsa/projects/studies/gad/tiantan_pp/train/logs_tb
export GPU=1
export VAL_SPLIT=0.1
export TRAIN_MPR=1
export RESIZE=240
./scripts/train.sh
