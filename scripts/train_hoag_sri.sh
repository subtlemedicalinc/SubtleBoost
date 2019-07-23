export GPU=1
export DATA_DIR=/home/srivathsa/projects/studies/gad/hoag/preprocess/data
export DATA_LIST=data_lists/data_train_hoag_sri.txt LEARNING_RATE=.001
export MAX_DATA_SETS=100
export BATCH_SIZE=8
export NUM_EPOCHS=70
export QUEUE_SIZE=4
export SLICES_PER_INPUT=5
export SHUFFLE=1
export FILE_EXT=h5
export VAL_SPLIT=0.1
export LOG_DIR=/home/srivathsa/projects/studies/gad/hoag/train/logs
export TB_DIR=/home/srivathsa/projects/studies/gad/hoag/train/tb
export HIST_DIR=/home/srivathsa/projects/studies/gad/hoag/train/history
export CHECKPOINT_DIR=/home/srivathsa/projects/studies/gad/hoag/train/checkpoints
export L1_LAMBDA=.8
export SSIM_LAMBDA=.2
export NO_SAVE_BEST_ONLY=1
export TRAIN_MPR=1
export RESIZE=320
./scripts/train.sh
