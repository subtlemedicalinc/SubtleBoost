export GPU=0
export DATA_DIR=/home/srivathsa/projects/studies/gad/stanford/preprocess/data
export DATA_LIST=data_lists/data_train_stanford_sri.txt
export LEARNING_RATE=.001
export MAX_DATA_SETS=100
export BATCH_SIZE=8
export NUM_EPOCHS=100
export QUEUE_SIZE=4
export SLICES_PER_INPUT=5
export SHUFFLE=1
export FILE_EXT=h5
export VAL_SPLIT=0.2
export LOG_DIR=/home/srivathsa/projects/studies/gad/stanford/train/logs
export TB_DIR=/home/srivathsa/projects/studies/gad/stanford/train/tb
export HIST_DIR=/home/srivathsa/projects/studies/gad/stanford/train/history
export CHECKPOINT_DIR=/home/srivathsa/projects/studies/gad/stanford/train/checkpoints
export L1_LAMBDA=.8
export SSIM_LAMBDA=.2
export NO_SAVE_BEST_ONLY=1
./scripts/train.sh
