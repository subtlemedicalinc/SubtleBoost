GPU=$1
idx=$2
num_epochs=$3

GPU=$GPU DATA_DIR=/raid/jon/data_full_stanford/data DATA_LIST=data_lists/data_train_stanford_20190203.txt_${idx} NUM_CHANNEL_FIRST=32 MAX_DATA_SETS=1000 NUM_EPOCHS=${num_epochs} FILE_EXT=h5 VAL_SPLIT=0.05 NUM_WORKERS=4 QUEUE_SIZE=4 LEARN_RESIDUAL=0 SLICES_PER_INPUT=7 BATCH_SIZE=8 LOG_DIR=/raid/jon/logs TB_DIR=/raid/jon/logs_tb HIST_DIR=/raid/jon/history CHECKPOINT_DIR=/raid/jon/checkpoints L1_LAMBDA=.4 SSIM_LAMBDA=.6 NO_SAVE_BEST_ONLY=1 ./scripts/train.sh
