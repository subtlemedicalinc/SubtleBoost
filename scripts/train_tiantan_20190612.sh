DATA_DIR=/raid/jon/data_full_tiantan/data DATA_LIST=data_lists/data_train_tiantan_20190612.txt GPU=2 MAX_DATA_SETS=100 BATCH_SIZE=8 NUM_EPOCHS=100 NUM_WORKERS=4 QUEUE_SIZE=4 MULTIPROCESSING=0 SLICES_PER_INPUT=7 VAL_SPLIT=0.1 LOG_DIR=/raid/jon/logs TB_DIR=/raid/jon/logs_tb HIST_DIR=/raid/jon/history CHECKPOINT_DIR=/raid/jon/checkpoints L1_LAMBDA=.6 SSIM_LAMBDA=.4 NO_SAVE_BEST_ONLY=1 ./scripts/train.sh