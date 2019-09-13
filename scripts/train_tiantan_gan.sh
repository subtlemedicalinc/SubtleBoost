export GPU=2
export DATA_DIR=/home/srivathsa/projects/studies/gad/tiantan/preprocess/data
export DATA_LIST=data_lists/data_train_tiantan_gan.txt
export LEARNING_RATE=.001
export MAX_DATA_SETS=100
export BATCH_SIZE=8
export NUM_EPOCHS=100
export QUEUE_SIZE=4
export SLICES_PER_INPUT=7
export SHUFFLE=1
export FILE_EXT=h5
export VAL_SPLIT=0.2
export LOG_DIR=/home/srivathsa/projects/studies/gad/tiantan/train/logs
export TB_DIR=/home/srivathsa/projects/studies/gad/tiantan/train/tb
export HIST_DIR=/home/srivathsa/projects/studies/gad/tiantan/train/history
export CHECKPOINT_DIR=/home/srivathsa/projects/studies/gad/tiantan/train/checkpoints
export L1_LAMBDA=0.15
export SSIM_LAMBDA=0.15
export WLOSS_LAMBDA=0.5
export PERCEPTUAL_LAMBDA=0.2
export NO_SAVE_BEST_ONLY=1
export RESIZE=240
export USE_RESPATH=0
export GAN_MODE=1
./scripts/train.sh
