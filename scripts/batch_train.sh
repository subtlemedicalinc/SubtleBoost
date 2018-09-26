export GPU=0
export DATA_DIR=/raid/jon/data_full/data
export DATA_LIST=/home/subtle/jon/dev/SubtleGad/data_lists/data_train.txt
export MAX_DATA_SETS=50
export BATCH_SIZE=8
export NUM_EPOCHS=20
export MULTIPROCESSING=1
export NUM_WORKERS=4
export QUEUE_SIZE=4
export SHUFFLE=1
export SPLIT=0
export SLICES_PER_INPUT=5
export FILE_EXT=npy
export RANDOM_SEED=723
export VAL_SPLIT=0.08
export LOG_DIR=/raid/jon/logs
export HIST_DIR=/raid/jon/history
export CHECKPOINT_DIR=/raid/jon/checkpoints
export TB_DIR=/raid/jon/logs_tb

LEARNING_RATES=( 0.01 0.001 0.0001 )
LEARN_RESIDUALS=( 0 1)
BATCH_NORMS=( 0 1)

for learning_rate in ${LEARNING_RATES[@]} ; do
	for learn_residual in ${LEARN_RESIDUALS[@]} ; do
		for batch_norm in ${BATCH_NORMS[@]} ; do
			LEARNING_RATE=${learning_rate} BATCH_NORM=${batch_norm} LEARN_RESIDUAL=${learn_residual} L1_LAMBDA=.5 SSIM_LAMBDA=.5 ./scripts/train.sh
			LEARNING_RATE=${learning_rate} BATCH_NORM=${batch_norm} LEARN_RESIDUAL=${learn_residual} L1_LAMBDA=1.0 SSIM_LAMBDA=.0 ./scripts/train.sh
			LEARNING_RATE=${learning_rate} BATCH_NORM=${batch_norm} LEARN_RESIDUAL=${learn_residual} L1_LAMBDA=.0 SSIM_LAMBDA=1.0 ./scripts/train.sh
		done
	done
done
