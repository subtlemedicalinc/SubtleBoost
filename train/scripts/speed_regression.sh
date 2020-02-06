queue_sizes=(1 2 4 8 16 32 64)
num_workers=(1 2 4 8 16 32 64)
file_types=('npy' 'h5' 'h5z')

MAX_DATA_SETS=20
SLICES_PER_INPUT=5
NUM_EPOCHS=3

echo "--- file_type queue_size workers use_multiprocessing ---"

for file_type in ${file_types[@]} ; do
	for queue_size in ${queue_sizes[@]} ; do
		for workers in ${num_workers[@]} ; do

			echo "--- ${file_type} ${queue_size} ${workers} False ---"
			python train.py --data_dir ../data_full/data --file_ext ${file_type} --data_list ../data_full/data_test.txt --num_epochs ${NUM_EPOCHS} --batch_size 16 --shuffle --gpu 3 --checkpoint test5.checkpoint --validation_split 0 --random_seed 723 --log_dir ../logs_tb/ --max_data_sets ${MAX_DATA_SETS} --learn_residual --learning_rate .0001 --num_workers ${workers} --max_queue_size ${queue_size} --slices_per_input ${SLICES_PER_INPUT} --history_file test5.history.npy --id test5 --verbose

			sleep 5
			echo "--- ${file_type} ${queue_size} ${workers} True ---"
			python train.py --data_dir ../data_full/data --file_ext ${file_type} --data_list ../data_full/data_test.txt --num_epochs ${NUM_EPOCHS} --batch_size 16 --shuffle --gpu 3 --checkpoint test5.checkpoint --validation_split 0 --random_seed 723 --log_dir ../logs_tb/ --max_data_sets ${MAX_DATA_SETS} --learn_residual --learning_rate .0001 --num_workers ${workers} --max_queue_size ${queue_size} --slices_per_input ${SLICES_PER_INPUT} --history_file test5.history.npy --id test5 --verbose --use_multiprocessing

			sleep 5
		done
	done
done
