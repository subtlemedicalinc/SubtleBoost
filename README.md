# SubtleGad
Gadolinium Contrast Enhancement using Deep Learning

## Pre-processing
```bash
python preproccess.py --path_base /home/subtle/Data/Stanford/lowcon/Patient_0121 --verbose --output Patient_0121.npy --discard_start_percent .1 --discard_end_percent .1 --normalize --normalize_fun mean 
```

## Training
```bash
python train.py --data_train_list ../data_full/data_train_small.txt --verbose --num_epochs 20 --batch_size 8 --gpu 0 --checkpoint xyz.checkpoint --validation_split 0. --random_seed 723 --log_dir ../logs_tb/ --max_data_sets 50 --learn_residual --learning_rate .0001 --batch_norm --num_workers 1 --shuffle --history_file history_xyz.npy --id 123xyz
```

or use helper script
```bash
GPU=0 DATA_LIST=../data_full/data_train_small.txt NYM_EPOCHS=20\ 
BATCH_SIZE=8 LEARN_RESIDUAL=1 BATCH_NORM=1 \
LEARNING_RATE=.001 MAX_DATA_SETS=50 NUM_WORKERS=1 ./scripts/train.sh 123xyz
```
