# SubtleGad
Gadolinium Contrast Enhancement using Deep Learning


## Structure
The `subtle` submodule contains all shared code for I/O, models and generators, data processing, plotting, and command-line arguments.  
The three main programs that use this submodule are `preprocess.py`, `train.py`, and `inference.py`. Each gets its parameters from `subtle/subtle_args.py`, though not all are used.  

## Usage
General workflow:
1. Preprocess data with `preprocess.py`, so that it is ready for training
1. Train a model with the pre-processed data using `train.py` 
1. Test the result on validation/test data using `inference.py`

In general, the helper shell scripts should be used instead of directly calling the python programs. This helps with batch processing and automatic logging

## Pre-processing
```bash
python preprocess.py -h
python preproccess.py --path_base /home/subtle/Data/Stanford/lowcon/Patient_0121 --verbose --output Patient_0121.npy --discard_start_percent .1 --discard_end_percent .1 --normalize --normalize_fun mean 
```
See `scripts/batch_preprocess_*.py` for examples of running batch preprocessing on a list of data

## Training
```bash
python train.py -h
python train.py --data_dir /raid/jon/data_full_tiantan/data --data_list data_lists/data_train_tiantan_20190612.txt --file_ext h5 --shuffle --resize 240 --slice_axis 3 --num_epochs 100 --verbose --batch_size 8 --validation_split 0.1 --learning_rate .001 --slices_per_input 5 --l1_lambda .6 --ssim_lambda .4 --num_channel_first 32 --gpu 1 --checkpoint /raid/jon/checkpoints/mycheckpoint.checkpoint --log_dir /raid/jon/logs_tb --history_file /raid/jon/history/myhistory.npy --id test
```

or use helper script `scripts/train.sh`:
```bash
GPU=0 DATA_LIST=../data_full/data_train_small.txt NYM_EPOCHS=20\ 
BATCH_SIZE=8 LEARN_RESIDUAL=1 BATCH_NORM=1 \
LEARNING_RATE=.001 MAX_DATA_SETS=50 NUM_WORKERS=1 ./scripts/train.sh 123xyz
```

See `scripts/train_*.sh` for examples of running the helper script on different datasets

## Inference
Note: Currently, the inference script requires dicoms/data from all three contrasts. This is so that the full-dose image can be used for internal testing. However, it is not used in the inference pipeline.
```bash
python inference.py -h
python inference.py --path_base /home/subtle/Data/Stanford/lowcon/Patient_0121 --path_out /raid/jon/predictions/dicoms/Patient_0121 --verbose ... # see all args in subtle/subtle_args.py
```

Or use the helper script `scripts/inference.sh`:
```bash
GPU=0 DATA_LIST=../data_full/data_train_small.txt NYM_EPOCHS=20\ 
BATCH_SIZE=8 LEARN_RESIDUAL=1 BATCH_NORM=1 \
LEARNING_RATE=.001 MAX_DATA_SETS=50 NUM_WORKERS=1 ./scripts/inference.sh 123xyz
```

See `scripts/inference_*.sh` for examples of running the helper script on different datasets
