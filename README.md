# SubtleGad
Gadolinium Contrast Enhancement using Deep Learning


## Structure
The `subtle` submodule contains all shared code for I/O, models and generators, data processing, plotting, and command-line arguments.  
The three main programs that use this submodule are `preprocess.py`, `train.py`, and `inference.py`. Each gets its parameters from the respective experiment configs and `subtle/subtle_args.py`, though not all are used.

## Usage
General workflow:
1. Preprocess data with `preprocess.py`, so that it is ready for training
1. Train a model with the pre-processed data using `train.py`
1. Test the result on validation/test data using `inference.py`

In general, the helper shell scripts should be used instead of directly calling the python programs. This helps with batch processing and automatic logging.

## Creating experiment configs
All the three processes in the workflow depend on experiment configs present in different experiment folders under `experiments`. The params are configured in `config.json` and the data lists are configured in `data.json`.

An example `config.json` would look like

```json
{
  "log_dir": "/my/log/dir",
  "hist_dir": "/my/hist/dir",
  "checkpoint_dir": "/my/check/point",
  "data_dir": "/my/data/dir",
  "gpu": 1,
  "preprocess": {
    "dicom_data": "/my_dicom/data",
    "verbose": 1,
    "out_dir": "/my_data/out_dir",
    "out_dir_plots": "/my_plots/png",
    "discard_start_percent": 0,
    "discard_end_percent": 0,
  },
  "train": {
    "learning_rate": 0.001,
    "batch_size": 8,
    "num_epochs": 70,
    "slices_per_input": 5,
    "val_split": 0
  },
  "inference": {
    "slices_per_input": 5,
    "stats_base": "/my/inference/metdics",
    "num_channel_first": 32,
    "checkpoint": "hoag.checkpoint",
  }
}
```

An example `data.json` would look like
```json
{
  "train": [
    "101_Id_052study",
    "101_Id_055study",
    "101_Id_056study",
    "101_Id_059study",
    "101_Id_060study",
    "101_Id_061study",
    "101_Id_066study"
  ],
  "test": [
    "Id0032Neuro_Brain-16479659",
    "Id0018Neuro_Brain-16473090",
    "101_Id_045study"
  ]
}
```

## Pre-processing
General template of running the preprocess pipeline is
```
./scripts/batch_preprocess.sh [experiment_name] [log_dir]
```

Example:
```bash
./scripts/batch_preprocess.sh tiantan_sri /mylogs/tiantan_sri_logs/
```

The logs will be written to `[log_dir]` and the preprocessed data will be saved to `out_dir` present in config/preprocess.

## Training
General template of running the training pipeline is
```
./scripts/train.sh [experiment_name] [log_dir]
```

Example:
```bash
./scripts/train.sh tiantan_sri /mylogs/tiantan_sri_logs/
```

The logs will be written to `[log_dir]` and the checkpoint will be saved to `checkpoint_dir` as configured in config/train.

## Inference
General template of running the inference pipeline is
```
./scripts/inference.sh [experiment_name] [log_dir]
```

Example:
```bash
./scripts/inference.sh tiantan_sri /mylogs/tiantan_sri_logs/
```

The logs will be written to `[log_dir]` and the dicoms will be written to `data_dir` as configured in config/inference.

## Note on sub experiments
All three workflows can have sub-experiments i.e multiple experiments under the same config but with minor changes to only a few params. A sample training sub-experiment config would look like

`experiments/train_tiantan/config.json`

```
...
"train": {
  "gpu": 1,
  "data_dir": "/my_data",
  "learning_rate": 0.001,
  "batch_size": 8,
  "num_epochs": 70,
  "slices_per_input": 5,
  "val_split": 0,
  "log_dir": "/my/log/dir",
  "hist_dir": "/my/hist/dir",
  "checkpoint_dir": "/my/check/point",
  "without_mpr": {
    "train_mpr": false
  },
  "with_mpr": {
    "train_mpr": true
  }
}
...
```

Using the above config we can run the following sub-experiments
```bash
./scripts/train.sh tiantan_sri/without_mpr /mylogs/tiantan_sri_logs/
./scripts/train.sh tiantan_sri/with_mpr /mylogs/tiantan_sri_logs/
./scripts/train.sh tiantan_sri /mylogs/tiantan_sri_logs/ # runs with default config
```
