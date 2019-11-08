# SubtleGad
Gadolinium Contrast Enhancement using Deep Learning


- [Install](#install)
- [Structure](#structure)
- [Usage](#usage)
- [Creating experiment configs](#creating-experiment-configs)
  * [Pre-processing](#pre-processing)
  * [Training](#training)
  * [Inference](#inference)
  * [Note on sub experiments](#note-on-sub-experiments)
- [Hyperparameter Search](#hyperparameter-search)
  * [The `tunable` object](#the--tunable--object)
  * [Running the hyperparameter search](#running-the-hyperparameter-search)
  * [HypMonitor dashboard](#hypmonitor-dashboard)


## Install
Currently we use Tensorflow 1.14.0. However, due to a dependency issue, we also install TF 2.0.0.  
To fix this, manually remove TF 2.0.0 after installing the requirements:
```bash
pip install git+https://www.github.com/keras-team/keras-contrib.git # not available in main channel
pip install -r requirements.txt
rm -r /path/to/site-packages/tensorflow*
pip install tensorflow-gpu==1.14.0
```

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
All the three processes in the workflow depend on experiment configs present in different experiment folders under `configs/experiments`. The params are configured in `config.json` and the data lists are configured in `data.json`.

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
    "num_filters_first_conv": 32,
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

### Pre-processing
General template of running the preprocess pipeline is
```
./scripts/batch_preprocess.sh [experiment_name] [log_dir]
```

Example:
```bash
./scripts/batch_preprocess.sh tiantan_sri /mylogs/tiantan_sri_logs/
```

The logs will be written to `[log_dir]` and the preprocessed data will be saved to `out_dir` present in config/preprocess.

### Training
General template of running the training pipeline is
```
./scripts/train.sh [experiment_name] [log_dir]
```

Example:
```bash
./scripts/train.sh tiantan_sri /mylogs/tiantan_sri_logs/
```

The logs will be written to `[log_dir]` and the checkpoint will be saved to `checkpoint_dir` as configured in config/train.

### Inference
General template of running the inference pipeline is
```
./scripts/inference.sh [experiment_name] [log_dir]
```

Example:
```bash
./scripts/inference.sh tiantan_sri /mylogs/tiantan_sri_logs/
```

The logs will be written to `[log_dir]` and the dicoms will be written to `data_dir` as configured in config/inference.

### Note on sub experiments
All three workflows can have sub-experiments i.e multiple experiments under the same config but with minor changes to only a few params. A sample training sub-experiment config would look like

`configs/experiments/train_tiantan/config.json`

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

## Hyperparameter Search

Hyperparameter search experiments can be defined in `configs/hyperparam`.
Example can be found in `configs/hyperparam/loss_weights.json`

The following are the high-level parameters that can be defined in a hyperparam
config

- `name` - name of the hyperparam experiment
- `trials` - number of random trials
- `jobs_per_gpu` - number of parallel training jobs that can be run on a single gpu
- `gpus` - array of GPU IDs that are available for the hyperparam script
- `strategy` - hyperparam search strategy - `random_search` or `grid_search`
- `base_experiment` - base experiment config name from which the default parameters are to be taken
- `base_model` - base model config from which model defaults are to be taken
- `tunable` - set of tunable parameters for the experiment and the model
- `plot` - set of case and slice numbers that are plotted on tensorboard
- `log_dir` - directory path where the hyperparameter script creates a folder
with logs, metrics and tensorboard protobuf objects

### The `tunable` object

The `tunable` object has `experiment` and `model` sub-objects where the
parameters that need to be tuned are defined. `tunable` params can either be a
`range` which has a `low` and a `high` value and the values are randomly
sampled from a uniform distribution or the param can be a `list` with `options`
specified.

**Example**

```json
{
  "tunable": {
    "experiment": {
      "l1_lambda": {
        "type": "range",
        "low": 0,
        "high": 1
      }
    },
    "model": {
      "num_filters_first_conv": {
        "type": "list",
        "options": [8, 16, 24, 32]
      }
    }
  }
}
```

### Running the hyperparameter search

Once the hyperparam config is defined in `configs/hyperparam`, the hyperparam script can be started by running

```
$ ./scripts/hyperparam.sh <hyperparam_name>
```

where `hyperparam_name` is the name of the JSON file you have defined in `configs/hyperparam`.

### HypMonitor dashboard

HypMonitor is a simple web GUI dashboard for monitoring hyperparameter progress
and easy tracking of training logs. Run the following to setup the dashboard

```
$ ./hypmonitor/setup.sh <base_log_path> <port>
$ source ~/.bashrc
```

where the `base_log_path` should be the base path of hyperparameter logs and
and the server is run on the specified `port`.

To start/stop the hypmonitor service run
```
$ hypmonitor [start | stop]
```

The dashboard can be accessed at `http://localhost:<port>` or by clicking the
`Detailed logs` link in the tensorboard's `Text` tab.

The server access logs can be found in `~/.hypmonitor`.
