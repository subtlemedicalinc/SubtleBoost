# SubtleGad
Gadolinium Contrast Enhancement using Deep Learning

## Pre-processing
```bash
python preprocess.py --path_zero /home/subtle/Data/Stanford/lowcon/Patient_0121/7_AX_BRAVO_PRE --path_low /home/subtle/Data/Stanford/lowcon/Patient_0121/10_AX_BRAVO_+C --path_full /home/subtle/Data/Stanford/lowcon/Patient_0121/13_AX_BRAVO+C --verbose --output data/Patient_0121.npy --discard_start_percent .1 --discard_end_percent .1
```

## Training
```bash
python train.py --data_dir data --verbose --checkpoint test.checkpoint
```
