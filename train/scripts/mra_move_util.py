import shutil
import os
from glob import glob
from tqdm import tqdm

from subtle.utils.experiment import get_experiment_data

if __name__ == '__main__':
    src_path = '/home/srivathsa/projects/studies/gad/mra_synth/preprocess/slices'
    dest_path = '/mnt/raid/srivathsa/mra_synth/preprocess/slices'

    train_cases = get_experiment_data('mra_synth', dataset='train')
    val_cases = get_experiment_data('mra_synth', dataset='val')

    all_cases = list(train_cases) + list(val_cases)

    for cnum in tqdm(all_cases, total=len(all_cases)):
        src_case = os.path.join(src_path, cnum)
        dest_case = os.path.join(dest_path, cnum)

        shutil.move(src_case, dest_case)
