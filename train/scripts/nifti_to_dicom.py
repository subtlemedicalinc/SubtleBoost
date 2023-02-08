import pydicom
from glob import glob
import nibabel as nib
import numpy as np
from tqdm import tqdm
from subtle.utils.io import write_dicoms
from skimage.transform import resize

def nifti2dicom(fpath_nii, dcm_template, out_dir, patient_id, series_desc, series_num, study_uid):
    nib_arr = nib.load(fpath_nii).get_fdata()
    nib_arr = proc_vol(nib_arr)

    nib_arr = nib_arr.transpose(2, 0, 1)
    nib_arr = np.rot90(nib_arr, axes=(1, 2))

    write_dicoms(
        dcm_template, nib_arr, out_dir, patient_id=patient_id, series_desc_pre='', desc=series_desc,
        series_desc_post='', series_num=series_num, study_uid=study_uid
    )

def proc_vol(data_vol, size=512):
    if data_vol.shape[0] == size:
        return data_vol

    data_vol = resize(data_vol, (size, size, data_vol.shape[2]))
    return data_vol

if __name__ == '__main__':
    snum_map = {
        'T1': 100,
        'T2': 110,
        'MRA': 120
    }

    case_base = '/mnt/datasets/srivathsa/jiang_raid/projects/SubtleGAN/data/IXI/IXI_MRA/split/train'
    bpath = '/mnt/datasets/srivathsa/jiang_raid/projects/SubtleGAN/data/IXI/IXI_MRA/coreg'
    dest_path = '/mnt/datasets/srivathsa/jiang_raid/projects/SubtleGAN/data/IXI/IXI_MRA/dcms'
    dcm_temp = '/mnt/datasets/srivathsa/jiang_raid/projects/SubtleGAN/data/IXI/IXI_MRA/dcms/IXI017-Guys-0698/120_MRA'

    cons = [c for c in snum_map.keys()]

    with open('scripts/ixi_cases.txt', 'r') as cases_file:
        all_cases = [c for c in cases_file.read().split('\n') if len(c) > 0]

    cases = all_cases

    for cnum in tqdm(cases, total=len(cases)):
        try:
            study_uid = pydicom.uid.generate_uid()

            for con_str in cons:
                fpath_nii = '{}/{}/{}-{}.nii.gz'.format(bpath, cnum, cnum, con_str)
                snum = snum_map[con_str]
                ser_desc = '{}_{}'.format(snum, con_str)
                dcm_out = '{}/{}/{}'.format(dest_path, cnum, ser_desc)
                nifti2dicom(fpath_nii, dcm_temp, dcm_out, cnum, ser_desc, snum, study_uid)
        except Exception as err:
            print('Could not process {}:{}'.format(cnum, err))
