# conda activate simpleitk
import SimpleITK as sitk
import nibabel as nib
import time
import numpy as np
import glob
import os
from tqdm import tqdm

def register_im(im_fixed, im_moving, param_map=None, verbose=True, im_fixed_spacing=None,
im_moving_spacing=None, max_iter=200, return_params=True, non_rigid=False, fixed_mask=None,
moving_mask=None, ref_fixed=None, ref_moving=None):
    '''
    Image registration using SimpleElastix.
    Register im_moving to im_fixed
    '''

    default_transform = 'translation'

    sim0 = sitk.GetImageFromArray(im_fixed)
    sim1 = sitk.GetImageFromArray(im_moving)

    if im_fixed_spacing is not None:
        sim0.SetSpacing(im_fixed_spacing)

    if im_moving_spacing is not None:
        sim1.SetSpacing(im_moving_spacing)

    if ref_fixed is not None:
        sim0.CopyInformation(ref_fixed)

    if ref_moving is not None:
        sim1.CopyInformation(ref_moving)

    if param_map is None:
        if verbose:
            print("using default '{}' parameter map".format(default_transform))
        param_map = sitk.GetDefaultParameterMap(default_transform)

    if non_rigid:
        param_map = sitk.VectorOfParameterMap()
        param_map.append(sitk.GetDefaultParameterMap('affine'))
        param_map.append(sitk.GetDefaultParameterMap('bspline'))

    param_map['MaximumNumberOfIterations'] = [str(max_iter)]

    ef = sitk.ElastixImageFilter()
    ef.SetLogToConsole(True)
    ef.SetFixedImage(sim0)
    ef.SetMovingImage(sim1)

    if fixed_mask is not None:
        fmask = sitk.GetImageFromArray(fixed_mask.astype(np.uint8))
        fmask.SetSpacing(im_fixed_spacing)
        ef.SetFixedMask(fmask)

    if moving_mask is not None:
        mmask = sitk.GetImageFromArray(moving_mask.astype(np.uint8))
        mmask.SetSpacing(im_moving_spacing)
        ef.SetMovingMask(mmask)

    ef.SetParameterMap(param_map)

    if verbose:
        print('image registration')
        tic = time.time()

    # TODO: Set mask for registration by using ef.SetFixedMask(brain_mask)
    ef.Execute()

    if verbose:
        toc = time.time()
        print('registration done, {:.3} s'.format(toc - tic))

    sim_out = ef.GetResultImage()
    param_map_out = ef.GetTransformParameterMap()

    im_out = sitk.GetArrayFromImage(sim_out)
    im_out = np.clip(im_out, 0, im_out.max())

    if not return_params:
        return im_out, sim_out

    return im_out, sim_out, param_map_out

data_dir = f'/mnt/raid/jiang/projects/SubtleGAN/data/IXI/IXI_dataset/'
save_dir = f'/mnt/raid/jiang/projects/SubtleGAN/data/IXI/IXI_dataset_coregistered/'
cases = glob.glob(f"{data_dir}/IXI*")
for case in tqdm(cases):
    print(case)
    case_name = case.split("/")[-1]
    save_case_dir = os.path.join(save_dir, case_name)
    os.makedirs(save_case_dir, exist_ok=True)
    t1_file = glob.glob(f"{case}/*T1.nii.gz")[0]
    t2_file = glob.glob(f"{case}/*T2.nii.gz")[0]
    pd_file = glob.glob(f"{case}/*PD.nii.gz")[0]
    
    t1_image = sitk.ReadImage(t1_file)
    t2_image = sitk.ReadImage(t2_file)
    pd_image = sitk.ReadImage(pd_file)
    _, t1_r_img, _ = register_im(sitk.GetArrayFromImage(t2_image), sitk.GetArrayFromImage(t1_image), ref_fixed=t2_image, ref_moving=t1_image, param_map=sitk.GetDefaultParameterMap('affine'), verbose=False)
    _, pd_r_img, _ = register_im(sitk.GetArrayFromImage(t2_image), sitk.GetArrayFromImage(pd_image), ref_fixed=t2_image, ref_moving=pd_image, param_map=sitk.GetDefaultParameterMap('affine'), verbose=False)
    
    t1_fn = os.path.join(save_case_dir, t1_file.split("/")[-1])
    t2_fn = os.path.join(save_case_dir, t2_file.split("/")[-1])
    pd_fn = os.path.join(save_case_dir, pd_file.split("/")[-1])
    sitk.WriteImage(t1_r_img, t1_fn)
    sitk.WriteImage(t2_image, t2_fn)
    sitk.WriteImage(pd_r_img, pd_fn)