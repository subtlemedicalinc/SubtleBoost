import numpy as np
import SimpleITK as sitk
import os

# subtle utils
def adjust_scale_for_cost(data_low, data_pre,
						 list_scale = np.linspace(0.8,1.2,30),
						 cost_func = lambda x: np.mean(np.abs(x[np.abs(x)>0.1].flatten())),
						 iteration = 1):
	best_scale = 1
	best_cost = np.inf
	for index_iter in xrange(iteration):
		for scale_adjust in list_scale:
			data_low_adjust = data_low * scale_adjust
			diff = data_low_adjust - data_pre
			diff_threshold = diff[np.abs(diff)>0.1]
			diff_cost = np.mean(np.abs(diff_threshold.flatten()))
			if diff_cost < best_cost:
				best_scale = scale_adjust
				best_cost = diff_cost
		delta_scale = list_scale[1]-list_scale[0]
		list_scale = np.linspace(best_scale-delta_scale, best_scale+delta_scale,11)
	return best_scale  

def conduct_coreg(filepath_nifti_low, filepath_nifti_pre, filepath_nifti_low_coreg, overwrite=0):
	if os.path.exists(filepath_nifti_low_coreg) and overwrite==0:
		return
	elastixImageFilter = sitk.ElastixImageFilter()
	elastixImageFilter.SetFixedImage(sitk.ReadImage(filepath_nifti_pre))
	elastixImageFilter.SetMovingImage(sitk.ReadImage(filepath_nifti_low))
	elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
	elastixImageFilter.Execute()
	sitk.WriteImage(elastixImageFilter.GetResultImage(), filepath_nifti_low_coreg)    


