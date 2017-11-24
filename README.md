# SubtleGad
Gadolinium Contrast Enhancement using Deep Learning

* Usage:

time python subtle_main.py lowcon "../data/data_lowcon/lowcon_0006/011/" precon "../data/data_lowcon/lowcon_0006/007/" highcon "../data/data_lowcon/lowcon_0006/014/" "../data/data_lowcon/lowcon_0006/011_enhanced/"

* Dependencies:

sudo apt install python-pip 

sudo apt install cmake

install CUDA 8.0 and cudnn 6.0 if need GPU support

install tensorflow  (pip install --upgrade tensorflow )
https://www.tensorflow.org/install/install_linux#InstallingNativePip

install SimpleElastic (uninstall SimpleITK if needed)
https://simpleelastix.readthedocs.io/GettingStarted.html#compiling-on-linux

pip install keras

pip install dicom2nifti

pip install matplotlib

pip install scikit-image

pip install h5py




* Speed

** On Longo (GTX1080TI):

On longo:

Using TensorFlow backend.

('converted dicom to nifti, take time:', datetime.timedelta(0, 39, 971460))

Subtle Log: event converting dicom to nifti: time 0:00:39.971695, 

Subtle Log: event finishing cor-registration: time 0:00:39.971731, 

Subtle Log: event finishing load data: time 0:00:43.434336, 

Subtle Log: event finishing scaling differet series and delete data: time 0:00:58.069665, 

load model from ../ckpt/model_with_generator_1120.ckpt taking 0:00:00.682827

Subtle Log: event finishing init and incldingd differet seriesload data: time 0:01:01.274893, 

concate 331 images using time 0:00:10.152865

Subtle Log: event finishing preparing data for predicting: time 0:01:11.427896, 

332/332 [==============================] - 35s      

predict for data size (332, 512, 512, 10), time 0:00:35.250851

Subtle Log: event finishing predicting data: time 0:01:46.678865, 

Subtle Log: event finishing mixing contrast predictions: time 0:01:49.940874, 

Subtle Log: event finishing rescaling predicted contrasts: time 0:01:54.197243, 

export 336 files using 0:00:15.671407

Subtle Log: event finishing exporting dicom files: time 0:02:09.869467, 

real	2m14.124s
user	1m49.144s
sys	0m25.716s

** On Alienware laptop with GTX1070:
Coregistration takes 45sec x 2
Prediction takes 32sec (almost the same as 1080TI)
Exporting DICOMS takes 12sec
Total Real time 3min55sec

