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

enhaog@longo:~/project_lowcon/scripts$ time python subtle_main.py lowcon "../data/data_lowcon/lowcon_0006/011/" precon "../data/data_lowcon/lowcon_0006/007/" highcon "../data/data_lowcon/lowcon_0006/014/" "../data/data_lowcon/lowcon_0006/011_enhanced/"
Subtle Log: event converting dicom to nifti: time 0:00:43.625750, 
Subtle Log: event finishing cor-registration: time 0:00:43.625786, 
Subtle Log: event finishing load data: time 0:00:47.095892, 
Subtle Log: event finishing scaling differet seriesload data: time 0:01:06.047124, 
Subtle Log: event finishing scaling differet seriesload data: time 0:01:06.142844, 
Subtle Log: event finishing init and incldingd differet seriesload data: time 0:01:09.308304, 
Subtle Log: event finishing preparing data for predicting: time 0:01:18.761776, 
332/332 [==============================] - 31s      
predict for data size (332, 512, 512, 10), time 0:00:31.315040
Subtle Log: event finishing predicting data: time 0:01:50.076935, 
Subtle Log: event finishing mixing contrast predictions: time 0:01:53.284466, 
Subtle Log: event finishing rescaling predicted contrasts: time 0:01:56.830787, 
export 336 files using 0:00:15.587733
Subtle Log: event finishing exporting dicom files: time 0:02:12.419120, 
('calling command:', '/usr/local/bin/notify_dispatcher.py ../data/data_lowcon/lowcon_0006/011_enhanced/')

real	2m16.722s
user	1m49.144s
sys	0m25.716s

** On Alienware laptop with GTX1070:
