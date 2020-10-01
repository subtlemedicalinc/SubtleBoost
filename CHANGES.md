### Changes since 0.0.5
- Introduced a proc config `allocate_available_gpus` to control app's behaviour in cloud mode. When this config value is `False`, then the app will not allocate available GPUs on its own.
- Changed the output directory structure to `/output/dicoms/series/*.dcm`

### Changes since 0.0.4
- Removed `gpustat` library and replaced it with `GPUtil` to get available GPUs. The `gpustat`
library was having a dependency on `pynvml` which was throwing a driver mismatch error
when packaged with `pyinstaller`.

### Changes since 0.0.3
- Make processing params (e.g. `num_rotations`, `noise_mask_threshold` etc) configurable through
exec config (config.yml)
- Make registration optional for QC testing
- Optionally blur low dose image with anisotropic gaussian blur to get rid of CS streaks
- Resample, zero pad and crop methods to support the new unified model
- When app is installed in CPU mode, `CUDA_VISIBLE_DEVICES` is not specified. In this case,
identify GPU IDs with the required memory

### Changes since 0.0.2
- Integrate the latest Siemens model
- Zero pad and resample to isotropic resolution as required by the model
- Optionally re-run registration for full brain instead of using params from masked brain
registration
- Fix the bug in swapping parameters passed to `get_scale_intensity` method
- Incorporate latest error handling changes in `infer.py` according to app utilities dev branch

### Change in 0.0.1 and 0.0.2 (initial versions)

- Added scripts and files as required by SubtleApp template
- Added `infer.py` which defines the SubtleGAD app by extending subtle.util.subtle_app.SubtleApp
  - Includes `run` method which defines and runs the `SubtleGADJobType` defined in `subtle_gad_jobs`
- Added `subtle_gad_jobs.py` which has the `_preprocess`, `_process` and `_postprocess` methods which calls the respective methods to perform the different stages of the inference pipeline
- Added `Jenkinsfile`, `run.sh`, and `manifest.json`
- Added `config.yml` where separate jobs are defined for separate manufacturers as there are separate models for each manufacturer. Each job has two series_required - a zero dose and a low dose.
