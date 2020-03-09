### Change in 0.0.1 and 0.0.2 (initial versions)

- Added scripts and files as required by SubtleApp template
- Added `infer.py` which defines the SubtleGAD app by extending subtle.util.subtle_app.SubtleApp
  - Includes `run` method which defines and runs the `SubtleGADJobType` defined in `subtle_gad_jobs`
- Added `subtle_gad_jobs.py` which has the `_preprocess`, `_process` and `_postprocess` methods which calls the respective methods to perform the different stages of the inference pipeline
- Added `Jenkinsfile`, `run.sh`, and `manifest.json`
- Added `config.yml` where separate jobs are defined for separate manufacturers as there are separate models for each manufacturer. Each job has two series_required - a zero dose and a low dose.
