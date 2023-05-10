# SubtleGad
Training/inference pipeline for Gadolinium Contrast Enhancement using Deep Learning

## Docker instructions

```
docker build -t gad --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) . -f ./Dockerfile
```

```
docker run -it --runtime=nvidia -v ./train:/opt/train -v <input_data>:/opt/sample_data -v
<checkpoint_dir>:/opt/ckp gad --dcm_pre /opt/sample_data/<pre_contrast> --dcm_post /opt/sample_data/
<post_contrast> --config train/configs/inference/unified_mpr.json --checkpoint /opt/ckp/
unified_mpr_05092023.pth --out_folder <dicom_out>
```
