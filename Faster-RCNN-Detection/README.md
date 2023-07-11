# Running detections
This script will output a `parquet`-file with the labels and coordinates of the bounding boxes for detected clumps in a set of galaxy images. It takes all the images in one folder and treats the image name as the corresponding `id` for the resulting detection table. For that, the filenames of the images need to be numeric.

The labels are:

```
{
  1: 'clump',
  2: 'odd clump',
}
```

Run the Python-script `run_predictions.py` with the following parameters:

`--model`: Select a model from `resnet`, `resnet_trainable`, `zoobot_clumps`, `zoobot`, `zoobot_trainable`.

`--path`: The path to the folder containing the galaxy cutout image files.

Example execution:

```
python ./run_predictions.py --model zoobot --path /path/to/pngs/
```
The checkpoints for the Faster R-CNN models are not stored with this repository. Please download the models from here: https://s3.msi.umn.edu/FRCNN_clumps/models_final.zip
and extract the files into a folder named `models_final` in the same directory.

# Running post-processing
Run the Python-script `run_post_processing.py` with the following parameters:

`--preds`: Filename of the predictions output file (parquet-file). The path is hard-coded as `PREDICTIONS_FILE_PATH = './predictions/'`.

`--score`: The score threshold to be applied to the predictions. Note, that due to merging of adjacent clumps, post-processing can be executed only for a specific threshold.

Example execution:

```
python ./run_post_processing.py --preds predictions.gzip --score 0.3
```

The script requires galaxy data from a csv-file `cutout_data_enhanced.csv`. Preparing for the photometry measurements, these fields per galaxy image are required:
```
cols = [
    'specobjid', 
    'arcsec_per_side', 'pix_per_arcsec', 'pix_per_side',
    'petroFlux_u',
    'psffwhm_u', 'psffwhm_g', 'psffwhm_r', 'psffwhm_i', 'psffwhm_z',
    'err_fit_m_u', 'err_fit_m_g', 'err_fit_m_r', 'err_fit_m_i', 'err_fit_m_z', 
    'err_fit_b_u', 'err_fit_b_g', 'err_fit_b_r', 'err_fit_b_i', 'err_fit_b_z'
]
```

# Running photometry
Run the Python-script `run_photometry.py` with the following parameters:

`--preds`: Filename of the predictions output file (parquet-file), i.e. after post-processing. The path is hard-coded as `PREDICTIONS_FILE_PATH = './predictions/'`.

`--score`: The score threshold to be applied to the predictions. Note, that due to masking of all the clumps in the host galaxy, photometry can be executed only for a specific threshold.

Example execution:

```
python ./run_photometry.py --preds predictions_post.gzip --score 0.3
```
