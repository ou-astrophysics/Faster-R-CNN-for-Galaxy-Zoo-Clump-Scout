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
./run_predictions.py --model zoobot --path /path/to/pngs/
```
The checkpoints for the Faster R-CNN models are not stored with this repository. Please download the models from here: https://s3.msi.umn.edu/FRCNN_clumps/models_final.zip
and extract the files into a folder named `models_final` in the same directory.
