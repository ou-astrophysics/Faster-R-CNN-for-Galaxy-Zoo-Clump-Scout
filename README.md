# Faster-R-CNN-for-Galaxy-Zoo-Clump-Scout
Faster R-CNN and Region Proposal Network using Zoobot finetunded for SDSS images from [Galaxy Zoo](https://github.com/mwalmsley/zoobot): Clump Scout as feature extractor.
## Backbone CNN
Finetune or train Zoobot from scratch, alternatively train EfficientNetB0 from scratch on SDSS images:
* Finetune Zoobot (Tensorflow) - `Finetune Zoobot to find Clumps.ipynb`
* Finetune Zoobot (Pytorch) - `Finetune Zoobot to find Clumps (Pytorch).ipynb`
* Train Zoobot from scratch - `Train Zoobot to find Clumps.ipynb`
* Train EfficientNetB0 from scratch - `Train EfficientNet to find Clumps.ipynb`
* Create TF Record files from png-images - `Create TFRecords for backbone CNN.ipynb`
* Final model - `/Faster_RCNN - TFOD/pre_trained_models/Zoobot` - Finetuned Zoobot model (for clumps from SDSS images), full model (`zoobot_for_clumps.keras`) and checkpoint (`cp-0102-0.84.ckpt`)
## Faster R-CNN
* Folder: Faster-RCNN - Pytorch
  * adjusted `define_model.py` from Zoobot library
  * helper libraries from Torchvision
  * Train Faster R-CNN with Zoobot as backbone model - `GZ2_ClumpScout_Faster_RCNN_Pytorch.ipynb`
  * Dataset class - `SDSSGalaxyDataset.py`
* Folder: Faster-RCNN - TFOD (Tensorflow Object Detection framework)
  * feature extracter and test file
  * adjusted `model_builder.py` with Zoobot feature extracter registered
  * config-file - `faster_rcnn_zoobot.config`
  * Create TF Record files from png-images with bounding boxes and labels - `Create TFRecords for object detection.ipynb`
  * Notebook based on dissected TFOD API, utilising and recreating parts from TFOD: `GZ2_ClumpScout_Zoobot_EfficientDet.ipynb`
