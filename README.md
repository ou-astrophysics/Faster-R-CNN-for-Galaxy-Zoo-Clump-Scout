# Faster-R-CNN-for-Galaxy-Zoo-Clump-Scout
Region Proposal Network using Zoobot finetunded for SDSS images from Galaxy Zoo: Clump Scout as backbone
## Backbone CNN
Finetune or train Zoobot from scratch, alternatively train EfficientNetB0 from scratch on SDSS images:
* Finetune Zoobot - `Finetune Zoobot to find Clumps.ipynb`
* Train Zoobot from scratch - `Train Zoobot to find Clumps.ipynb`
* Train EfficientNetB0 from scratch - `Train EfficientNet to find Clumps.ipynb`
* Create TF Record files from png-images - `Create TFRecords for backbone CNN.ipynb`
* Final model (finetuned Zoobot model) - `zoobot_for_clumps.keras`
## Faster R-CNN
* Create TF Record files from png-images with bounding boxes and labels - `Create TFRecords for object detection.ipynb`
