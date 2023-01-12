import torch
import torchvision
import torch.nn as nn
import define_model
from zoobot.pytorch.estimators import efficientnet_standard, resnet_torchvision_custom

def copy_Zoobot_weights_to_Resnet(model, ckpt_path, device, trainable_layers=0):
    """Returns a model that with pretrained weights from Zoobot.

    Args:
      model: a Resnet model with a standard architecture and parameter names
      ckpt_path: a full path to the Zoobot checkpoint 
      device: torch device
      trainable_layers: number of layers counted from the head of the backbone
        model which will be unfreezed for training

    Returns:
      A Pytorch model
    """

    # load Zoobot (Resnet) and checkpoint
    zoobot = define_model.ZoobotLightningModule(
        output_dim=34,
        question_index_groups=['idx1', 'idx2'],
        include_top=True,
        channels=3,
        use_imagenet_weights=False,
        always_augment=True,
        dropout_rate=0.2,
        architecture_name='resnet_torchvision'
    )
    checkpoint = torch.load(ckpt_path, map_location=torch.device(device))
    zoobot.load_state_dict(checkpoint['state_dict'])

    # copy the weights
    model.backbone.body.conv1.weight = zoobot.model[0][0].weight
    model.backbone.body.bn1.weight = zoobot.model[0][1].weight
    model.backbone.body.bn1.bias = zoobot.model[0][1].bias
    model.backbone.body.bn1.running_mean = zoobot.model[0][1].running_mean
    model.backbone.body.bn1.running_var = zoobot.model[0][1].running_var
    model.backbone.body.layer1[0].conv1.weight = zoobot.model[0][4][0].conv1.weight
    model.backbone.body.layer1[0].bn1.weight = zoobot.model[0][4][0].bn1.weight
    model.backbone.body.layer1[0].bn1.bias = zoobot.model[0][4][0].bn1.bias
    model.backbone.body.layer1[0].bn1.running_mean = zoobot.model[0][4][0].bn1.running_mean
    model.backbone.body.layer1[0].bn1.running_var = zoobot.model[0][4][0].bn1.running_var
    model.backbone.body.layer1[0].conv2.weight = zoobot.model[0][4][0].conv2.weight
    model.backbone.body.layer1[0].bn2.weight = zoobot.model[0][4][0].bn2.weight
    model.backbone.body.layer1[0].bn2.bias = zoobot.model[0][4][0].bn2.bias
    model.backbone.body.layer1[0].bn2.running_mean = zoobot.model[0][4][0].bn2.running_mean
    model.backbone.body.layer1[0].bn2.running_var = zoobot.model[0][4][0].bn2.running_var
    model.backbone.body.layer1[0].conv3.weight = zoobot.model[0][4][0].conv3.weight
    model.backbone.body.layer1[0].bn3.weight = zoobot.model[0][4][0].bn3.weight
    model.backbone.body.layer1[0].bn3.bias = zoobot.model[0][4][0].bn3.bias
    model.backbone.body.layer1[0].bn3.running_mean = zoobot.model[0][4][0].bn3.running_mean
    model.backbone.body.layer1[0].bn3.running_var = zoobot.model[0][4][0].bn3.running_var
    model.backbone.body.layer1[0].downsample[0].weight = zoobot.model[0][4][0].downsample[0].weight
    model.backbone.body.layer1[0].downsample[1].weight = zoobot.model[0][4][0].downsample[1].weight
    model.backbone.body.layer1[0].downsample[1].bias = zoobot.model[0][4][0].downsample[1].bias
    model.backbone.body.layer1[0].downsample[1].running_mean = zoobot.model[0][4][0].downsample[1].running_mean
    model.backbone.body.layer1[0].downsample[1].running_var = zoobot.model[0][4][0].downsample[1].running_var
    model.backbone.body.layer1[1].conv1.weight = zoobot.model[0][4][1].conv1.weight
    model.backbone.body.layer1[1].bn1.weight = zoobot.model[0][4][1].bn1.weight
    model.backbone.body.layer1[1].bn1.bias = zoobot.model[0][4][1].bn1.bias
    model.backbone.body.layer1[1].bn1.running_mean = zoobot.model[0][4][1].bn1.running_mean
    model.backbone.body.layer1[1].bn1.running_var = zoobot.model[0][4][1].bn1.running_var
    model.backbone.body.layer1[1].conv2.weight = zoobot.model[0][4][1].conv2.weight
    model.backbone.body.layer1[1].bn2.weight = zoobot.model[0][4][1].bn2.weight
    model.backbone.body.layer1[1].bn2.bias = zoobot.model[0][4][1].bn2.bias
    model.backbone.body.layer1[1].bn2.running_mean = zoobot.model[0][4][1].bn2.running_mean
    model.backbone.body.layer1[1].bn2.running_var = zoobot.model[0][4][1].bn2.running_var
    model.backbone.body.layer1[1].conv3.weight = zoobot.model[0][4][1].conv3.weight
    model.backbone.body.layer1[1].bn3.weight = zoobot.model[0][4][1].bn3.weight
    model.backbone.body.layer1[1].bn3.bias = zoobot.model[0][4][1].bn3.bias
    model.backbone.body.layer1[1].bn3.running_mean = zoobot.model[0][4][1].bn3.running_mean
    model.backbone.body.layer1[1].bn3.running_var = zoobot.model[0][4][1].bn3.running_var
    model.backbone.body.layer1[2].conv1.weight = zoobot.model[0][4][2].conv1.weight
    model.backbone.body.layer1[2].bn1.weight = zoobot.model[0][4][2].bn1.weight
    model.backbone.body.layer1[2].bn1.bias = zoobot.model[0][4][2].bn1.bias
    model.backbone.body.layer1[2].bn1.running_mean = zoobot.model[0][4][2].bn1.running_mean
    model.backbone.body.layer1[2].bn1.running_var = zoobot.model[0][4][2].bn1.running_var
    model.backbone.body.layer1[2].conv2.weight = zoobot.model[0][4][2].conv2.weight
    model.backbone.body.layer1[2].bn2.weight = zoobot.model[0][4][2].bn2.weight
    model.backbone.body.layer1[2].bn2.bias = zoobot.model[0][4][2].bn2.bias
    model.backbone.body.layer1[2].bn2.running_mean = zoobot.model[0][4][2].bn2.running_mean
    model.backbone.body.layer1[2].bn2.running_var = zoobot.model[0][4][2].bn2.running_var
    model.backbone.body.layer1[2].conv3.weight = zoobot.model[0][4][2].conv3.weight
    model.backbone.body.layer1[2].bn3.weight = zoobot.model[0][4][2].bn3.weight
    model.backbone.body.layer1[2].bn3.bias = zoobot.model[0][4][2].bn3.bias
    model.backbone.body.layer1[2].bn3.running_mean = zoobot.model[0][4][2].bn3.running_mean
    model.backbone.body.layer1[2].bn3.running_var = zoobot.model[0][4][2].bn3.running_var
    model.backbone.body.layer2[0].conv1.weight = zoobot.model[0][5][0].conv1.weight
    model.backbone.body.layer2[0].bn1.weight = zoobot.model[0][5][0].bn1.weight
    model.backbone.body.layer2[0].bn1.bias = zoobot.model[0][5][0].bn1.bias
    model.backbone.body.layer2[0].bn1.running_mean = zoobot.model[0][5][0].bn1.running_mean
    model.backbone.body.layer2[0].bn1.running_var = zoobot.model[0][5][0].bn1.running_var
    model.backbone.body.layer2[0].conv2.weight = zoobot.model[0][5][0].conv2.weight
    model.backbone.body.layer2[0].bn2.weight = zoobot.model[0][5][0].bn2.weight
    model.backbone.body.layer2[0].bn2.bias = zoobot.model[0][5][0].bn2.bias
    model.backbone.body.layer2[0].bn2.running_mean = zoobot.model[0][5][0].bn2.running_mean
    model.backbone.body.layer2[0].bn2.running_var = zoobot.model[0][5][0].bn2.running_var
    model.backbone.body.layer2[0].conv3.weight = zoobot.model[0][5][0].conv3.weight
    model.backbone.body.layer2[0].bn3.weight = zoobot.model[0][5][0].bn3.weight
    model.backbone.body.layer2[0].bn3.bias = zoobot.model[0][5][0].bn3.bias
    model.backbone.body.layer2[0].bn3.running_mean = zoobot.model[0][5][0].bn3.running_mean
    model.backbone.body.layer2[0].bn3.running_var = zoobot.model[0][5][0].bn3.running_var
    model.backbone.body.layer2[0].downsample[0].weight = zoobot.model[0][5][0].downsample[0].weight
    model.backbone.body.layer2[0].downsample[1].weight = zoobot.model[0][5][0].downsample[1].weight
    model.backbone.body.layer2[0].downsample[1].bias = zoobot.model[0][5][0].downsample[1].bias
    model.backbone.body.layer2[0].downsample[1].running_mean = zoobot.model[0][5][0].downsample[1].running_mean
    model.backbone.body.layer2[0].downsample[1].running_var = zoobot.model[0][5][0].downsample[1].running_var
    model.backbone.body.layer2[1].conv1.weight = zoobot.model[0][5][1].conv1.weight
    model.backbone.body.layer2[1].bn1.weight = zoobot.model[0][5][1].bn1.weight
    model.backbone.body.layer2[1].bn1.bias = zoobot.model[0][5][1].bn1.bias
    model.backbone.body.layer2[1].bn1.running_mean = zoobot.model[0][5][1].bn1.running_mean
    model.backbone.body.layer2[1].bn1.running_var = zoobot.model[0][5][1].bn1.running_var
    model.backbone.body.layer2[1].conv2.weight = zoobot.model[0][5][1].conv2.weight
    model.backbone.body.layer2[1].bn2.weight = zoobot.model[0][5][1].bn2.weight
    model.backbone.body.layer2[1].bn2.bias = zoobot.model[0][5][1].bn2.bias
    model.backbone.body.layer2[1].bn2.running_mean = zoobot.model[0][5][1].bn2.running_mean
    model.backbone.body.layer2[1].bn2.running_var = zoobot.model[0][5][1].bn2.running_var
    model.backbone.body.layer2[1].conv3.weight = zoobot.model[0][5][1].conv3.weight
    model.backbone.body.layer2[1].bn3.weight = zoobot.model[0][5][1].bn3.weight
    model.backbone.body.layer2[1].bn3.bias = zoobot.model[0][5][1].bn3.bias
    model.backbone.body.layer2[1].bn3.running_mean = zoobot.model[0][5][1].bn3.running_mean
    model.backbone.body.layer2[1].bn3.running_var = zoobot.model[0][5][1].bn3.running_var
    model.backbone.body.layer2[2].conv1.weight = zoobot.model[0][5][2].conv1.weight
    model.backbone.body.layer2[2].bn1.weight = zoobot.model[0][5][2].bn1.weight
    model.backbone.body.layer2[2].bn1.bias = zoobot.model[0][5][2].bn1.bias
    model.backbone.body.layer2[2].bn1.running_mean = zoobot.model[0][5][2].bn1.running_mean
    model.backbone.body.layer2[2].bn1.running_var = zoobot.model[0][5][2].bn1.running_var
    model.backbone.body.layer2[2].conv2.weight = zoobot.model[0][5][2].conv2.weight
    model.backbone.body.layer2[2].bn2.weight = zoobot.model[0][5][2].bn2.weight
    model.backbone.body.layer2[2].bn2.bias = zoobot.model[0][5][2].bn2.bias
    model.backbone.body.layer2[2].bn2.running_mean = zoobot.model[0][5][2].bn2.running_mean
    model.backbone.body.layer2[2].bn2.running_var = zoobot.model[0][5][2].bn2.running_var
    model.backbone.body.layer2[2].conv3.weight = zoobot.model[0][5][2].conv3.weight
    model.backbone.body.layer2[2].bn3.weight = zoobot.model[0][5][2].bn3.weight
    model.backbone.body.layer2[2].bn3.bias = zoobot.model[0][5][2].bn3.bias
    model.backbone.body.layer2[2].bn3.running_mean = zoobot.model[0][5][2].bn3.running_mean
    model.backbone.body.layer2[2].bn3.running_var = zoobot.model[0][5][2].bn3.running_var
    model.backbone.body.layer2[3].conv1.weight = zoobot.model[0][5][3].conv1.weight
    model.backbone.body.layer2[3].bn1.weight = zoobot.model[0][5][3].bn1.weight
    model.backbone.body.layer2[3].bn1.bias = zoobot.model[0][5][3].bn1.bias
    model.backbone.body.layer2[3].bn1.running_mean = zoobot.model[0][5][3].bn1.running_mean
    model.backbone.body.layer2[3].bn1.running_var = zoobot.model[0][5][3].bn1.running_var
    model.backbone.body.layer2[3].conv2.weight = zoobot.model[0][5][3].conv2.weight
    model.backbone.body.layer2[3].bn2.weight = zoobot.model[0][5][3].bn2.weight
    model.backbone.body.layer2[3].bn2.bias = zoobot.model[0][5][3].bn2.bias
    model.backbone.body.layer2[3].bn2.running_mean = zoobot.model[0][5][3].bn2.running_mean
    model.backbone.body.layer2[3].bn2.running_var = zoobot.model[0][5][3].bn2.running_var
    model.backbone.body.layer2[3].conv3.weight = zoobot.model[0][5][3].conv3.weight
    model.backbone.body.layer2[3].bn3.weight = zoobot.model[0][5][3].bn3.weight
    model.backbone.body.layer2[3].bn3.bias = zoobot.model[0][5][3].bn3.bias
    model.backbone.body.layer2[3].bn3.running_mean = zoobot.model[0][5][3].bn3.running_mean
    model.backbone.body.layer2[3].bn3.running_var = zoobot.model[0][5][3].bn3.running_var
    model.backbone.body.layer3[0].conv1.weight = zoobot.model[0][6][0].conv1.weight
    model.backbone.body.layer3[0].bn1.weight = zoobot.model[0][6][0].bn1.weight
    model.backbone.body.layer3[0].bn1.bias = zoobot.model[0][6][0].bn1.bias
    model.backbone.body.layer3[0].bn1.running_mean = zoobot.model[0][6][0].bn1.running_mean
    model.backbone.body.layer3[0].bn1.running_var = zoobot.model[0][6][0].bn1.running_var
    model.backbone.body.layer3[0].conv2.weight = zoobot.model[0][6][0].conv2.weight
    model.backbone.body.layer3[0].bn2.weight = zoobot.model[0][6][0].bn2.weight
    model.backbone.body.layer3[0].bn2.bias = zoobot.model[0][6][0].bn2.bias
    model.backbone.body.layer3[0].bn2.running_mean = zoobot.model[0][6][0].bn2.running_mean
    model.backbone.body.layer3[0].bn2.running_var = zoobot.model[0][6][0].bn2.running_var
    model.backbone.body.layer3[0].conv3.weight = zoobot.model[0][6][0].conv3.weight
    model.backbone.body.layer3[0].bn3.weight = zoobot.model[0][6][0].bn3.weight
    model.backbone.body.layer3[0].bn3.bias = zoobot.model[0][6][0].bn3.bias
    model.backbone.body.layer3[0].bn3.running_mean = zoobot.model[0][6][0].bn3.running_mean
    model.backbone.body.layer3[0].bn3.running_var = zoobot.model[0][6][0].bn3.running_var
    model.backbone.body.layer3[0].downsample[0].weight = zoobot.model[0][6][0].downsample[0].weight
    model.backbone.body.layer3[0].downsample[1].weight = zoobot.model[0][6][0].downsample[1].weight
    model.backbone.body.layer3[0].downsample[1].bias = zoobot.model[0][6][0].downsample[1].bias
    model.backbone.body.layer3[0].downsample[1].running_mean = zoobot.model[0][6][0].downsample[1].running_mean
    model.backbone.body.layer3[0].downsample[1].running_var = zoobot.model[0][6][0].downsample[1].running_var
    model.backbone.body.layer3[1].conv1.weight = zoobot.model[0][6][1].conv1.weight
    model.backbone.body.layer3[1].bn1.weight = zoobot.model[0][6][1].bn1.weight
    model.backbone.body.layer3[1].bn1.bias = zoobot.model[0][6][1].bn1.bias
    model.backbone.body.layer3[1].bn1.running_mean = zoobot.model[0][6][1].bn1.running_mean
    model.backbone.body.layer3[1].bn1.running_var = zoobot.model[0][6][1].bn1.running_var
    model.backbone.body.layer3[1].conv2.weight = zoobot.model[0][6][1].conv2.weight
    model.backbone.body.layer3[1].bn2.weight = zoobot.model[0][6][1].bn2.weight
    model.backbone.body.layer3[1].bn2.bias = zoobot.model[0][6][1].bn2.bias
    model.backbone.body.layer3[1].bn2.running_mean = zoobot.model[0][6][1].bn2.running_mean
    model.backbone.body.layer3[1].bn2.running_var = zoobot.model[0][6][1].bn2.running_var
    model.backbone.body.layer3[1].conv3.weight = zoobot.model[0][6][1].conv3.weight
    model.backbone.body.layer3[1].bn3.weight = zoobot.model[0][6][1].bn3.weight
    model.backbone.body.layer3[1].bn3.bias = zoobot.model[0][6][1].bn3.bias
    model.backbone.body.layer3[1].bn3.running_mean = zoobot.model[0][6][1].bn3.running_mean
    model.backbone.body.layer3[1].bn3.running_var = zoobot.model[0][6][1].bn3.running_var
    model.backbone.body.layer3[2].conv1.weight = zoobot.model[0][6][2].conv1.weight
    model.backbone.body.layer3[2].bn1.weight = zoobot.model[0][6][2].bn1.weight
    model.backbone.body.layer3[2].bn1.bias = zoobot.model[0][6][2].bn1.bias
    model.backbone.body.layer3[2].bn1.running_mean = zoobot.model[0][6][2].bn1.running_mean
    model.backbone.body.layer3[2].bn1.running_var = zoobot.model[0][6][2].bn1.running_var
    model.backbone.body.layer3[2].conv2.weight = zoobot.model[0][6][2].conv2.weight
    model.backbone.body.layer3[2].bn2.weight = zoobot.model[0][6][2].bn2.weight
    model.backbone.body.layer3[2].bn2.bias = zoobot.model[0][6][2].bn2.bias
    model.backbone.body.layer3[2].bn2.running_mean = zoobot.model[0][6][2].bn2.running_mean
    model.backbone.body.layer3[2].bn2.running_var = zoobot.model[0][6][2].bn2.running_var
    model.backbone.body.layer3[2].conv3.weight = zoobot.model[0][6][2].conv3.weight
    model.backbone.body.layer3[2].bn3.weight = zoobot.model[0][6][2].bn3.weight
    model.backbone.body.layer3[2].bn3.bias = zoobot.model[0][6][2].bn3.bias
    model.backbone.body.layer3[2].bn3.running_mean = zoobot.model[0][6][2].bn3.running_mean
    model.backbone.body.layer3[2].bn3.running_var = zoobot.model[0][6][2].bn3.running_var
    model.backbone.body.layer3[3].conv1.weight = zoobot.model[0][6][3].conv1.weight
    model.backbone.body.layer3[3].bn1.weight = zoobot.model[0][6][3].bn1.weight
    model.backbone.body.layer3[3].bn1.bias = zoobot.model[0][6][3].bn1.bias
    model.backbone.body.layer3[3].bn1.running_mean = zoobot.model[0][6][3].bn1.running_mean
    model.backbone.body.layer3[3].bn1.running_var = zoobot.model[0][6][3].bn1.running_var
    model.backbone.body.layer3[3].conv2.weight = zoobot.model[0][6][3].conv2.weight
    model.backbone.body.layer3[3].bn2.weight = zoobot.model[0][6][3].bn2.weight
    model.backbone.body.layer3[3].bn2.bias = zoobot.model[0][6][3].bn2.bias
    model.backbone.body.layer3[3].bn2.running_mean = zoobot.model[0][6][3].bn2.running_mean
    model.backbone.body.layer3[3].bn2.running_var = zoobot.model[0][6][3].bn2.running_var
    model.backbone.body.layer3[3].conv3.weight = zoobot.model[0][6][3].conv3.weight
    model.backbone.body.layer3[3].bn3.weight = zoobot.model[0][6][3].bn3.weight
    model.backbone.body.layer3[3].bn3.bias = zoobot.model[0][6][3].bn3.bias
    model.backbone.body.layer3[3].bn3.running_mean = zoobot.model[0][6][3].bn3.running_mean
    model.backbone.body.layer3[3].bn3.running_var = zoobot.model[0][6][3].bn3.running_var
    model.backbone.body.layer3[4].conv1.weight = zoobot.model[0][6][4].conv1.weight
    model.backbone.body.layer3[4].bn1.weight = zoobot.model[0][6][4].bn1.weight
    model.backbone.body.layer3[4].bn1.bias = zoobot.model[0][6][4].bn1.bias
    model.backbone.body.layer3[4].bn1.running_mean = zoobot.model[0][6][4].bn1.running_mean
    model.backbone.body.layer3[4].bn1.running_var = zoobot.model[0][6][4].bn1.running_var
    model.backbone.body.layer3[4].conv2.weight = zoobot.model[0][6][4].conv2.weight
    model.backbone.body.layer3[4].bn2.weight = zoobot.model[0][6][4].bn2.weight
    model.backbone.body.layer3[4].bn2.bias = zoobot.model[0][6][4].bn2.bias
    model.backbone.body.layer3[4].bn2.running_mean = zoobot.model[0][6][4].bn2.running_mean
    model.backbone.body.layer3[4].bn2.running_var = zoobot.model[0][6][4].bn2.running_var
    model.backbone.body.layer3[4].conv3.weight = zoobot.model[0][6][4].conv3.weight
    model.backbone.body.layer3[4].bn3.weight = zoobot.model[0][6][4].bn3.weight
    model.backbone.body.layer3[4].bn3.bias = zoobot.model[0][6][4].bn3.bias
    model.backbone.body.layer3[4].bn3.running_mean = zoobot.model[0][6][4].bn3.running_mean
    model.backbone.body.layer3[4].bn3.running_var = zoobot.model[0][6][4].bn3.running_var
    model.backbone.body.layer3[5].conv1.weight = zoobot.model[0][6][5].conv1.weight
    model.backbone.body.layer3[5].bn1.weight = zoobot.model[0][6][5].bn1.weight
    model.backbone.body.layer3[5].bn1.bias = zoobot.model[0][6][5].bn1.bias
    model.backbone.body.layer3[5].bn1.running_mean = zoobot.model[0][6][5].bn1.running_mean
    model.backbone.body.layer3[5].bn1.running_var = zoobot.model[0][6][5].bn1.running_var
    model.backbone.body.layer3[5].conv2.weight = zoobot.model[0][6][5].conv2.weight
    model.backbone.body.layer3[5].bn2.weight = zoobot.model[0][6][5].bn2.weight
    model.backbone.body.layer3[5].bn2.bias = zoobot.model[0][6][5].bn2.bias
    model.backbone.body.layer3[5].bn2.running_mean = zoobot.model[0][6][5].bn2.running_mean
    model.backbone.body.layer3[5].bn2.running_var = zoobot.model[0][6][5].bn2.running_var
    model.backbone.body.layer3[5].conv3.weight = zoobot.model[0][6][5].conv3.weight
    model.backbone.body.layer3[5].bn3.weight = zoobot.model[0][6][5].bn3.weight
    model.backbone.body.layer3[5].bn3.bias = zoobot.model[0][6][5].bn3.bias
    model.backbone.body.layer3[5].bn3.running_mean = zoobot.model[0][6][5].bn3.running_mean
    model.backbone.body.layer3[5].bn3.running_var = zoobot.model[0][6][5].bn3.running_var
    model.backbone.body.layer4[0].conv1.weight = zoobot.model[0][7][0].conv1.weight
    model.backbone.body.layer4[0].bn1.weight = zoobot.model[0][7][0].bn1.weight
    model.backbone.body.layer4[0].bn1.bias = zoobot.model[0][7][0].bn1.bias
    model.backbone.body.layer4[0].bn1.running_mean = zoobot.model[0][7][0].bn1.running_mean
    model.backbone.body.layer4[0].bn1.running_var = zoobot.model[0][7][0].bn1.running_var
    model.backbone.body.layer4[0].conv2.weight = zoobot.model[0][7][0].conv2.weight
    model.backbone.body.layer4[0].bn2.weight = zoobot.model[0][7][0].bn2.weight
    model.backbone.body.layer4[0].bn2.bias = zoobot.model[0][7][0].bn2.bias
    model.backbone.body.layer4[0].bn2.running_mean = zoobot.model[0][7][0].bn2.running_mean
    model.backbone.body.layer4[0].bn2.running_var = zoobot.model[0][7][0].bn2.running_var
    model.backbone.body.layer4[0].conv3.weight = zoobot.model[0][7][0].conv3.weight
    model.backbone.body.layer4[0].bn3.weight = zoobot.model[0][7][0].bn3.weight
    model.backbone.body.layer4[0].bn3.bias = zoobot.model[0][7][0].bn3.bias
    model.backbone.body.layer4[0].bn3.running_mean = zoobot.model[0][7][0].bn3.running_mean
    model.backbone.body.layer4[0].bn3.running_var = zoobot.model[0][7][0].bn3.running_var
    model.backbone.body.layer4[0].downsample[0].weight = zoobot.model[0][7][0].downsample[0].weight
    model.backbone.body.layer4[0].downsample[1].weight = zoobot.model[0][7][0].downsample[1].weight
    model.backbone.body.layer4[0].downsample[1].bias = zoobot.model[0][7][0].downsample[1].bias
    model.backbone.body.layer4[0].downsample[1].running_mean = zoobot.model[0][7][0].downsample[1].running_mean
    model.backbone.body.layer4[0].downsample[1].running_var = zoobot.model[0][7][0].downsample[1].running_var
    model.backbone.body.layer4[1].conv1.weight = zoobot.model[0][7][1].conv1.weight
    model.backbone.body.layer4[1].bn1.weight = zoobot.model[0][7][1].bn1.weight
    model.backbone.body.layer4[1].bn1.bias = zoobot.model[0][7][1].bn1.bias
    model.backbone.body.layer4[1].bn1.running_mean = zoobot.model[0][7][1].bn1.running_mean
    model.backbone.body.layer4[1].bn1.running_var = zoobot.model[0][7][1].bn1.running_var
    model.backbone.body.layer4[1].conv2.weight = zoobot.model[0][7][1].conv2.weight
    model.backbone.body.layer4[1].bn2.weight = zoobot.model[0][7][1].bn2.weight
    model.backbone.body.layer4[1].bn2.bias = zoobot.model[0][7][1].bn2.bias
    model.backbone.body.layer4[1].bn2.running_mean = zoobot.model[0][7][1].bn2.running_mean
    model.backbone.body.layer4[1].bn2.running_var = zoobot.model[0][7][1].bn2.running_var
    model.backbone.body.layer4[1].conv3.weight = zoobot.model[0][7][1].conv3.weight
    model.backbone.body.layer4[1].bn3.weight = zoobot.model[0][7][1].bn3.weight
    model.backbone.body.layer4[1].bn3.bias = zoobot.model[0][7][1].bn3.bias
    model.backbone.body.layer4[1].bn3.running_mean = zoobot.model[0][7][1].bn3.running_mean
    model.backbone.body.layer4[1].bn3.running_var = zoobot.model[0][7][1].bn3.running_var
    model.backbone.body.layer4[2].conv1.weight = zoobot.model[0][7][2].conv1.weight
    model.backbone.body.layer4[2].bn1.weight = zoobot.model[0][7][2].bn1.weight
    model.backbone.body.layer4[2].bn1.bias = zoobot.model[0][7][2].bn1.bias
    model.backbone.body.layer4[2].bn1.running_mean = zoobot.model[0][7][2].bn1.running_mean
    model.backbone.body.layer4[2].bn1.running_var = zoobot.model[0][7][2].bn1.running_var
    model.backbone.body.layer4[2].conv2.weight = zoobot.model[0][7][2].conv2.weight
    model.backbone.body.layer4[2].bn2.weight = zoobot.model[0][7][2].bn2.weight
    model.backbone.body.layer4[2].bn2.bias = zoobot.model[0][7][2].bn2.bias
    model.backbone.body.layer4[2].bn2.running_mean = zoobot.model[0][7][2].bn2.running_mean
    model.backbone.body.layer4[2].bn2.running_var = zoobot.model[0][7][2].bn2.running_var
    model.backbone.body.layer4[2].conv3.weight = zoobot.model[0][7][2].conv3.weight
    model.backbone.body.layer4[2].bn3.weight = zoobot.model[0][7][2].bn3.weight
    model.backbone.body.layer4[2].bn3.bias = zoobot.model[0][7][2].bn3.bias
    model.backbone.body.layer4[2].bn3.running_mean = zoobot.model[0][7][2].bn3.running_mean
    model.backbone.body.layer4[2].bn3.running_var = zoobot.model[0][7][2].bn3.running_var

    # make sure, backbone layers are freezed after copying the weights
    for name, parameter in model.named_parameters():
        if name.startswith('backbone.body.'):
            parameter.requires_grad = False
    
    # unfreeze selected layers
    layers_to_train = ['backbone.body.layer4', 'backbone.body.layer3', 'backbone.body.layer2', 'backbone.body.layer1', 'backbone.body.conv1'][:trainable_layers]
    
    for layer in layers_to_train:
        for name, parameter in model.named_parameters():
            if name.startswith(layer):
                parameter.requires_grad_(True)

    return model


def copy_Zoobot_clumps_weights_to_Resnet(model, ckpt_path, device, trainable_layers=0):
    """Returns a model that with pretrained weights from Zoobot.

    Args:
      model: a Resnet model with a standard architecture and parameter names
      ckpt_path: a full path to the Zoobot checkpoint 
      device: torch device
      trainable_layers: number of layers counted from the head of the backbone
        model which will be unfreezed for training

    Returns:
      A Pytorch model
    """

    # Get Zoobot model and weights - Resnet
    # needs to be model with head, otherwise the checkpoint won't fit
    zoobot = define_model.ZoobotLightningModule(
        output_dim=34,
        question_index_groups=['idx1', 'idx2'],
        include_top=True,
        channels=3,
        use_imagenet_weights=False,
        always_augment=True,
        dropout_rate=0.2,
        architecture_name='resnet_torchvision'
    )

    # select layers for feature map
    zoobot_clumps = zoobot.model[0][0:10]
    
    zoobot_clumps.classifier = nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(in_features=2048, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=2),
    )

    checkpoint = torch.load(ckpt_path, map_location=torch.device(device))
    zoobot_clumps.load_state_dict(checkpoint)

    # copy the weights
    model.backbone.body.conv1.weight = zoobot_clumps[0].weight
    model.backbone.body.bn1.weight = zoobot_clumps[1].weight
    model.backbone.body.bn1.bias = zoobot_clumps[1].bias
    model.backbone.body.bn1.running_mean = zoobot_clumps[1].running_mean
    model.backbone.body.bn1.running_var = zoobot_clumps[1].running_var
    model.backbone.body.layer1[0].conv1.weight = zoobot_clumps[4][0].conv1.weight
    model.backbone.body.layer1[0].bn1.weight = zoobot_clumps[4][0].bn1.weight
    model.backbone.body.layer1[0].bn1.bias = zoobot_clumps[4][0].bn1.bias
    model.backbone.body.layer1[0].bn1.running_mean = zoobot_clumps[4][0].bn1.running_mean
    model.backbone.body.layer1[0].bn1.running_var = zoobot_clumps[4][0].bn1.running_var
    model.backbone.body.layer1[0].conv2.weight = zoobot_clumps[4][0].conv2.weight
    model.backbone.body.layer1[0].bn2.weight = zoobot_clumps[4][0].bn2.weight
    model.backbone.body.layer1[0].bn2.bias = zoobot_clumps[4][0].bn2.bias
    model.backbone.body.layer1[0].bn2.running_mean = zoobot_clumps[4][0].bn2.running_mean
    model.backbone.body.layer1[0].bn2.running_var = zoobot_clumps[4][0].bn2.running_var
    model.backbone.body.layer1[0].conv3.weight = zoobot_clumps[4][0].conv3.weight
    model.backbone.body.layer1[0].bn3.weight = zoobot_clumps[4][0].bn3.weight
    model.backbone.body.layer1[0].bn3.bias = zoobot_clumps[4][0].bn3.bias
    model.backbone.body.layer1[0].bn3.running_mean = zoobot_clumps[4][0].bn3.running_mean
    model.backbone.body.layer1[0].bn3.running_var = zoobot_clumps[4][0].bn3.running_var
    model.backbone.body.layer1[0].downsample[0].weight = zoobot_clumps[4][0].downsample[0].weight
    model.backbone.body.layer1[0].downsample[1].weight = zoobot_clumps[4][0].downsample[1].weight
    model.backbone.body.layer1[0].downsample[1].bias = zoobot_clumps[4][0].downsample[1].bias
    model.backbone.body.layer1[0].downsample[1].running_mean = zoobot_clumps[4][0].downsample[1].running_mean
    model.backbone.body.layer1[0].downsample[1].running_var = zoobot_clumps[4][0].downsample[1].running_var
    model.backbone.body.layer1[1].conv1.weight = zoobot_clumps[4][1].conv1.weight
    model.backbone.body.layer1[1].bn1.weight = zoobot_clumps[4][1].bn1.weight
    model.backbone.body.layer1[1].bn1.bias = zoobot_clumps[4][1].bn1.bias
    model.backbone.body.layer1[1].bn1.running_mean = zoobot_clumps[4][1].bn1.running_mean
    model.backbone.body.layer1[1].bn1.running_var = zoobot_clumps[4][1].bn1.running_var
    model.backbone.body.layer1[1].conv2.weight = zoobot_clumps[4][1].conv2.weight
    model.backbone.body.layer1[1].bn2.weight = zoobot_clumps[4][1].bn2.weight
    model.backbone.body.layer1[1].bn2.bias = zoobot_clumps[4][1].bn2.bias
    model.backbone.body.layer1[1].bn2.running_mean = zoobot_clumps[4][1].bn2.running_mean
    model.backbone.body.layer1[1].bn2.running_var = zoobot_clumps[4][1].bn2.running_var
    model.backbone.body.layer1[1].conv3.weight = zoobot_clumps[4][1].conv3.weight
    model.backbone.body.layer1[1].bn3.weight = zoobot_clumps[4][1].bn3.weight
    model.backbone.body.layer1[1].bn3.bias = zoobot_clumps[4][1].bn3.bias
    model.backbone.body.layer1[1].bn3.running_mean = zoobot_clumps[4][1].bn3.running_mean
    model.backbone.body.layer1[1].bn3.running_var = zoobot_clumps[4][1].bn3.running_var
    model.backbone.body.layer1[2].conv1.weight = zoobot_clumps[4][2].conv1.weight
    model.backbone.body.layer1[2].bn1.weight = zoobot_clumps[4][2].bn1.weight
    model.backbone.body.layer1[2].bn1.bias = zoobot_clumps[4][2].bn1.bias
    model.backbone.body.layer1[2].bn1.running_mean = zoobot_clumps[4][2].bn1.running_mean
    model.backbone.body.layer1[2].bn1.running_var = zoobot_clumps[4][2].bn1.running_var
    model.backbone.body.layer1[2].conv2.weight = zoobot_clumps[4][2].conv2.weight
    model.backbone.body.layer1[2].bn2.weight = zoobot_clumps[4][2].bn2.weight
    model.backbone.body.layer1[2].bn2.bias = zoobot_clumps[4][2].bn2.bias
    model.backbone.body.layer1[2].bn2.running_mean = zoobot_clumps[4][2].bn2.running_mean
    model.backbone.body.layer1[2].bn2.running_var = zoobot_clumps[4][2].bn2.running_var
    model.backbone.body.layer1[2].conv3.weight = zoobot_clumps[4][2].conv3.weight
    model.backbone.body.layer1[2].bn3.weight = zoobot_clumps[4][2].bn3.weight
    model.backbone.body.layer1[2].bn3.bias = zoobot_clumps[4][2].bn3.bias
    model.backbone.body.layer1[2].bn3.running_mean = zoobot_clumps[4][2].bn3.running_mean
    model.backbone.body.layer1[2].bn3.running_var = zoobot_clumps[4][2].bn3.running_var
    model.backbone.body.layer2[0].conv1.weight = zoobot_clumps[5][0].conv1.weight
    model.backbone.body.layer2[0].bn1.weight = zoobot_clumps[5][0].bn1.weight
    model.backbone.body.layer2[0].bn1.bias = zoobot_clumps[5][0].bn1.bias
    model.backbone.body.layer2[0].bn1.running_mean = zoobot_clumps[5][0].bn1.running_mean
    model.backbone.body.layer2[0].bn1.running_var = zoobot_clumps[5][0].bn1.running_var
    model.backbone.body.layer2[0].conv2.weight = zoobot_clumps[5][0].conv2.weight
    model.backbone.body.layer2[0].bn2.weight = zoobot_clumps[5][0].bn2.weight
    model.backbone.body.layer2[0].bn2.bias = zoobot_clumps[5][0].bn2.bias
    model.backbone.body.layer2[0].bn2.running_mean = zoobot_clumps[5][0].bn2.running_mean
    model.backbone.body.layer2[0].bn2.running_var = zoobot_clumps[5][0].bn2.running_var
    model.backbone.body.layer2[0].conv3.weight = zoobot_clumps[5][0].conv3.weight
    model.backbone.body.layer2[0].bn3.weight = zoobot_clumps[5][0].bn3.weight
    model.backbone.body.layer2[0].bn3.bias = zoobot_clumps[5][0].bn3.bias
    model.backbone.body.layer2[0].bn3.running_mean = zoobot_clumps[5][0].bn3.running_mean
    model.backbone.body.layer2[0].bn3.running_var = zoobot_clumps[5][0].bn3.running_var
    model.backbone.body.layer2[0].downsample[0].weight = zoobot_clumps[5][0].downsample[0].weight
    model.backbone.body.layer2[0].downsample[1].weight = zoobot_clumps[5][0].downsample[1].weight
    model.backbone.body.layer2[0].downsample[1].bias = zoobot_clumps[5][0].downsample[1].bias
    model.backbone.body.layer2[0].downsample[1].running_mean = zoobot_clumps[5][0].downsample[1].running_mean
    model.backbone.body.layer2[0].downsample[1].running_var = zoobot_clumps[5][0].downsample[1].running_var
    model.backbone.body.layer2[1].conv1.weight = zoobot_clumps[5][1].conv1.weight
    model.backbone.body.layer2[1].bn1.weight = zoobot_clumps[5][1].bn1.weight
    model.backbone.body.layer2[1].bn1.bias = zoobot_clumps[5][1].bn1.bias
    model.backbone.body.layer2[1].bn1.running_mean = zoobot_clumps[5][1].bn1.running_mean
    model.backbone.body.layer2[1].bn1.running_var = zoobot_clumps[5][1].bn1.running_var
    model.backbone.body.layer2[1].conv2.weight = zoobot_clumps[5][1].conv2.weight
    model.backbone.body.layer2[1].bn2.weight = zoobot_clumps[5][1].bn2.weight
    model.backbone.body.layer2[1].bn2.bias = zoobot_clumps[5][1].bn2.bias
    model.backbone.body.layer2[1].bn2.running_mean = zoobot_clumps[5][1].bn2.running_mean
    model.backbone.body.layer2[1].bn2.running_var = zoobot_clumps[5][1].bn2.running_var
    model.backbone.body.layer2[1].conv3.weight = zoobot_clumps[5][1].conv3.weight
    model.backbone.body.layer2[1].bn3.weight = zoobot_clumps[5][1].bn3.weight
    model.backbone.body.layer2[1].bn3.bias = zoobot_clumps[5][1].bn3.bias
    model.backbone.body.layer2[1].bn3.running_mean = zoobot_clumps[5][1].bn3.running_mean
    model.backbone.body.layer2[1].bn3.running_var = zoobot_clumps[5][1].bn3.running_var
    model.backbone.body.layer2[2].conv1.weight = zoobot_clumps[5][2].conv1.weight
    model.backbone.body.layer2[2].bn1.weight = zoobot_clumps[5][2].bn1.weight
    model.backbone.body.layer2[2].bn1.bias = zoobot_clumps[5][2].bn1.bias
    model.backbone.body.layer2[2].bn1.running_mean = zoobot_clumps[5][2].bn1.running_mean
    model.backbone.body.layer2[2].bn1.running_var = zoobot_clumps[5][2].bn1.running_var
    model.backbone.body.layer2[2].conv2.weight = zoobot_clumps[5][2].conv2.weight
    model.backbone.body.layer2[2].bn2.weight = zoobot_clumps[5][2].bn2.weight
    model.backbone.body.layer2[2].bn2.bias = zoobot_clumps[5][2].bn2.bias
    model.backbone.body.layer2[2].bn2.running_mean = zoobot_clumps[5][2].bn2.running_mean
    model.backbone.body.layer2[2].bn2.running_var = zoobot_clumps[5][2].bn2.running_var
    model.backbone.body.layer2[2].conv3.weight = zoobot_clumps[5][2].conv3.weight
    model.backbone.body.layer2[2].bn3.weight = zoobot_clumps[5][2].bn3.weight
    model.backbone.body.layer2[2].bn3.bias = zoobot_clumps[5][2].bn3.bias
    model.backbone.body.layer2[2].bn3.running_mean = zoobot_clumps[5][2].bn3.running_mean
    model.backbone.body.layer2[2].bn3.running_var = zoobot_clumps[5][2].bn3.running_var
    model.backbone.body.layer2[3].conv1.weight = zoobot_clumps[5][3].conv1.weight
    model.backbone.body.layer2[3].bn1.weight = zoobot_clumps[5][3].bn1.weight
    model.backbone.body.layer2[3].bn1.bias = zoobot_clumps[5][3].bn1.bias
    model.backbone.body.layer2[3].bn1.running_mean = zoobot_clumps[5][3].bn1.running_mean
    model.backbone.body.layer2[3].bn1.running_var = zoobot_clumps[5][3].bn1.running_var
    model.backbone.body.layer2[3].conv2.weight = zoobot_clumps[5][3].conv2.weight
    model.backbone.body.layer2[3].bn2.weight = zoobot_clumps[5][3].bn2.weight
    model.backbone.body.layer2[3].bn2.bias = zoobot_clumps[5][3].bn2.bias
    model.backbone.body.layer2[3].bn2.running_mean = zoobot_clumps[5][3].bn2.running_mean
    model.backbone.body.layer2[3].bn2.running_var = zoobot_clumps[5][3].bn2.running_var
    model.backbone.body.layer2[3].conv3.weight = zoobot_clumps[5][3].conv3.weight
    model.backbone.body.layer2[3].bn3.weight = zoobot_clumps[5][3].bn3.weight
    model.backbone.body.layer2[3].bn3.bias = zoobot_clumps[5][3].bn3.bias
    model.backbone.body.layer2[3].bn3.running_mean = zoobot_clumps[5][3].bn3.running_mean
    model.backbone.body.layer2[3].bn3.running_var = zoobot_clumps[5][3].bn3.running_var
    model.backbone.body.layer3[0].conv1.weight = zoobot_clumps[6][0].conv1.weight
    model.backbone.body.layer3[0].bn1.weight = zoobot_clumps[6][0].bn1.weight
    model.backbone.body.layer3[0].bn1.bias = zoobot_clumps[6][0].bn1.bias
    model.backbone.body.layer3[0].bn1.running_mean = zoobot_clumps[6][0].bn1.running_mean
    model.backbone.body.layer3[0].bn1.running_var = zoobot_clumps[6][0].bn1.running_var
    model.backbone.body.layer3[0].conv2.weight = zoobot_clumps[6][0].conv2.weight
    model.backbone.body.layer3[0].bn2.weight = zoobot_clumps[6][0].bn2.weight
    model.backbone.body.layer3[0].bn2.bias = zoobot_clumps[6][0].bn2.bias
    model.backbone.body.layer3[0].bn2.running_mean = zoobot_clumps[6][0].bn2.running_mean
    model.backbone.body.layer3[0].bn2.running_var = zoobot_clumps[6][0].bn2.running_var
    model.backbone.body.layer3[0].conv3.weight = zoobot_clumps[6][0].conv3.weight
    model.backbone.body.layer3[0].bn3.weight = zoobot_clumps[6][0].bn3.weight
    model.backbone.body.layer3[0].bn3.bias = zoobot_clumps[6][0].bn3.bias
    model.backbone.body.layer3[0].bn3.running_mean = zoobot_clumps[6][0].bn3.running_mean
    model.backbone.body.layer3[0].bn3.running_var = zoobot_clumps[6][0].bn3.running_var
    model.backbone.body.layer3[0].downsample[0].weight = zoobot_clumps[6][0].downsample[0].weight
    model.backbone.body.layer3[0].downsample[1].weight = zoobot_clumps[6][0].downsample[1].weight
    model.backbone.body.layer3[0].downsample[1].bias = zoobot_clumps[6][0].downsample[1].bias
    model.backbone.body.layer3[0].downsample[1].running_mean = zoobot_clumps[6][0].downsample[1].running_mean
    model.backbone.body.layer3[0].downsample[1].running_var = zoobot_clumps[6][0].downsample[1].running_var
    model.backbone.body.layer3[1].conv1.weight = zoobot_clumps[6][1].conv1.weight
    model.backbone.body.layer3[1].bn1.weight = zoobot_clumps[6][1].bn1.weight
    model.backbone.body.layer3[1].bn1.bias = zoobot_clumps[6][1].bn1.bias
    model.backbone.body.layer3[1].bn1.running_mean = zoobot_clumps[6][1].bn1.running_mean
    model.backbone.body.layer3[1].bn1.running_var = zoobot_clumps[6][1].bn1.running_var
    model.backbone.body.layer3[1].conv2.weight = zoobot_clumps[6][1].conv2.weight
    model.backbone.body.layer3[1].bn2.weight = zoobot_clumps[6][1].bn2.weight
    model.backbone.body.layer3[1].bn2.bias = zoobot_clumps[6][1].bn2.bias
    model.backbone.body.layer3[1].bn2.running_mean = zoobot_clumps[6][1].bn2.running_mean
    model.backbone.body.layer3[1].bn2.running_var = zoobot_clumps[6][1].bn2.running_var
    model.backbone.body.layer3[1].conv3.weight = zoobot_clumps[6][1].conv3.weight
    model.backbone.body.layer3[1].bn3.weight = zoobot_clumps[6][1].bn3.weight
    model.backbone.body.layer3[1].bn3.bias = zoobot_clumps[6][1].bn3.bias
    model.backbone.body.layer3[1].bn3.running_mean = zoobot_clumps[6][1].bn3.running_mean
    model.backbone.body.layer3[1].bn3.running_var = zoobot_clumps[6][1].bn3.running_var
    model.backbone.body.layer3[2].conv1.weight = zoobot_clumps[6][2].conv1.weight
    model.backbone.body.layer3[2].bn1.weight = zoobot_clumps[6][2].bn1.weight
    model.backbone.body.layer3[2].bn1.bias = zoobot_clumps[6][2].bn1.bias
    model.backbone.body.layer3[2].bn1.running_mean = zoobot_clumps[6][2].bn1.running_mean
    model.backbone.body.layer3[2].bn1.running_var = zoobot_clumps[6][2].bn1.running_var
    model.backbone.body.layer3[2].conv2.weight = zoobot_clumps[6][2].conv2.weight
    model.backbone.body.layer3[2].bn2.weight = zoobot_clumps[6][2].bn2.weight
    model.backbone.body.layer3[2].bn2.bias = zoobot_clumps[6][2].bn2.bias
    model.backbone.body.layer3[2].bn2.running_mean = zoobot_clumps[6][2].bn2.running_mean
    model.backbone.body.layer3[2].bn2.running_var = zoobot_clumps[6][2].bn2.running_var
    model.backbone.body.layer3[2].conv3.weight = zoobot_clumps[6][2].conv3.weight
    model.backbone.body.layer3[2].bn3.weight = zoobot_clumps[6][2].bn3.weight
    model.backbone.body.layer3[2].bn3.bias = zoobot_clumps[6][2].bn3.bias
    model.backbone.body.layer3[2].bn3.running_mean = zoobot_clumps[6][2].bn3.running_mean
    model.backbone.body.layer3[2].bn3.running_var = zoobot_clumps[6][2].bn3.running_var
    model.backbone.body.layer3[3].conv1.weight = zoobot_clumps[6][3].conv1.weight
    model.backbone.body.layer3[3].bn1.weight = zoobot_clumps[6][3].bn1.weight
    model.backbone.body.layer3[3].bn1.bias = zoobot_clumps[6][3].bn1.bias
    model.backbone.body.layer3[3].bn1.running_mean = zoobot_clumps[6][3].bn1.running_mean
    model.backbone.body.layer3[3].bn1.running_var = zoobot_clumps[6][3].bn1.running_var
    model.backbone.body.layer3[3].conv2.weight = zoobot_clumps[6][3].conv2.weight
    model.backbone.body.layer3[3].bn2.weight = zoobot_clumps[6][3].bn2.weight
    model.backbone.body.layer3[3].bn2.bias = zoobot_clumps[6][3].bn2.bias
    model.backbone.body.layer3[3].bn2.running_mean = zoobot_clumps[6][3].bn2.running_mean
    model.backbone.body.layer3[3].bn2.running_var = zoobot_clumps[6][3].bn2.running_var
    model.backbone.body.layer3[3].conv3.weight = zoobot_clumps[6][3].conv3.weight
    model.backbone.body.layer3[3].bn3.weight = zoobot_clumps[6][3].bn3.weight
    model.backbone.body.layer3[3].bn3.bias = zoobot_clumps[6][3].bn3.bias
    model.backbone.body.layer3[3].bn3.running_mean = zoobot_clumps[6][3].bn3.running_mean
    model.backbone.body.layer3[3].bn3.running_var = zoobot_clumps[6][3].bn3.running_var
    model.backbone.body.layer3[4].conv1.weight = zoobot_clumps[6][4].conv1.weight
    model.backbone.body.layer3[4].bn1.weight = zoobot_clumps[6][4].bn1.weight
    model.backbone.body.layer3[4].bn1.bias = zoobot_clumps[6][4].bn1.bias
    model.backbone.body.layer3[4].bn1.running_mean = zoobot_clumps[6][4].bn1.running_mean
    model.backbone.body.layer3[4].bn1.running_var = zoobot_clumps[6][4].bn1.running_var
    model.backbone.body.layer3[4].conv2.weight = zoobot_clumps[6][4].conv2.weight
    model.backbone.body.layer3[4].bn2.weight = zoobot_clumps[6][4].bn2.weight
    model.backbone.body.layer3[4].bn2.bias = zoobot_clumps[6][4].bn2.bias
    model.backbone.body.layer3[4].bn2.running_mean = zoobot_clumps[6][4].bn2.running_mean
    model.backbone.body.layer3[4].bn2.running_var = zoobot_clumps[6][4].bn2.running_var
    model.backbone.body.layer3[4].conv3.weight = zoobot_clumps[6][4].conv3.weight
    model.backbone.body.layer3[4].bn3.weight = zoobot_clumps[6][4].bn3.weight
    model.backbone.body.layer3[4].bn3.bias = zoobot_clumps[6][4].bn3.bias
    model.backbone.body.layer3[4].bn3.running_mean = zoobot_clumps[6][4].bn3.running_mean
    model.backbone.body.layer3[4].bn3.running_var = zoobot_clumps[6][4].bn3.running_var
    model.backbone.body.layer3[5].conv1.weight = zoobot_clumps[6][5].conv1.weight
    model.backbone.body.layer3[5].bn1.weight = zoobot_clumps[6][5].bn1.weight
    model.backbone.body.layer3[5].bn1.bias = zoobot_clumps[6][5].bn1.bias
    model.backbone.body.layer3[5].bn1.running_mean = zoobot_clumps[6][5].bn1.running_mean
    model.backbone.body.layer3[5].bn1.running_var = zoobot_clumps[6][5].bn1.running_var
    model.backbone.body.layer3[5].conv2.weight = zoobot_clumps[6][5].conv2.weight
    model.backbone.body.layer3[5].bn2.weight = zoobot_clumps[6][5].bn2.weight
    model.backbone.body.layer3[5].bn2.bias = zoobot_clumps[6][5].bn2.bias
    model.backbone.body.layer3[5].bn2.running_mean = zoobot_clumps[6][5].bn2.running_mean
    model.backbone.body.layer3[5].bn2.running_var = zoobot_clumps[6][5].bn2.running_var
    model.backbone.body.layer3[5].conv3.weight = zoobot_clumps[6][5].conv3.weight
    model.backbone.body.layer3[5].bn3.weight = zoobot_clumps[6][5].bn3.weight
    model.backbone.body.layer3[5].bn3.bias = zoobot_clumps[6][5].bn3.bias
    model.backbone.body.layer3[5].bn3.running_mean = zoobot_clumps[6][5].bn3.running_mean
    model.backbone.body.layer3[5].bn3.running_var = zoobot_clumps[6][5].bn3.running_var
    model.backbone.body.layer4[0].conv1.weight = zoobot_clumps[7][0].conv1.weight
    model.backbone.body.layer4[0].bn1.weight = zoobot_clumps[7][0].bn1.weight
    model.backbone.body.layer4[0].bn1.bias = zoobot_clumps[7][0].bn1.bias
    model.backbone.body.layer4[0].bn1.running_mean = zoobot_clumps[7][0].bn1.running_mean
    model.backbone.body.layer4[0].bn1.running_var = zoobot_clumps[7][0].bn1.running_var
    model.backbone.body.layer4[0].conv2.weight = zoobot_clumps[7][0].conv2.weight
    model.backbone.body.layer4[0].bn2.weight = zoobot_clumps[7][0].bn2.weight
    model.backbone.body.layer4[0].bn2.bias = zoobot_clumps[7][0].bn2.bias
    model.backbone.body.layer4[0].bn2.running_mean = zoobot_clumps[7][0].bn2.running_mean
    model.backbone.body.layer4[0].bn2.running_var = zoobot_clumps[7][0].bn2.running_var
    model.backbone.body.layer4[0].conv3.weight = zoobot_clumps[7][0].conv3.weight
    model.backbone.body.layer4[0].bn3.weight = zoobot_clumps[7][0].bn3.weight
    model.backbone.body.layer4[0].bn3.bias = zoobot_clumps[7][0].bn3.bias
    model.backbone.body.layer4[0].bn3.running_mean = zoobot_clumps[7][0].bn3.running_mean
    model.backbone.body.layer4[0].bn3.running_var = zoobot_clumps[7][0].bn3.running_var
    model.backbone.body.layer4[0].downsample[0].weight = zoobot_clumps[7][0].downsample[0].weight
    model.backbone.body.layer4[0].downsample[1].weight = zoobot_clumps[7][0].downsample[1].weight
    model.backbone.body.layer4[0].downsample[1].bias = zoobot_clumps[7][0].downsample[1].bias
    model.backbone.body.layer4[0].downsample[1].running_mean = zoobot_clumps[7][0].downsample[1].running_mean
    model.backbone.body.layer4[0].downsample[1].running_var = zoobot_clumps[7][0].downsample[1].running_var
    model.backbone.body.layer4[1].conv1.weight = zoobot_clumps[7][1].conv1.weight
    model.backbone.body.layer4[1].bn1.weight = zoobot_clumps[7][1].bn1.weight
    model.backbone.body.layer4[1].bn1.bias = zoobot_clumps[7][1].bn1.bias
    model.backbone.body.layer4[1].bn1.running_mean = zoobot_clumps[7][1].bn1.running_mean
    model.backbone.body.layer4[1].bn1.running_var = zoobot_clumps[7][1].bn1.running_var
    model.backbone.body.layer4[1].conv2.weight = zoobot_clumps[7][1].conv2.weight
    model.backbone.body.layer4[1].bn2.weight = zoobot_clumps[7][1].bn2.weight
    model.backbone.body.layer4[1].bn2.bias = zoobot_clumps[7][1].bn2.bias
    model.backbone.body.layer4[1].bn2.running_mean = zoobot_clumps[7][1].bn2.running_mean
    model.backbone.body.layer4[1].bn2.running_var = zoobot_clumps[7][1].bn2.running_var
    model.backbone.body.layer4[1].conv3.weight = zoobot_clumps[7][1].conv3.weight
    model.backbone.body.layer4[1].bn3.weight = zoobot_clumps[7][1].bn3.weight
    model.backbone.body.layer4[1].bn3.bias = zoobot_clumps[7][1].bn3.bias
    model.backbone.body.layer4[1].bn3.running_mean = zoobot_clumps[7][1].bn3.running_mean
    model.backbone.body.layer4[1].bn3.running_var = zoobot_clumps[7][1].bn3.running_var
    model.backbone.body.layer4[2].conv1.weight = zoobot_clumps[7][2].conv1.weight
    model.backbone.body.layer4[2].bn1.weight = zoobot_clumps[7][2].bn1.weight
    model.backbone.body.layer4[2].bn1.bias = zoobot_clumps[7][2].bn1.bias
    model.backbone.body.layer4[2].bn1.running_mean = zoobot_clumps[7][2].bn1.running_mean
    model.backbone.body.layer4[2].bn1.running_var = zoobot_clumps[7][2].bn1.running_var
    model.backbone.body.layer4[2].conv2.weight = zoobot_clumps[7][2].conv2.weight
    model.backbone.body.layer4[2].bn2.weight = zoobot_clumps[7][2].bn2.weight
    model.backbone.body.layer4[2].bn2.bias = zoobot_clumps[7][2].bn2.bias
    model.backbone.body.layer4[2].bn2.running_mean = zoobot_clumps[7][2].bn2.running_mean
    model.backbone.body.layer4[2].bn2.running_var = zoobot_clumps[7][2].bn2.running_var
    model.backbone.body.layer4[2].conv3.weight = zoobot_clumps[7][2].conv3.weight
    model.backbone.body.layer4[2].bn3.weight = zoobot_clumps[7][2].bn3.weight
    model.backbone.body.layer4[2].bn3.bias = zoobot_clumps[7][2].bn3.bias
    model.backbone.body.layer4[2].bn3.running_mean = zoobot_clumps[7][2].bn3.running_mean
    model.backbone.body.layer4[2].bn3.running_var = zoobot_clumps[7][2].bn3.running_var

    # make sure, backbone layers are freezed after copying the weights
    for name, parameter in model.named_parameters():
        if name.startswith('backbone.body.'):
            parameter.requires_grad = False
    
    # unfreeze selected layers
    layers_to_train = ['backbone.body.layer4', 'backbone.body.layer3', 'backbone.body.layer2', 'backbone.body.layer1', 'backbone.body.conv1'][:trainable_layers]
    
    for layer in layers_to_train:
        for name, parameter in model.named_parameters():
            if name.startswith(layer):
                parameter.requires_grad_(True)

    return model