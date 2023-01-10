# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Zoobot Faster R-CNN implementation.

See "Deep Residual Learning for Image Recognition" by He et al., 2015.
https://arxiv.org/abs/1512.03385

Note: this implementation assumes that the classification checkpoint used
to finetune this model is trained using the same configuration as that of
the MSRA provided checkpoints
(see https://github.com/KaimingHe/deep-residual-networks), e.g., with
same preprocessing, batch norm scaling, etc.
"""

import tensorflow.compat.v1 as tf

from object_detection.meta_architectures import faster_rcnn_meta_arch
from zoobot.tensorflow.estimators import define_model, efficientnet_standard, efficientnet_custom, custom_layers
from object_detection.utils import model_util

CUT_LAYER = 'block7a_project_conv'


class FasterRCNNZoobotFeatureExtractor(
    faster_rcnn_meta_arch.FasterRCNNKerasFeatureExtractor):
  """Faster R-CNN with Zoobot feature extractor implementation."""

  def __init__(self,
               is_training,
               first_stage_features_stride=16,
               batch_norm_trainable=False,
               weight_decay=0.0):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      weight_decay: See base class.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16.
    """
    if first_stage_features_stride != 16:
      raise ValueError('`first_stage_features_stride` must be 16.')
    super(FasterRCNNZoobotFeatureExtractor, self).__init__(
        is_training, first_stage_features_stride, batch_norm_trainable,
        weight_decay)
    self.classification_backbone = define_model.load_model(
        './pre_trained_models/Zoobot_EfficientnetB0_colour/checkpoint', 
        expect_partial=True,
        include_top=False, 
        input_size=300,  
        crop_size=225,  
        resize_size=224, 
        output_dim=None,
        channels=3 
    ).get_layer('sequential_1').get_layer('efficientnet-b0')
    self._variable_dict = {}


  def preprocess(self, resized_inputs):
    """Faster R-CNN Resnet V1 preprocessing.

    Args:
      resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
        representing a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: A [batch, height_out, width_out, channels] float32
        tensor representing a batch of images.

    """
    if resized_inputs.shape.as_list()[3] == 3:
      channel_means = [123.68, 116.779, 103.939]
      resized_inputs - [[channel_means]]
    
    return (resized_inputs/255.0)


  def get_proposal_feature_extractor_model(self, name=None):
    """Returns a model that extracts first stage RPN features.

    Extracts features using the first part of the headless EfficientNetB0 network

    Args:
      name: A scope name to construct all variables within.

    Returns:
      A Keras model that takes preprocessed_inputs:
        A [batch, height, width, channels] float32 tensor
        representing a batch of images.

      And returns rpn_feature_map:
        A tensor with shape [batch, height, width, depth]
    """
    
    with tf.name_scope(name):
      with tf.name_scope('EffnetB0'):
        proposal_features = self.classification_backbone.get_layer(name=CUT_LAYER).output
        keras_model = tf.keras.Model(
          inputs=self.classification_backbone.inputs,
          outputs=proposal_features)
        for variable in keras_model.variables:
          self._variable_dict[variable.name[:-2]] = variable
          
        return keras_model


  def get_box_classifier_feature_extractor_model(self, name=None):
    """Returns a model that extracts second stage box classifier features.

    This function reconstructs the "second half" of the EfficientNetB0
    network after the part defined in `get_proposal_feature_extractor_model`.

    Args:
      name: A scope name to construct all variables within.

    Returns:
      A Keras model that takes proposal_feature_maps:
        A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
        representing the feature map cropped to each proposal.
      And returns proposal_classifier_features:
        A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, height, width, depth]
        representing box classifier features for each proposal.
    """

    with tf.name_scope(name):
      with tf.name_scope('EffnetB0'):
        proposal_feature_maps = self.classification_backbone.get_layer(name=CUT_LAYER).output
        proposal_classifier_features = self.classification_backbone.get_layer(name='top_activation').output
    
        keras_model = model_util.extract_submodel(
                model=self.classification_backbone,
                inputs=proposal_feature_maps,
                outputs=proposal_classifier_features)
        for variable in keras_model.variables:
          self._variable_dict[variable.name[:-2]] = variable
        
        return keras_model