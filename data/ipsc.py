# coding=utf-8
# Copyright 2022 The Pix2Seq Authors.
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
import utils
import vocab
from data import dataset as dataset_lib
from data import tf_record
from data import decode_utils
import tensorflow as tf


def _xy_to_yx(tensor):
    """Convert a tensor in xy order into yx order.

    Args:
      tensor: of shape [num_instances, points * 2] in the xy order.

    Returns:
      a tensor of shape [num_instances, points, 2] in the yx order.
    """
    max_points = tf.shape(tensor)[1] / 2
    t = tf.reshape(tensor, [-1, max_points, 2])
    t = tf.stack([t[:, :, 1], t[:, :, 0]], axis=2)
    return tf.reshape(t, [-1, max_points * 2])


@dataset_lib.DatasetRegistry.register('ipsc_object_detection')
class IPSCObjectDetectionTFRecordDataset(tf_record.TFRecordDataset):
    """IPSC object detection dataset."""

    def get_feature_map(self, training):
        """Returns feature map for parsing the TFExample."""
        del training
        image_feature_map = decode_utils.get_feature_map_for_image()
        detection_feature_map = decode_utils.get_feature_map_for_object_detection()
        return {**image_feature_map, **detection_feature_map}

    def filter_example(self, example, training):
        # Filter out examples with no instances.
        if training:
            return tf.shape(example['image/object/bbox/xmin'])[0] > 0
        else:
            return True

    def extract(self, example, training):
        """Extracts needed features & annotations into a flat dictionary.

        Note:
          - label starts at 1 instead of 0, as 0 is reserved for special use
            (such as padding).
          - coordinates (e.g. bbox) are (normalized to be) in [0, 1].

        Args:
          example: `dict` of raw features.
          training: `bool` of training vs eval mode.

        Returns:
          example: `dict` of relevant features and labels.
        """

        img_id = example['image/source_id']
        height = example['image/height']
        width = example['image/width']
        image = decode_utils.decode_image(example)

        orig_image_size = tf.shape(image)[:2]
        # orig_image_size2 = [height, width]

        # target_size = self.task_config.image_size
        target_size = self.config.target_size

        if target_size is not None:
            image = tf.image.resize(
                image, target_size, method='bilinear',
                antialias=False, preserve_aspect_ratio=False)

        resized_image_size = tf.shape(image)[:2]

        # tf.print(f'orig_image_size: {orig_image_size}')
        # tf.print(f'orig_image_size2: {orig_image_size2}')
        # tf.print(f'resized_image_size: {resized_image_size}')

        new_example = {
            'image': image,
            'orig_image_size': orig_image_size,
            'image/id': img_id,
            'image/resized': resized_image_size,
        }

        bbox = decode_utils.decode_boxes(example)

        # hratio = tf.cast(target_size[0], tf.float32) / tf.cast(height, tf.float32)
        # wratio = tf.cast(target_size[1], tf.float32) / tf.cast(width, tf.float32)
        # scale = tf.stack([hratio, wratio])
        # scale = 1. / utils.tf_float32(target_size)

        # border_height, border_width = padded_height - height, padded_width - width
        scale = 1. / utils.tf_float32([height, width])
        bbox = utils.scale_points(bbox, scale)

        new_example.update({
            'label': example['image/object/class/label'],
            'bbox': bbox,
            'area': decode_utils.decode_areas(example),
            'is_crowd': decode_utils.decode_is_crowd(example),
        })
        return new_example


@dataset_lib.DatasetRegistry.register('ipsc_semantic_segmentation')
class IPSCSemanticSegmentationTFRecordDataset(tf_record.TFRecordDataset):

    def filter_example(self, example, training):
        # probabilistically filter out examples with no foreground
        if training:
            if tf.shape(example['image/rle'])[0] > 0:
                return True
            rand_num = tf.random.uniform(shape=[1])
            if rand_num[0] < self.config.empty_seg_prob:
                return True
            return False
        else:
            return True

    def parse_example(self, example, training):
        """Parse the serialized example into a dictionary of tensors.

        Args:
          example: the serialized tf.train.Example or tf.train.SequenceExample.
          training: `bool` of training vs eval mode.

        Returns:
          a dictionary of feature name to tensors.
        """
        # print(f'example: {example}')

        # raise AssertionError()

        feature_map = self.get_feature_map(training)
        if isinstance(feature_map, dict):
            example = tf.io.parse_single_example(example, feature_map)
        else:
            raise AssertionError('unsupported type of feature_map')

        for k in example:
            if isinstance(example[k], tf.SparseTensor):
                if example[k].dtype == tf.string:
                    example[k] = tf.sparse.to_dense(example[k], default_value='')
                else:
                    example[k] = tf.sparse.to_dense(example[k], default_value=0)
        return example

    def get_feature_map(self, training):
        img_feature_map = decode_utils.get_feature_map_for_image()
        seg_feature_map = decode_utils.get_feature_map_for_semantic_segmentation()
        return {**img_feature_map, **seg_feature_map}

    def extract(self, example, training):
        img_id = example['image/source_id']
        image = decode_utils.decode_image(example)
        rle = example['image/rle']

        orig_image_size = tf.shape(image)[:2]

        new_example = {
            'image': image,
            'rle': rle,
            'orig_image_size': orig_image_size,
            'image/id': img_id,
        }
        return new_example


@dataset_lib.DatasetRegistry.register('ipsc_instance_segmentation')
class IPSCInstanceSegmentationTFRecordDataset(tf_record.TFRecordDataset):
    """Coco instance segmentation dataset."""

    def get_feature_map(self, unused_training):
        """Returns feature map for parsing the TFExample."""
        image_feature_map = decode_utils.get_feature_map_for_image()
        detection_feature_map = decode_utils.get_feature_map_for_object_detection()
        seg_feature_map = decode_utils.get_feature_map_for_instance_segmentation()
        return {**image_feature_map, **detection_feature_map, **seg_feature_map}

    def filter_example(self, example, training):
        # Filter out examples with no instances, to avoid error when converting
        # RaggedTensor to tensor: `Invalid first partition input. Tensor requires
        # at least one element.`
        return tf.shape(example['image/object/bbox/xmin'])[0] > 0

    def extract(self, example, training):
        """Extracts needed features & annotations into a flat dictionary.

        Note:
          - label starts at 1 instead of 0, as 0 is reserved for special use
            (such as padding).
          - coordinates (e.g. bbox) are (normalized to be) in [0, 1].

        Args:
          example: `dict` of raw features.
          training: `bool` of training vs eval mode.

        Returns:
          example: `dict` of relevant features and labels.
        """
        assert not self.task_config.shuffle_polygon_start_point
        new_example = {
            'image': decode_utils.decode_image(example),
            'image/id': tf.strings.to_number(example['image/source_id'], tf.int64),
        }

        max_points = self.task_config.max_points_per_object
        polygons = example['image/object/segmentation'].to_tensor(
            default_value=vocab.PADDING_FLOAT,
            shape=[None, max_points * 2])
        polygons = _xy_to_yx(polygons)

        bbox = decode_utils.decode_boxes(example)
        labels = example['image/object/class/label']
        iscrowd = decode_utils.decode_is_crowd(example)
        areas = decode_utils.decode_areas(example)
        scores = decode_utils.decode_scores(example)

        # Drop crowded object annotation during both training and eval.
        is_valid = tf.logical_not(iscrowd)
        bbox = tf.boolean_mask(bbox, is_valid)
        iscrowd = tf.boolean_mask(iscrowd, is_valid)
        labels = tf.boolean_mask(labels, is_valid)
        areas = tf.boolean_mask(areas, is_valid)
        polygons = tf.boolean_mask(polygons, is_valid)
        scores = tf.boolean_mask(scores, is_valid)

        scale = 1. / utils.tf_float32(tf.shape(new_example['image'])[:2])

        new_example.update({
            'label': labels,
            'bbox': utils.scale_points(bbox, scale),
            'area': areas,
            'is_crowd': iscrowd,
            'polygon': utils.scale_points(polygons, scale),
            'scores': scores
        })
        return new_example
