from data import dataset as dataset_lib
from data import decode_utils
import utils

import tensorflow as tf


@dataset_lib.DatasetRegistry.register('ipsc_video_detection')
class IPSCVideoDetectionTFRecordDataset(dataset_lib.TFRecordDataset):

    def parse_example(self, example, training):
        """Parse the serialized example into a dictionary of tensors.

        Args:
          example: the serialized tf.train.Example or tf.train.SequenceExample.
          training: `bool` of training vs eval mode.

        Returns:
          a dictionary of feature name to tensors.
        """
        print(f'example: {example}')

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
        video_feature_map = decode_utils.get_feature_map_for_video()
        detection_feature_map = decode_utils.get_feature_map_for_video_detection(self.config.length)
        return {**video_feature_map, **detection_feature_map}

    def extract(self, example, training):
        """Extracts needed features & annotations into a flat dictionary.

        Args:
          example: `dict` of raw features from tfrecord file
          training: `bool` of training vs eval mode.

        Returns:
          example: `dict` of relevant features and labels.
        """
        h, w = int(example['video/height']), int(example['video/width'])
        num_frames = int(example['video/num_frames'])

        filenames = example['video/filenames']
        print(f'h, w : {h, w}')
        print(f'num_frames: {num_frames}')
        print(f'filenames: {filenames}')
        # exit()

        # def read_video_frames(x):
        #     print(f'x: {x}')
        #     exit()

        frames = tf.map_fn(
            lambda x: tf.io.decode_jpeg(x, channels=3),
            # read_video_frames,
            filenames, tf.uint8
        )
        frames.set_shape([None, None, None, 3])

        new_example = {
            'video/frames': tf.image.convert_image_dtype(frames, tf.float32),
            'video/num_frames': tf.cast(num_frames, tf.int32),
            'video/id': tf.cast(example['video/source_id'], tf.int32),
        }

        bbox = decode_utils.decode_video_boxes(example, self.config.length)
        scale = 1. / utils.tf_float32((h, w))
        bbox = utils.scale_points(bbox, scale)

        new_example.update({
            'shape': utils.tf_float32((h, w)),
            'label': example['video/object/class/label'],
            'bbox': bbox,
            'area': decode_utils.decode_video_areas(example, self.config.length),
            'is_crowd': tf.cast(example['video/object/is_crowd'], dtype=tf.bool),
        })

        return new_example
