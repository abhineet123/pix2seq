from data import dataset as dataset_lib
from data import decode_utils
import utils


import tensorflow as tf


@dataset_lib.DatasetRegistry.register('ipsc_video_detection')
class IPSCVideoDetectionTFRecordDataset(dataset_lib.TFRecordDataset):
    def get_feature_map(self, config):
        video_feature_map = decode_utils.get_feature_map_for_video()
        detection_feature_map = decode_utils.get_feature_map_for_video_detection(config.vid_len)
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
        frames = tf.map_fn(lambda x: tf.io.decode_jpeg(x, channels=3),
                           example['video/filenames'], tf.uint8)
        frames.set_shape([num_frames, h, w, 3])

        new_example = {
            'video/frames': tf.image.convert_image_dtype(frames, tf.float32),
            'video/num_frames': tf.cast(num_frames, tf.int32),
            'video/id': tf.cast(example['video/source_id'], tf.int32),
        }

        bbox = decode_utils.decode_video_boxes(example)
        scale = 1. / utils.tf_float32((h, w))
        bbox = utils.scale_points(bbox, scale)

        new_example.update({
            'shape': utils.tf_float32((h, w)),
            'label': example['video/object/class/label'],
            'bbox': bbox,
            'area': decode_utils.decode_video_areas(example),
            'is_crowd': tf.cast(example['video/object/is_crowd'], dtype=tf.bool),
        })

        return new_example