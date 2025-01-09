import os.path

from data import dataset as dataset_lib
from data import decode_utils
import utils
from data import tf_record

import tensorflow as tf

from tasks import task_utils

@dataset_lib.DatasetRegistry.register('ipsc_static_video_detection')
class IPSCStaticVideoDetectionTFRecordDataset(tf_record.TFRecordDataset):

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
        video_feature_map = decode_utils.get_feature_map_for_video(self.config.length)
        detection_feature_map = decode_utils.get_feature_map_for_video_detection(self.config.length)
        return {**video_feature_map, **detection_feature_map}

    def extract(self, example, training):
        h, w = int(example['video/height']), int(example['video/width'])
        # h, w = self.task_config.image_size

        num_frames = int(example['video/num_frames'])

        filenames = example['video/file_names']

        x = filenames[0]
        frame = tf.io.decode_image(tf.io.read_file(x), channels=3)
        length = self.config.length

        frame.set_shape([None, None, 3])

        frame = tf.image.convert_image_dtype(frame, tf.float32)

        target_size = self.config.target_size
        if target_size is not None:
            frame = tf.image.resize(
                frame, target_size, method='bilinear',
                antialias=False, preserve_aspect_ratio=False)
        # area = bbox = None

        area = decode_utils.decode_video_areas(example, self.config.length)
        bbox = decode_utils.decode_video_boxes(example, self.config.length)

        scale = 1. / utils.tf_float32((h, w))
        bbox = utils.scale_points(bbox, scale)

        resized_vid_size = tf.shape(frame)[:2]

        new_example = {
            'orig_video_size': [h, w],
            'video/resized': resized_vid_size,
            'video/file_names': tf.cast(example['video/file_names'], tf.string),
            'video/file_ids': tf.cast(example['video/file_ids'], tf.int64),
            'video/id': tf.cast(example['video/source_id'], tf.int64),
            'video/frame': frame,
            'video/num_frames': tf.cast(num_frames, tf.int32),
            'video/size': tf.cast(example['video/size'], tf.int64),
            'shape': utils.tf_float32((h, w)),
            'class_name': tf.cast(example['video/object/class/text'], dtype=tf.string),
            'class_id': example['video/object/class/label'],
            'bbox': bbox,
            'area': area,
            'is_crowd': tf.cast(example['video/object/is_crowd'], dtype=tf.bool),
        }

        return new_example


@dataset_lib.DatasetRegistry.register('ipsc_static_video_segmentation')
class IPSCStaticVideoSegmentationTFRecordDataset(tf_record.TFRecordDataset):
    def __init__(self, config):
        super().__init__(config)
        self.vid_id_to_rle = None
        self.vid_id_to_rle_len = None

    def load_dataset(self, input_context, training):

        if self.config.rle_from_json:
            json_dict = task_utils.load_json(self.config.category_names_path)

            keys = [video['id'] for video in json_dict['videos']]
            keys_tensor = tf.constant(keys, dtype=tf.int64)

            rles = [' '.join(map(str, video['rle'])) for video in json_dict['videos']]
            rles_tensor = tf.constant(rles, dtype=tf.string)
            # rles = [video['rle'] for video in json_dict['videos']]
            # rles_tensor = tf.constant(rles)

            rle_lens = [video['rle_len'] for video in json_dict['videos']]
            rle_lens_tensor = tf.constant(rle_lens, dtype=tf.int64)

            init_rle = tf.lookup.KeyValueTensorInitializer(
                keys_tensor, rles_tensor)
            self.vid_id_to_rle = tf.lookup.StaticHashTable(init_rle, default_value='')

            init_rle_len = tf.lookup.KeyValueTensorInitializer(
                keys_tensor, rle_lens_tensor)
            self.vid_id_to_rle_len = tf.lookup.StaticHashTable(init_rle_len, default_value=-1)

        return super().load_dataset(input_context, training)

    def get_feature_map(self, training):
        feat_dict = {
            'video/id': tf.io.FixedLenFeature((), tf.int64, -1),
            'video/height': tf.io.FixedLenFeature((), tf.int64, -1),
            'video/width': tf.io.FixedLenFeature((), tf.int64, -1),
            'video/num_frames': tf.io.FixedLenFeature((), tf.int64, -1),
            'video/path': tf.io.FixedLenFeature((), tf.string),
            'video/mask_path': tf.io.FixedLenFeature((), tf.string),
            'video/seq': tf.io.FixedLenFeature((), tf.string),
            'video/n_runs': tf.io.FixedLenFeature((), tf.int64, -1),
        }
        if not self.config.rle_from_json:
            feat_dict.update({
                'video/rle_len': tf.io.FixedLenFeature((), tf.int64, -1),
                'video/rle': tf.io.VarLenFeature(tf.int64),
            }
            )

        for _id in range(self.config.length):
            frame_feat_dict = {
                f'video/frame-{_id}/filename': tf.io.FixedLenFeature((), tf.string),
                f'video/frame-{_id}/image_id': tf.io.FixedLenFeature((), tf.string),
                f'video/frame-{_id}/frame_id': tf.io.FixedLenFeature((), tf.int64, -1),
                f'video/frame-{_id}/key/sha256': tf.io.FixedLenFeature((), tf.string),
                f'video/frame-{_id}/encoded': tf.io.FixedLenFeature((), tf.string),
                f'video/frame-{_id}/format': tf.io.FixedLenFeature((), tf.string),
            }
            feat_dict.update(frame_feat_dict)
        return feat_dict

    def parse_example(self, example, training):
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

    def filter_example(self, example, training):
        """
        probabilistically filter out examples with no foreground
        """
        if training or self.config.eval.filter:
            if example['video/n_runs'] > 0:
                return True
            rand_num = tf.random.uniform(shape=[1])
            if rand_num[0] < self.config.empty_seg_prob:
                return True
            return False
        else:
            return True

    def extract(self, example, training):
        h, w = int(example['video/height']), int(example['video/width'])
        filenames = []
        frame_ids = []
        image_ids = []
        frame = decode_utils.decode_image(
            example, f'video/frame-0/encoded')

        for _id in range(self.config.length):
            filename = example[f'video/frame-{_id}/filename']
            image_id = example[f'video/frame-{_id}/image_id']
            frame_id = example[f'video/frame-{_id}/frame_id']
            filenames.append(filename)
            frame_ids.append(frame_id)
            image_ids.append(image_id)

        vid_id = example['video/id']
        n_runs = example['video/n_runs']

        # tf.debugging.assert_greater_equal(n_runs, tf.cast(0, tf.int64), "n_runs must be >= 0")
        # assert n_runs >=0, "n_runs must be >= 0"

        tf.debugging.assert_greater_equal(vid_id, tf.cast(0, tf.int64), "vid_id must be >= 0")

        if self.config.rle_from_json:
            rle_str = self.vid_id_to_rle.lookup(vid_id)
            rle = tf.cond(
                tf.strings.length(rle_str) == 0,
                lambda: tf.convert_to_tensor([], dtype=tf.int64),
                lambda: tf.strings.to_number(tf.strings.split(rle_str, sep=' '),
                                             out_type=tf.int64)
            )
            rle_len = self.vid_id_to_rle_len.lookup(vid_id)
            tf.debugging.assert_greater_equal(rle_len, tf.cast(0, tf.int64), "rle_len must be > 0")
        else:
            rle = example['video/rle']
            rle_len = example['video/rle_len']
        vid_path = example['video/path']
        mask_vid_path = example['video/mask_path']
        seq = example['video/seq']

        orig_image_size = tf.shape(frame)[:2]

        new_example = {
            'video': frame,
            'vid_id': vid_id,
            'n_runs': n_runs,
            'rle': rle,
            'rle_len': rle_len,
            'vid_size': (h, w),
            'orig_image_size': orig_image_size,
            'filenames': filenames,
            'image_ids': image_ids,
            'frame_ids': frame_ids,
            'vid_path': vid_path,
            'mask_vid_path': mask_vid_path,
            'seq': seq,
        }
        return new_example
