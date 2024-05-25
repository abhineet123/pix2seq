import os.path

from data import dataset as dataset_lib
from data import decode_utils
import utils
from data import tf_record

import tensorflow as tf


@dataset_lib.DatasetRegistry.register('ipsc_video_detection')
class IPSCVideoDetectionTFRecordDataset(tf_record.TFRecordDataset):

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

        # print(f'h, w : {h, w}')
        # print(f'num_frames: {num_frames}')
        # print(f'filenames: {filenames}')
        # exit()

        # def read_video_frames(x):
        #     print(f'x: {x}')
        #     tf.print(f'x: {x}')

        # root_dir = tf.convert_to_tensor(self.config.root_dir)
        # file_path = tf.strings.join([root_dir, x], os.path.sep)
        # print(f'file_path: {file_path}')
        # tf.print(f'file_path: {file_path}')

        # return tf.io.decode_image(tf.io.read_file(x), channels=3)
        # exit()

        decode_fn = tf.io.decode_image
        # decode_fn = tf.io.decode_jpeg
        frames = tf.map_fn(
            lambda x: decode_fn(tf.io.read_file(x), channels=3),
            # read_video_frames,
            filenames,
            fn_output_signature=tf.uint8
        )
        length = self.config.length

        # frames.set_shape([length, h, w, 3])
        frames.set_shape([length, None, None, 3])
        # frames.set_shape([None, None, None, 3])

        frames = tf.image.convert_image_dtype(frames, tf.float32)

        target_size = self.config.target_size
        if target_size is not None:
            frames = tf.image.resize(
                frames, target_size, method='bilinear',
                antialias=False, preserve_aspect_ratio=False)
        # area = bbox = None

        area = decode_utils.decode_video_areas(example, self.config.length)
        bbox = decode_utils.decode_video_boxes(example, self.config.length)

        scale = 1. / utils.tf_float32((h, w))
        bbox = utils.scale_points(bbox, scale)

        resized_vid_size = tf.shape(frames)[1:3]

        new_example = {
            'orig_video_size': [h, w],
            'video/resized': resized_vid_size,
            'video/file_names': tf.cast(example['video/file_names'], tf.string),
            'video/file_ids': tf.cast(example['video/file_ids'], tf.int64),
            'video/id': tf.cast(example['video/source_id'], tf.int64),
            'video/frames': frames,
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


@dataset_lib.DatasetRegistry.register('ipsc_video_segmentation')
class IPSCVideoSegmentationTFRecordDataset(tf_record.TFRecordDataset):

    def filter_example(self, example, training):
        """
        probabilistically filter out examples with no foreground
        """
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
        vid_seg_feature_map = decode_utils.get_feature_map_for_video_segmentation(
            self.config.length)
        return vid_seg_feature_map

    def extract(self, example, training):
        h, w = int(example['video/height']), int(example['video/width'])
        num_frames = int(example['video/num_frames'])

        assert num_frames == self.config.length, "num_frames mismatch"

        # frame_ids = example['video/frame_ids']
        # image_ids = example['video/image_ids']

        images = []
        filenames = []
        frame_ids = []
        image_ids = []
        for _id in range(num_frames):
            image = decode_utils.decode_image(
                example, f'video/frame-{_id}/encoded')
            images.append(image)

            filename = example[f'video/frame-{_id}/filename']
            image_id = example[f'video/frame-{_id}/image_id']
            frame_id = example[f'video/frame-{_id}/frame_id']
            filenames.append(filename)
            frame_ids.append(frame_id)
            image_ids.append(image_id)

        images = tf.stack(images, axis=0)

        rle = example['video/rle']
        vid_path = example['video/path']
        mask_vid_path = example['video/mask_vid_path']
        seq = example['video/seq']

        orig_image_size = tf.shape(images)[1:3]

        new_example = {
            'images': images,
            'rle': rle,
            'orig_image_size': orig_image_size,
            'filenames': filenames,
            'image_ids': image_ids,
            'frame_ids': frame_ids,
            'vid_path': vid_path,
            'mask_vid_path': mask_vid_path,
            'seq': seq,
        }
        return new_example
