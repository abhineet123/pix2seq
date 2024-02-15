import copy

from data import data_utils
from data.transforms import TransformRegistry, Transform

import tensorflow as tf

@TransformRegistry.register('record_original_video_size')
class RecordOriginalVideoSize(Transform):
    def process_example(self, example: dict[str, tf.Tensor]):
        # video_key = self.config.get('video_key', 'video')
        # orig_video_size_key = self.config.get('original_video_size_key',
        #                                       DEFAULT_ORIG_VIDEO_SIZE_KEY)
        # example[orig_video_size_key] = tf.shape(example[video_key])[1:3]
        return example


@TransformRegistry.register('fixed_size_crop_video')
class FixedSizeVideoCrop(Transform):
    def process_example(self, example: dict[str, tf.Tensor]):
        example_out = copy.copy(example)
        target_height, target_width = self.config.target_size

        # input_size_key = self.config.input_size_key
        # input_size = example_out[input_size_key]

        input_size = tf.shape(example[self.config.inputs[0]])[1:3]

        output_size = tf.stack([target_height, target_width])

        max_offset = tf.subtract(input_size, output_size)
        max_offset = tf.cast(tf.maximum(max_offset, 0), tf.float32)
        offset = tf.multiply(max_offset, tf.random.uniform([], 0.0, 1.0))
        offset = tf.cast(offset, tf.int32)

        region = (offset[0], offset[1],
                  tf.minimum(output_size[0], input_size[0] - offset[0]),
                  tf.minimum(output_size[1], input_size[1] - offset[1]))
        object_coordinate_keys = self.config.get('object_coordinate_keys', [])

        example_out = data_utils.video_crop(example_out, region, self.config.inputs,
                                            object_coordinate_keys)

        example_out['fixed_size_crop_video'] = output_size

        return example_out


@TransformRegistry.register('resize_video')
class ResizeVideo(Transform):
    def process_example(self, example: dict[str, tf.Tensor]):
        example = copy.copy(example)
        num_inputs = len(self.config.inputs)
        resize_methods = self.config.get('resize_method', ['bilinear'] * num_inputs)
        antialias_list = self.config.get('antialias', [False] * num_inputs)
        preserve_ar = self.config.get('preserve_aspect_ratio', [True] * num_inputs)

        for k, resize_method, antialias, p_ar in zip(
                self.config.inputs, resize_methods, antialias_list, preserve_ar):
            example[k] = tf.image.resize(
                example[k],
                size=self.config.target_size, method=resize_method,
                antialias=antialias, preserve_aspect_ratio=p_ar)
        return example


@TransformRegistry.register('scale_jitter_video')
class ScaleJitterVideo(Transform):
    def process_example(self, example: dict[str, tf.Tensor]):
        example_out = copy.copy(example)
        target_height, target_width = self.config.target_size

        # input_size_key = self.config.input_size_key
        # input_size = example_out[input_size_key]

        input_size = tf.cast(tf.shape(example_out[self.config.inputs[0]])[1:3], tf.float32)

        min_scale, max_scale = self.config.min_scale, self.config.max_scale
        # k = self.config.inputs[0]
        output_size = tf.constant([target_height, target_width], tf.float32)
        random_scale = tf.random.uniform([], min_scale, max_scale)
        random_scale_size = tf.multiply(output_size, random_scale)
        scale = tf.minimum(
            random_scale_size[0] / input_size[0],
            random_scale_size[1] / input_size[1]
        )
        scaled_size = tf.cast(tf.multiply(input_size, scale), tf.int32)

        num_inputs = len(self.config.inputs)
        resize_methods = self.config.get('resize_method', ['bilinear'] * num_inputs)
        antialias_list = self.config.get('antialias', [False] * num_inputs)
        for k, resize_method, antialias in zip(self.config.inputs,
                                               resize_methods, antialias_list):
            video = example_out[k]
            resized_video = tf.image.resize(
                video, scaled_size,
                method=resize_method, antialias=antialias)
            example_out[k] = resized_video
        example_out['scale_jitter_video'] = scaled_size
        return example_out


@TransformRegistry.register('random_horizontal_flip_video')
class RandomHorizontalFlipVideo(Transform):
    def process_example(self, example: dict[str, tf.Tensor]):
        example = copy.copy(example)
        inputs = {k: example[k] for k in self.config.inputs}
        boxes = {k: example[k] for k in self.config.get('bbox_keys', [])}
        # keypoints = {k: example[k] for k in self.config.get('keypoints_keys', [])}
        # polygons = {k: example[k] for k in self.config.get('polygon_keys', [])}

        with tf.name_scope('RandomHorizontalFlipVideo'):
            coin_flip = tf.random.uniform([]) > 0.5
            if coin_flip:
                inputs = {k: tf.image.flip_left_right(v) for k, v in inputs.items()}
                boxes = {k: data_utils.flip_polygons_left_right(v) for k, v in boxes.items()}
        example.update(inputs)
        example.update(boxes)
        # example.update(keypoints)
        # example.update(polygons)
        return example


@TransformRegistry.register('inject_noise_bbox_video')
class InjectNoiseBboxVideo(Transform):
    def process_example(self, example: dict[str, tf.Tensor]):
        example = copy.copy(example)
        bbox_key = 'bbox'
        class_id_key = 'class_id'
        class_name_key = 'class_name'

        num_instances = tf.shape(example[bbox_key])[0]
        if num_instances < self.config.max_instances_per_image:
            n_noise_bbox = self.config.max_instances_per_image - num_instances
            example[bbox_key], example[class_id_key], example[class_name_key] = data_utils.augment_bbox_video(
                bbox=example[bbox_key],
                class_id=example[class_id_key],
                class_name=example[class_name_key],
                length=self.config.length,
                max_disp=self.config.max_disp,
                max_jitter=0.,
                n_noise_bboxes=n_noise_bbox)
        return example


@TransformRegistry.register('pad_video_to_max_size')
class PadVideoToMaxSize(Transform):
    def process_example(self, example: dict[str, tf.Tensor]):
        example_out = copy.copy(example)
        num_inputs = len(self.config.inputs)
        backgrnd_val = self.config.get('background_val', [0.3] * num_inputs)
        target_size = self.config.target_size

        # input_size_key = self.config.input_size_key
        # unpadded_video_size =example_out[input_size_key]

        unpadded_video_size = tf.shape(example_out[self.config.inputs[0]])[1:3]

        unpadded_video_size = tf.cast(unpadded_video_size, tf.float32)

        for k, backgrnd_val_ in zip(self.config.inputs, backgrnd_val):
            unpadded_video = example_out[k]
            if k == 'video':
                example_out['unpadded_video_size'] = unpadded_video_size
                height = unpadded_video_size[0]
                width = unpadded_video_size[1]

            example_out[k] = backgrnd_val_ + tf.image.pad_to_bounding_box(
                unpadded_video - backgrnd_val_,
                offset_height=0,
                offset_width=0,
                target_height=target_size[0],
                target_width=target_size[1]
            )

        example_out['fixed_size_crop_video'] = target_size

        # Adjust the coordinate fields.
        object_coordinate_keys = self.config.get('object_coordinate_keys', [])
        if object_coordinate_keys:

            assert 'video' in self.config.inputs, \
                "video must be in inputs for object_coordinate_keys to be processed"

            hratio = tf.cast(height, tf.float32) / tf.cast(target_size[0], tf.float32)
            wratio = tf.cast(width, tf.float32) / tf.cast(target_size[1], tf.float32)
            scale = tf.stack([hratio, wratio])
            for key in object_coordinate_keys:
                example_out[key] = data_utils.flatten_points(
                    data_utils.unflatten_points(example_out[key]) * scale)
        return example_out
