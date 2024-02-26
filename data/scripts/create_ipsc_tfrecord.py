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
r"""Convert COCO dataset to tfrecords."""

import collections
import json
import os
import sys

sys.path.append(os.getcwd())

# from absl import app
# from absl import flags
# from absl import logging
import numpy as np
import tensorflow as tf
import paramparse

import vocab
from data.scripts import tfrecord_lib


# flags.DEFINE_string('image_dir', './datasets/ipsc/well3/all_frames_roi', 'Directory containing images.')
# # flags.DEFINE_string('ann_file', './datasets/ipsc/well3/all_frames_roi/ext_reorg_roi_g2_0_53.json',
# # 'Instance annotation file.')
# flags.DEFINE_string('ann_file', 'ext_reorg_roi_g2_54_126.json', 'Instance annotation file.')
# # flags.DEFINE_string('pan_ann_file', '', 'Panoptic annotation file.')
# # flags.DEFINE_string('pan_masks_dir', '', 'Directory containing panoptic masks.')
# flags.DEFINE_string('output_dir', './datasets/ipsc/well3/all_frames_roi/tfrecord', 'Output directory')
# flags.DEFINE_integer('num_shards', 32, 'Number of shards for output file.')
# flags.DEFINE_integer('n_proc', 0, 'n_proc')
#
# FLAGS = flags.FLAGS


class Params(paramparse.CFG):

    def __init__(self):
        paramparse.CFG.__init__(self, cfg_prefix='p2s_tfrecord')
        self.ann_file = ''
        self.ann_suffix = ''
        self.image_dir = ''
        self.enable_masks = 1
        self.n_proc = 0
        self.ann_ext = 'json'
        self.num_shards = 32
        self.output_dir = ''
        self.xml_output_file = ''

        self.start_seq_id = 0
        self.end_seq_id = -1


def load_instance_annotations(annotation_path):
    print(f'Reading coco annotations from {annotation_path}')
    if annotation_path.endswith('.json'):
        import json
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
    elif annotation_path.endswith('.json.gz'):
        import compress_json
        annotations = compress_json.load(annotation_path)
    else:
        raise AssertionError(f'Invalid annotation_path: {annotation_path}')

    image_info = annotations['images']
    category_id_to_name_map = dict(
        (element['id'], element['name']) for element in annotations['categories'])

    img_to_ann = collections.defaultdict(list)
    for ann in annotations['annotations']:
        image_id = ann['image_id']
        img_to_ann[image_id].append(ann)

    return image_info, category_id_to_name_map, img_to_ann


def coco_annotations_to_lists(obj_annotations, id_to_name_map):
    data = dict((k, list()) for k in [
        'xmin', 'xmax', 'ymin', 'ymax', 'is_crowd', 'category_id',
        'category_names', 'area'])

    for ann in obj_annotations:
        (x, y, width, height) = tuple(ann['bbox'])
        xmin, xmax, ymin, ymax = float(x), float(x + width), float(y), float(y + height)

        assert xmax > xmin and ymax > ymin, f"invalid bbox: {[xmin, xmax, ymin, ymax]}"

        data['xmin'].append(xmin)
        data['xmax'].append(xmax)
        data['ymin'].append(ymin)
        data['ymax'].append(ymax)
        data['is_crowd'].append(ann['iscrowd'])
        category_id = int(ann['category_id'])
        data['category_id'].append(category_id)
        data['category_names'].append(id_to_name_map[category_id].encode('utf8'))
        data['area'].append(ann['area'])

    return data


def obj_annotations_to_feature_dict(obj_annotations, id_to_name_map):
    data = coco_annotations_to_lists(obj_annotations, id_to_name_map)

    feature_dict = {
        'image/object/bbox/xmin':
            tfrecord_lib.convert_to_feature(data['xmin']),
        'image/object/bbox/xmax':
            tfrecord_lib.convert_to_feature(data['xmax']),
        'image/object/bbox/ymin':
            tfrecord_lib.convert_to_feature(data['ymin']),
        'image/object/bbox/ymax':
            tfrecord_lib.convert_to_feature(data['ymax']),
        'image/object/class/text':
            tfrecord_lib.convert_to_feature(data['category_names']),
        'image/object/class/label':
            tfrecord_lib.convert_to_feature(data['category_id']),
        'image/object/is_crowd':
            tfrecord_lib.convert_to_feature(data['is_crowd']),
        'image/object/area':
            tfrecord_lib.convert_to_feature(data['area'], value_type='float_list'),
    }
    return feature_dict


def flatten_segmentation(seg):
    flat_seg = []
    for i, s in enumerate(seg):
        if i > 0:
            flat_seg.extend([vocab.SEPARATOR_FLOAT, vocab.SEPARATOR_FLOAT])
        flat_seg.extend(s)
    return flat_seg


def obj_annotations_to_seg_dict(obj_annotations):
    segs = []
    seg_lens = []
    for ann in obj_annotations:
        if ann['iscrowd']:
            seg_lens.append(0)
        else:
            seg = flatten_segmentation(ann['segmentation'])
            segs.extend(seg)
            seg_lens.append(len(seg))
    seg_sep = [0] + list(np.cumsum(seg_lens))
    return {
        'image/object/segmentation_v': tfrecord_lib.convert_to_feature(segs),
        'image/object/segmentation_sep': tfrecord_lib.convert_to_feature(seg_sep),
    }


def generate_annotations(images, image_dir,
                         # panoptic_masks_dir,
                         category_id_to_name_map,
                         img_to_obj_ann,
                         enable_masks,
                         # img_to_cap_ann,
                         # img_to_key_ann,
                         # img_to_pan_ann,
                         # is_category_thing,
                         ):
    """Generator for COCO annotations."""
    for image in images:
        object_ann = img_to_obj_ann.get(image['id'], {})

        # caption_ann = img_to_cap_ann.get(image['id'], {})
        #
        # keypoint_ann = img_to_key_ann.get(image['id'], {})
        #
        # panoptic_ann = img_to_pan_ann.get(image['id'], {})

        yield (image, image_dir,
               # panoptic_masks_dir,
               category_id_to_name_map,
               object_ann,
               enable_masks,
               # caption_ann,
               # keypoint_ann,
               # panoptic_ann,
               # is_category_thing,
               )


def create_tf_example(
        image,
        image_dir,
        # panoptic_masks_dir,
        category_id_to_name_map,
        object_ann,
        enable_masks,
        # caption_ann,
        # keypoint_ann, panoptic_ann,
        # is_category_thing
):
    """Converts image and annotations to a tf.Example proto."""
    # Add image features.
    image_height = image['height']
    image_width = image['width']
    filename = image['file_name']
    image_id = image['id']

    with tf.io.gfile.GFile(os.path.join(image_dir, filename), 'rb') as fid:
        encoded_jpg = fid.read()
    feature_dict = tfrecord_lib.image_info_to_feature_dict(
        image_height, image_width, filename, image_id, encoded_jpg, 'jpg')

    # Add annotation features.
    if object_ann:
        obj_feature_dict = obj_annotations_to_feature_dict(
            object_ann,
            category_id_to_name_map)
        feature_dict.update(obj_feature_dict)

        if enable_masks:
            seg_feature_dict = obj_annotations_to_seg_dict(object_ann)
            feature_dict.update(seg_feature_dict)

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example, 0


def main():
    params: Params = paramparse.process(Params)

    assert params.ann_file, "ann_file must be provided"
    assert params.image_dir, "image_dir must be provided"

    if params.ann_suffix:
        params.ann_file = f'{params.ann_file}-{params.ann_suffix}'

    if params.start_seq_id > 0 or params.end_seq_id >= 0:
        assert params.end_seq_id >= params.start_seq_id, "end_seq_id must to be >= start_seq_id"
        params.ann_file = f'{params.ann_file}-seq-{params.start_seq_id}_{params.end_seq_id}'

    params.ann_file = os.path.join(params.image_dir, f'{params.ann_file}.{params.ann_ext}')

    image_info, category_id_to_name_map, img_to_obj_ann = (
        load_instance_annotations(params.ann_file))

    if not params.output_dir:
        params.output_dir = os.path.join(params.image_dir, 'tfrecord')

    # img_to_key_ann = load_keypoint_annotations(params.key_ann_file)
    # img_to_cap_ann = load_caption_annotations(params.cap_ann_file)
    # img_to_pan_ann, is_category_thing = load_panoptic_annotations(
    #     params.pan_ann_file)

    os.makedirs(params.output_dir, exist_ok=True)

    out_name = os.path.basename(params.ann_file).split(os.extsep)[0]

    annotations_iter = generate_annotations(
        images=image_info,
        image_dir=params.image_dir,
        # panoptic_masks_dir=params.pan_masks_dir,
        category_id_to_name_map=category_id_to_name_map,
        img_to_obj_ann=img_to_obj_ann,
        enable_masks=params.enable_masks,
        # img_to_cap_ann=img_to_cap_ann,
        # img_to_key_ann=img_to_key_ann,
        # img_to_pan_ann=img_to_pan_ann,
        # is_category_thing=is_category_thing
    )
    output_path = os.path.join(params.output_dir, out_name)
    os.makedirs(output_path, exist_ok=True)

    tfrecord_pattern = os.path.join(output_path, 'shard')

    tfrecord_lib.write_tf_record_dataset(
        output_path=tfrecord_pattern,
        annotation_iterator=annotations_iter,
        process_func=create_tf_example,
        num_shards=params.num_shards,
        multiple_processes=params.n_proc)

    print(f'out_name: {out_name}')
    print(f'output_path: {output_path}')


if __name__ == '__main__':
    main()
