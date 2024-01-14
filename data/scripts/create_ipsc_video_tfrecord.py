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

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

import vocab
from data.scripts import tfrecord_lib

flags.DEFINE_string('image_dir', './datasets/ipsc/well3/all_frames_roi', 'Directory containing images.')
# flags.DEFINE_string('ann_file', './datasets/ipsc/well3/all_frames_roi/ext_reorg_roi_g2_0_53.json',
# 'Instance annotation file.')
flags.DEFINE_string('ann_file', 'ext_reorg_roi_g2_54_126.json', 'Instance annotation file.')
# flags.DEFINE_string('pan_ann_file', '', 'Panoptic annotation file.')
# flags.DEFINE_string('pan_masks_dir', '', 'Directory containing panoptic masks.')
flags.DEFINE_string('output_dir', './datasets/ipsc/well3/all_frames_roi/tfrecord', 'Output directory')
flags.DEFINE_integer('num_shards', 32, 'Number of shards for output file.')
flags.DEFINE_integer('n_proc', 0, 'n_proc')

FLAGS = flags.FLAGS


def load_ytvis_annotations(annotation_path):
    """Load instance annotations.

    Args:
      annotation_path: str. Path to the annotation file.

    Returns:
      image_info: a list of dicts, with information such as file name, image id,
          height, width, etc.
      category_id_to_name_map: dict of category ids to category names.
      img_to_ann: a dict of image_id to annotation.
    """
    logging.info(f'Reading ytvis annotations from {annotation_path}')
    with tf.io.gfile.GFile(annotation_path, 'r') as f:
        annotations = json.load(f)

    video_info = annotations['videos']
    category_id_to_name_map = dict(
        (element['id'], element['name']) for element in annotations['categories'])

    vid_to_ann = collections.defaultdict(list)
    for ann in annotations['annotations']:
        vid_id = ann['video_id']
        vid_to_ann[vid_id].append(ann)

    return video_info, category_id_to_name_map, vid_to_ann


def coco_annotations_to_lists(obj_annotations, id_to_name_map):
    """Converts COCO annotations to feature lists.

    Args:
      obj_annotations: a list of object annotations.
      id_to_name_map: category id to category name map.

    Returns:
      a dict of list features.
    """

    data = dict((k, list()) for k in [
        'xmin', 'xmax', 'ymin', 'ymax', 'is_crowd', 'category_id',
        'category_names', 'area'])

    for ann in obj_annotations:
        (x, y, width, height) = tuple(ann['bbox'])
        data['xmin'].append(float(x))
        data['xmax'].append(float(x + width))
        data['ymin'].append(float(y))
        data['ymax'].append(float(y + height))
        data['is_crowd'].append(ann['iscrowd'])
        category_id = int(ann['category_id'])
        data['category_id'].append(category_id)
        data['category_names'].append(id_to_name_map[category_id].encode('utf8'))
        data['area'].append(ann['area'])

    return data


def obj_annotations_to_feature_dict(obj_annotations, id_to_name_map):
    """Convert COCO annotations to an encoded feature dict.

    Args:
      obj_annotations: a list of object annotations.
      id_to_name_map: category id to category name map.

    Returns:
      a dict of tf features.
    """

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
    """Flatten the segmentation polygon list of lists into a single list."""
    flat_seg = []
    for i, s in enumerate(seg):
        if i > 0:
            flat_seg.extend([vocab.SEPARATOR_FLOAT, vocab.SEPARATOR_FLOAT])
        flat_seg.extend(s)
    return flat_seg


def obj_annotations_to_seg_dict(obj_annotations):
    """Get the segmentation features from instance annotations."""
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


def key_annotations_to_feature_dict(key_annotations, obj_annotations):
    """Get the keypoints features from keypoints annotations."""
    oids = [ann['id'] for ann in obj_annotations]
    keys = []
    key_lens = []
    num_keypoints = []
    for oid in oids:
        found = False
        for ann in key_annotations:
            if oid == ann['id']:
                found = True
                key = ann['keypoints']
                keys.extend(key)
                key_lens.append(len(key))
                num_keypoints.append(ann['num_keypoints'])
                break
        if not found:
            key_lens.append(0)
            num_keypoints.append(0)
    key_sep = [0] + list(np.cumsum(key_lens))
    return {
        'image/object/keypoints_v':
            tfrecord_lib.convert_to_feature(keys, value_type='float_list'),
        'image/object/keypoints_sep':
            tfrecord_lib.convert_to_feature(key_sep),
        'image/object/num_keypoints':
            tfrecord_lib.convert_to_feature(num_keypoints),
    }


def pan_annotations_to_feature_dict(pan_annotations):
    # TODO(lala) - decide what to do with panoptic annotations.
    return {}


def generate_video_annotations(
        videos,
        category_id_to_name_map,
        vid_to_obj_ann,
):
    """Generator for COCO annotations."""
    for video in videos:
        object_ann = vid_to_obj_ann.get(video['id'], {})
        yield (
            video,
            category_id_to_name_map,
            object_ann,
        )


def create_video_tf_example(
        video,
        category_id_to_name_map,
        object_ann,
):
    """Converts image and annotations to a tf.Example proto."""
    # Add image features.
    video_height = video['height']
    video_width = video['width']
    file_names = video['file_names']
    video_id = video['id']



    feature_dict = tfrecord_lib.video_info_to_feature_dict(
        video_height, video_width, file_names, video_id)

    # Add annotation features.
    if object_ann:
        # Bbox, area, etc.
        obj_feature_dict = obj_annotations_to_feature_dict(object_ann,
                                                           category_id_to_name_map)
        feature_dict.update(obj_feature_dict)

        # Polygons.
        seg_feature_dict = obj_annotations_to_seg_dict(object_ann)
        feature_dict.update(seg_feature_dict)

        # Keypoints.
        # key_feature_dict = key_annotations_to_feature_dict(keypoint_ann, object_ann)
        # feature_dict.update(key_feature_dict)

    # Captions.
    # feature_dict['image/caption'] = tfrecord_lib.convert_to_feature(caption_ann)

    # Panoptic masks.
    # pan_feature_dict = pan_annotations_to_feature_dict(panoptic_ann)
    # feature_dict.update(pan_feature_dict)

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example, 0


def main(_):
    FLAGS.ann_file = os.path.join(FLAGS.image_dir, FLAGS.ann_file)

    video_info, category_id_to_name_map, vid_to_obj_ann = (
        load_ytvis_annotations(FLAGS.ann_file))

    # img_to_key_ann = load_keypoint_annotations(FLAGS.key_ann_file)
    # img_to_cap_ann = load_caption_annotations(FLAGS.cap_ann_file)
    # img_to_pan_ann, is_category_thing = load_panoptic_annotations(
    #     FLAGS.pan_ann_file)

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    out_name = os.path.splitext(os.path.basename(FLAGS.ann_file))[0]
    annotations_iter = generate_video_annotations(
        videos=video_info,
        category_id_to_name_map=category_id_to_name_map,
        vid_to_obj_ann=vid_to_obj_ann,
    )
    output_path = os.path.join(FLAGS.output_dir, out_name)

    print(f'out_name: {out_name}')
    print(f'output_path: {output_path}')

    tfrecord_lib.write_tf_record_dataset(
        output_path=output_path,
        annotation_iterator=annotations_iter,
        process_func=create_video_tf_example,
        num_shards=FLAGS.num_shards,
        multiple_processes=FLAGS.n_proc)


# Note: internal version of the code overrides this function.
def run_main():
    app.run(main)


if __name__ == '__main__':
    run_main()
