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
import os
import sys

sys.path.append(os.getcwd())

from absl import app
from absl import flags
from absl import logging
import numpy as np

import paramparse
from data.scripts import tfrecord_lib


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
    if annotation_path.endswith('.json'):
        import json
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
    elif annotation_path.endswith('.json.gz'):
        import compress_json
        annotations = compress_json.load(annotation_path)
    else:
        raise AssertionError(f'Invalid annotation_path: {annotation_path}')

    video_info = annotations['videos']
    category_id_to_name_map = dict(
        (element['id'], element['name']) for element in annotations['categories'])

    vid_to_ann = collections.defaultdict(list)
    for ann in annotations['annotations']:
        vid_id = ann['video_id']
        vid_to_ann[vid_id].append(ann)

    return video_info, category_id_to_name_map, vid_to_ann


def ytvis_annotations_to_lists(obj_annotations: dict, id_to_name_map: dict, vid_len: int):
    """
    Converts YTVIS annotations to feature lists.
    """

    data = dict((k, list()) for k in [
        'is_crowd', 'category_id',
        'category_names', 'area', 'target_id'])

    for _id in range(vid_len):
        frame_data = dict((k, list()) for k in [
            f'xmin-{_id}', f'xmax-{_id}', f'ymin-{_id}', f'ymax-{_id}', f'area-{_id}'
        ])
        data.update(frame_data)

    for ann_id, ann in enumerate(obj_annotations):
        bboxes = ann['bboxes']
        areas = ann['areas']
        assert len(bboxes) == vid_len, \
            f"number of boxes: {len(bboxes)} does not match the video length: {vid_len}"
        assert len(areas) == vid_len, \
            f"number of areas: {len(areas)} does not match the video length: {vid_len}"

        data['is_crowd'].append(ann['iscrowd'])
        data['target_id'].append(int(ann['id']))
        category_id = int(ann['category_id'])
        data['category_id'].append(category_id)
        data['category_names'].append(id_to_name_map[category_id].encode('utf8'))

        for bbox_id, bbox in enumerate(bboxes):
            area = areas[bbox_id]

            if bbox is None:
                assert area is None, "null bbox for non-null area"
                # xmin, ymin, xmax, ymax = -1, -1, -1, -1
                # area = -1
                # xmin = ymin = xmax = ymax = area = -1
                xmin = ymin = xmax = ymax = area = None

            else:
                assert area is not None, "null area for non-null bbox"
                (x, y, width, height) = tuple(bbox)
                xmin, ymin, xmax, ymax = float(x), float(y), float(x + width), float(y + height)

            data[f'xmin-{bbox_id}'].append(xmin)
            data[f'xmax-{bbox_id}'].append(xmax)
            data[f'ymin-{bbox_id}'].append(ymin)
            data[f'ymax-{bbox_id}'].append(ymax)
            data[f'area-{bbox_id}'].append(area)
    return data


def obj_annotations_to_feature_dict(obj_annotations, id_to_name_map, vid_len):
    """Convert COCO annotations to an encoded feature dict.

    Args:
      obj_annotations: a list of object annotations.
      id_to_name_map: category id to category name map.

    Returns:
      a dict of tf features.
    """

    data = ytvis_annotations_to_lists(obj_annotations, id_to_name_map, vid_len)
    feature_dict = {
        'video/object/class/text':
            tfrecord_lib.convert_to_feature(data['category_names']),
        'video/object/class/label':
            tfrecord_lib.convert_to_feature(data['category_id']),
        'video/object/is_crowd':
            tfrecord_lib.convert_to_feature(data['is_crowd']),
        'video/object/target_id':
            tfrecord_lib.convert_to_feature(data['target_id']),
    }

    for _id in range(vid_len):
        frame_feature_dict = {
            f'video/object/bbox/xmin-{_id}':
                tfrecord_lib.convert_to_feature(data[f'xmin-{_id}']),
            f'video/object/bbox/xmax-{_id}':
                tfrecord_lib.convert_to_feature(data[f'xmax-{_id}']),
            f'video/object/bbox/ymin-{_id}':
                tfrecord_lib.convert_to_feature(data[f'ymin-{_id}']),
            f'video/object/bbox/ymax-{_id}':
                tfrecord_lib.convert_to_feature(data[f'ymax-{_id}']),
            f'video/object/area-{_id}':
                tfrecord_lib.convert_to_feature(data[f'area-{_id}'], value_type='float_list'),
        }
        feature_dict.update(frame_feature_dict)

    return feature_dict


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
    video_height = video['height']
    video_width = video['width']
    file_names = video['file_names']
    video_id = video['id']

    vid_len = len(file_names)

    feature_dict = tfrecord_lib.video_info_to_feature_dict(
        video_height, video_width, file_names, video_id)

    if object_ann:
        # Bbox, area, etc.
        obj_feature_dict = obj_annotations_to_feature_dict(
            object_ann,
            category_id_to_name_map,
            vid_len)
        feature_dict.update(obj_feature_dict)

    import tensorflow as tf
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example, 0


class Params(paramparse.CFG):
    def __init__(self):
        paramparse.CFG.__init__(self, cfg_root='cfg/video_tfrecord')
        self.ann_file = ''
        self.ann_ext = 'json'

        self.length = 0
        self.stride = 0

        self.image_dir = ''
        self.image_dir = ''
        self.n_proc = 0
        self.num_shards = 32
        self.output_dir = ''


def main(_):
    params = Params()
    paramparse.process(params)

    assert params.image_dir, "image_dir must be provided"
    assert params.ann_file, "ann_file must be provided"

    params.ann_file = os.path.join(params.image_dir, 'ytvis19', f'{params.ann_file}.{params.ann_ext}')

    video_info, category_id_to_name_map, vid_to_obj_ann = (
        load_ytvis_annotations(params.ann_file))

    if not params.output_dir:
        output_dir = os.path.join(params.image_dir, 'ytvis19', 'tfrecord')
        print(f'setting output_dir to {output_dir}')
        params.output_dir = output_dir

    os.makedirs(params.output_dir, exist_ok=True)

    out_name = os.path.basename(params.ann_file).split(os.extsep)[0]
    annotations_iter = generate_video_annotations(
        videos=video_info,
        category_id_to_name_map=category_id_to_name_map,
        vid_to_obj_ann=vid_to_obj_ann,
    )
    output_path = os.path.join(params.output_dir, out_name)

    print(f'out_name: {out_name}')
    print(f'output_path: {output_path}')

    tfrecord_lib.write_tf_record_dataset(
        output_path=output_path,
        annotation_iterator=annotations_iter,
        process_func=create_video_tf_example,
        num_shards=params.num_shards,
        multiple_processes=params.n_proc)


# Note: internal version of the code overrides this function.
def run_main():
    app.run(main)


if __name__ == '__main__':
    run_main()
