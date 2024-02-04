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
# from absl import flags
from absl import logging
# import numpy as np

import paramparse
from data.scripts import tfrecord_lib


def save_ytvis_annotations(json_dict, json_path):
    print(f'saving ytvis annotations to {json_path}')
    json_kwargs = dict(
        indent=4
    )
    if json_path.endswith('.json.gz'):
        import compress_json
        compress_json.dump(json_dict, json_path, json_kwargs=json_kwargs)
    elif json_path.endswith('.json'):
        import json
        output_json = json.dumps(json_dict, **json_kwargs)
        with open(json_path, 'w') as f:
            f.write(output_json)
    else:
        raise AssertionError(f'Invalid json_path: {json_path}')


def load_ytvis_annotations(annotation_path, vid_id_offset):
    print(f'Reading ytvis annotations from {annotation_path}')
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

    if vid_id_offset > 0:
        for vid in video_info:
            vid["id"] += vid_id_offset

    category_id_to_name_map = dict(
        (element['id'], element['name']) for element in annotations['categories'])

    vid_to_ann = collections.defaultdict(list)
    for ann in annotations['annotations']:
        ann['video_id'] += vid_id_offset
        vid_id = ann['video_id']
        vid_to_ann[vid_id].append(ann)

    return video_info, category_id_to_name_map, vid_to_ann, annotations


def ytvis_annotations_to_lists(obj_annotations: dict, id_to_name_map: dict, vid_len: int):
    """
    Converts YTVIS annotations to feature lists.
    """

    data = dict((k, list()) for k in [
        'is_crowd', 'category_id',
        'category_names', 'target_id'])

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

        valid_box_exists = False

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

                assert ymax > ymin and xmax > xmin, f"invalid bbox: {bbox}"

                valid_box_exists = True

            # if bbox_id == 0:
            #     xmin = ymin = xmax = ymax = area = None

            data[f'xmin-{bbox_id}'].append(xmin)
            data[f'xmax-{bbox_id}'].append(xmax)
            data[f'ymin-{bbox_id}'].append(ymin)
            data[f'ymax-{bbox_id}'].append(ymax)
            data[f'area-{bbox_id}'].append(area)

        if not valid_box_exists:
            raise AssertionError('all boxes for an object cannot be None')

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
        image_dir,

):
    """Generator for COCO annotations."""
    for video in videos:
        object_ann = vid_to_obj_ann.get(video['id'], {})
        yield (
            video,
            category_id_to_name_map,
            object_ann,
            image_dir,
        )


def create_video_tf_example(
        video,
        category_id_to_name_map,
        object_ann,
        image_dir,
):
    file_ids = video['file_ids']

    assert all(i < j for i, j in zip(file_ids, file_ids[1:])), \
        "file_ids should be strictly increasing"

    video_height = video['height']
    video_width = video['width']
    file_names = video['file_names']
    video_id = video['id']

    vid_len = len(file_names)

    feature_dict = tfrecord_lib.video_info_to_feature_dict(
        video_height, video_width, file_names, file_ids, video_id, image_dir)

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

        self.frame_gaps = []
        self.length = 0
        self.stride = 0

        self.image_dir = ''
        self.n_proc = 0
        self.num_shards = 32
        self.output_dir = ''


def main(_):
    params = Params()
    paramparse.process(params)

    assert params.image_dir, "image_dir must be provided"
    assert params.ann_file, "ann_file must be provided"

    assert os.path.exists(params.image_dir), f"image_dir does not exist: {params.image_dir}"

    if params.length:
        params.ann_file = f'{params.ann_file}-length-{params.length}'

    if params.stride:
        params.ann_file = f'{params.ann_file}-stride-{params.stride}'

    if params.frame_gaps:
        ann_files = [f'{params.ann_file}-frame_gap-{frame_gap}' if frame_gap > 1 else params.ann_file
                     for frame_gap in params.frame_gaps]
    else:
        ann_files = [params.ann_file, ]

    # params.ann_file = None

    ann_files = [os.path.join(params.image_dir, 'ytvis19', f'{ann_file}.{params.ann_ext}') for ann_file in ann_files]
    vid_id_offset = 0
    n_all_vid = 0
    n_all_ann = 0
    video_info = []
    annotations_all = {}
    category_id_to_name_map = {}
    vid_to_obj_ann = collections.defaultdict(list)

    for ann_file in ann_files:
        assert os.path.exists(ann_file), f"ann_file does not exist: {ann_file}"
        video_info_, category_id_to_name_map_, vid_to_obj_ann_, annotations_ = load_ytvis_annotations(
            ann_file, vid_id_offset)
        new_vid_ids = set(vid_to_obj_ann_.keys())
        n_new_vid_ids = len(new_vid_ids)
        counts_ = annotations_['info']['counts'][0]
        n_new_vids = len(annotations_['videos'])
        n_new_anns = len(annotations_['annotations'])

        assert n_new_vids == counts_['videos'], "videos counts mismatch"
        assert n_new_vids == n_new_vid_ids, "n_new_vid_ids mismatch"
        assert n_new_anns == counts_['annotations'], "annotations counts mismatch"

        existing_vid_ids = set(vid_to_obj_ann.keys())
        overlapped_vid_ids = existing_vid_ids.intersection(new_vid_ids)

        assert not overlapped_vid_ids, f"overlapped_vid_ids found: {overlapped_vid_ids}"

        if not annotations_all:
            annotations_all = annotations_
        else:
            annotations_all['videos'] += annotations_['videos']
            annotations_all['annotations'] += annotations_['annotations']
            # annotations_all['categories'] += annotations_['categories']

            counts = annotations_all['info']['counts'][0]

            counts['videos'] += n_new_vids
            counts['annotations'] += n_new_anns

        n_all_vid += n_new_vids
        n_all_ann += n_new_anns

        counts = annotations_all['info']['counts'][0]

        assert n_all_vid == counts['videos'], "n_all_vid mismatch"
        assert n_all_ann == counts['annotations'], "n_all_ann mismatch"

        assert n_all_vid == len(annotations_all['videos']), "annotations_all videos mismatch"
        assert n_all_ann == len(annotations_all['annotations']), "annotations_all annotations mismatch"

        video_info += video_info_
        vid_to_obj_ann.update(vid_to_obj_ann_)
        category_id_to_name_map.update(category_id_to_name_map_)

        vid_id_offset = max(vid_to_obj_ann.keys())

    # categories_all = annotations_all['categories']
    # categories_unique = [dict(t) for t in {tuple(d.items()) for d in categories_all}]
    # annotations_all['categories'] = categories_unique

    if not params.output_dir:
        output_dir = os.path.join(params.image_dir, 'ytvis19', 'tfrecord')
        print(f'setting output_dir to {output_dir}')
        params.output_dir = output_dir

    os.makedirs(params.output_dir, exist_ok=True)

    out_name = os.path.basename(ann_files[0]).split(os.extsep)[0]

    if len(params.frame_gaps) > 1:
        frame_gaps_suffix = 'fg_' + '_'.join(map(str, params.frame_gaps))
        if frame_gaps_suffix not in out_name:
            out_name = f'{out_name}-{frame_gaps_suffix}'

        out_json_path = os.path.join(params.image_dir, 'ytvis19', f'{out_name}.{params.ann_ext}')
        annotations_all['info']['description'] = out_name
        save_ytvis_annotations(annotations_all, out_json_path)

    annotations_iter = generate_video_annotations(
        videos=video_info,
        category_id_to_name_map=category_id_to_name_map,
        vid_to_obj_ann=vid_to_obj_ann,
        image_dir=params.image_dir,
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
