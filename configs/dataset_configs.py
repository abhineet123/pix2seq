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
"""Dataset configs."""
import os
from configs.config_base import D

_shared_dataset_config = D(
    batch_duplicates=1,
    cache_dataset=True,
)

# IPSC_NAME_TO_NUM = dict(
#     ext_reorg_roi_g2_0_53=1674,
#     ext_reorg_roi_g2_16_53=1178,
#     ext_reorg_roi_g2_54_126=2263,
#     ext_reorg_roi_g2_0_1=62,
#     ext_reorg_roi_g2_0_15=496,
#     ext_reorg_roi_g2_0_37=1178,
#     ext_reorg_roi_g2_38_53=496,
# )

# Generate tfrecords for the dataset using data/scripts/create_coco_tfrecord.py
# and add paths here.
COCO_TRAIN_TFRECORD_PATTERN = 'gs://pix2seq/multi_task/data/coco/tfrecord/train*'
COCO_VAL_TFRECORD_PATTERN = 'gs://pix2seq/multi_task/data/coco/tfrecord/val*'
# Download from gs://pix2seq/multi_task/data/coco/json
COCO_ANNOTATIONS_DIR = '/tmp/coco_annotations'

_shared_coco_dataset_config = D(
    # train_file_pattern=COCO_TRAIN_TFRECORD_PATTERN,
    # eval_file_pattern=COCO_VAL_TFRECORD_PATTERN,
    train_num_examples=118287,
    eval_num_examples=5000,
    train_split='train',
    eval_split='validation',
    # Directory of annotations used by the metrics.
    # Also need to set train_filename_for_metrics and eval_filename_for_metrics.
    # If unset, groundtruth annotations should be specified via
    # record_groundtruth.
    coco_annotations_dir_for_metrics=COCO_ANNOTATIONS_DIR,
    label_shift=0,
    **_shared_dataset_config
)


def get_ipsc_data():
    root_dir = './datasets/ipsc/well3/all_frames_roi'

    train_name = 'ext_reorg_roi_g2_0_53'
    eval_name = 'ext_reorg_roi_g2_16_53'

    return D(
        name='ipsc_object_detection',
        root_dir=root_dir,
        train_name=train_name,
        eval_name=eval_name,
        train_split='train',
        eval_split='validation',
        label_shift=0,
        compressed=0,
        **_shared_dataset_config
    )


def get_ipsc_video_data():
    root_dir = './datasets/ipsc/well3/all_frames_roi'

    train_name = 'ext_reorg_roi_g2_0_4'
    eval_name = 'ext_reorg_roi_g2_5_9'

    return D(
        name='ipsc_video_detection',
        root_dir=root_dir,
        train_name=train_name,
        eval_name=eval_name,
        train_split='train',
        eval_split='validation',
        label_shift=0,
        compressed=1,
        max_disp=0.01,
        length=2,

        train_stride=1,
        train_frame_gaps=[],
        eval_stride=1,
        eval_frame_gaps=[],

        **_shared_dataset_config
    )


def ipsc_post_process(cfg):
    import os

    is_video = 'video' in cfg.name

    root_dir = cfg.root_dir
    cfg.image_dir = root_dir

    if is_video:
        db_root_dir = os.path.join(root_dir, 'ytvis19')
        db_type = 'videos'
        for mode in ['train', 'eval']:
            name = cfg[f'{mode}_name']
            if cfg.length:
                length_suffix = f'length-{cfg.length}'
                if length_suffix not in name:
                    name = f'{name}-{length_suffix}'
            stride = cfg[f'{mode}_stride']
            if stride:
                stride_suffix = f'stride-{stride}'
                if stride_suffix not in name:
                    name = f'{name}-{stride_suffix}'
            frame_gaps = cfg[f'{mode}_frame_gaps']
            if frame_gaps:
                frame_gaps_suffix = 'fg_' + '_'.join(map(str, frame_gaps))
                if frame_gaps_suffix not in name:
                    name = f'{name}-{frame_gaps_suffix}'
            cfg[f'{mode}_name'] = name
    else:
        db_root_dir = root_dir
        db_type = 'images'

    cfg.db_root_dir = db_root_dir

    for mode in ['train', 'eval']:
        name = cfg[f'{mode}_name']

        json_name = f'{name}.json'
        if cfg.compressed:
            name += '.gz'
        json_path = os.path.join(db_root_dir, json_name)
        if cfg.compressed:
            import compress_json
            json_dict = compress_json.load(json_path)
        else:
            import json
            with open(json_path, 'r') as fid:
                json_dict = json.load(fid)

        num_examples = len(json_dict[db_type])
        cfg[f'{mode}_num_examples'] = num_examples
        cfg[f'{mode}_filename_for_metrics'] = json_name
        cfg[f'{mode}_file_pattern'] = os.path.join(db_root_dir, 'tfrecord', name + '*')

    cfg.category_names_path = os.path.join(db_root_dir, cfg.eval_filename_for_metrics)
    cfg.coco_annotations_dir_for_metrics = db_root_dir


dataset_configs = {
    'ipsc_object_detection': get_ipsc_data(),
    'ipsc_video_detection': get_ipsc_video_data(),
    'coco/2017_object_detection':
        D(
            name='coco/2017_object_detection',
            train_filename_for_metrics='instances_train2017.json',
            eval_filename_for_metrics='instances_val2017.json',
            category_names_path=os.path.join(
                _shared_coco_dataset_config['coco_annotations_dir_for_metrics'],
                'instances_val2017.json'),
            **_shared_coco_dataset_config
        ),
    'coco/2017_instance_segmentation':
        D(
            name='coco/2017_instance_segmentation',
            train_filename_for_metrics='instances_train2017.json',
            eval_filename_for_metrics='instances_val2017.json',
            category_names_path=os.path.join(
                _shared_coco_dataset_config['coco_annotations_dir_for_metrics'],
                'instances_val2017.json'),
            **_shared_coco_dataset_config
        ),
    'coco/2017_keypoint_detection':
        D(
            name='coco/2017_keypoint_detection',
            train_filename_for_metrics='person_keypoints_train2017.json',
            eval_filename_for_metrics='person_keypoints_val2017.json',
            category_names_path=os.path.join(
                _shared_coco_dataset_config['coco_annotations_dir_for_metrics'],
                'person_keypoints_val2017.json'),
            **_shared_coco_dataset_config
        ),
    'coco/2017_captioning':
        D(name='coco/2017_captioning',
          train_filename_for_metrics='captions_train2017_eval_compatible.json',
          eval_filename_for_metrics='captions_val2017_eval_compatible.json',
          **_shared_coco_dataset_config),
}
