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


def ipsc_post_process(dataset_cfg):
    import os

    is_video = 'video' in dataset_cfg.name

    root_dir = dataset_cfg.root_dir
    train_name = dataset_cfg.train_name
    eval_name = dataset_cfg.eval_name

    dataset_cfg.train.image_dir = root_dir

    if is_video:
        db_root_dir = os.path.join(root_dir, 'ytvis19')
        db_type = 'videos'

        if dataset_cfg.length:
            train_name = f'{train_name}-length-{dataset_cfg.length}'

        if dataset_cfg.stride:
            train_name = f'{train_name}-stride-{dataset_cfg.stride}'
    else:
        db_root_dir = root_dir
        db_type = 'images'

    train_json_name = f'{train_name}.json'
    eval_json_name = f'{eval_name}.json'

    train_json_path = os.path.join(db_root_dir, train_json_name)
    eval_json_path = os.path.join(db_root_dir, eval_json_name)

    if dataset_cfg.compressed:
        import compress_json
        train_json_path += '.gz'
        eval_json_path += '.gz'
        train_dict = compress_json.load(train_json_path)
        eval_dict = compress_json.load(eval_json_path)
    else:
        import json
        with open(train_json_path, 'r') as fid:
            train_dict = json.load(fid)
        with open(eval_json_path, 'r') as fid:
            eval_dict = json.load(fid)

    num_train = len(train_dict[db_type])
    num_eval = len(eval_dict[db_type])

    dataset_cfg.train_num_examples = num_train
    dataset_cfg.eval_num_examples = num_eval

    dataset_cfg.train_filename_for_metrics = train_json_name
    dataset_cfg.eval_filename_for_metrics = eval_json_name

    dataset_cfg.train_file_pattern = os.path.join(db_root_dir, 'tfrecord', train_name + '*')
    dataset_cfg.eval_file_pattern = os.path.join(db_root_dir, 'tfrecord', eval_name + '*')

    dataset_cfg.category_names_path = os.path.join(root_dir, dataset_cfg.eval_filename_for_metrics)
    dataset_cfg.coco_annotations_dir_for_metrics = root_dir


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

    train_name = 'ipsc-ext_reorg_roi_g2_0_4'
    eval_name = 'ipsc-ext_reorg_roi_g2_5_9'

    return D(
        name='ipsc_video_detection',
        root_dir=root_dir,
        train_name=train_name,
        eval_name=eval_name,
        train_split='train',
        eval_split='validation',
        label_shift=0,
        compressed=0,
        length=2,
        stride=1,
        **_shared_dataset_config
    )


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
