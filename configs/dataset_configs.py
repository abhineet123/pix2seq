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

_transforms_config = D(
    scale_jitter=1,
    fixed_crop=1,
    jitter_scale_min=0.3,
    jitter_scale_max=2.0,
    object_order='random',
)

_shared_dataset_config = D(
    batch_duplicates=1,
    cache_dataset=True,

    train_name='',
    train_suffix='',
    train_split='train',
    train_filename_for_metrics='',
    train_start_seq_id=0,
    train_end_seq_id=-1,
    train_start_frame_id=0,
    train_end_frame_id=-1,
    train_num_examples=-1,

    eval_name='',
    eval_filename_for_metrics='',
    eval_suffix='',
    eval_split='validation',
    eval_start_seq_id=0,
    eval_end_seq_id=-1,
    eval_start_frame_id=0,
    eval_end_frame_id=-1,
    eval_num_examples=-1,

    transforms=_transforms_config
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

    return D(
        name='ipsc_object_detection',
        root_dir=root_dir,
        label_shift=0,
        compressed=0,
        **_shared_dataset_config
    )


def get_ipsc_video_data():
    root_dir = './datasets/ipsc/well3/all_frames_roi'

    return D(
        name='ipsc_video_detection',
        root_dir=root_dir,
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


def ipsc_post_process(ds_cfg, training):
    import os

    is_video = 'video' in ds_cfg.name

    root_dir = ds_cfg.root_dir
    ds_cfg.image_dir = root_dir

    if is_video:
        db_root_dir = os.path.join(root_dir, 'ytvis19')
        db_type = 'videos'
    else:
        db_root_dir = root_dir
        db_type = 'images'

    ds_cfg.db_root_dir = db_root_dir

    # if not cfg.eval_name:
    #     cfg.eval_name = cfg.train_name

    if training:
        modes = ['train', 'eval']
    else:
        modes = ['eval']

    for mode in modes:
        name = ds_cfg[f'{mode}_name']
        if not name:
            print(f'skipping {mode} postprocessing with no name specified')
            continue

        if is_video:
            if ds_cfg.length:
                length_suffix = f'length-{ds_cfg.length}'
                if length_suffix not in name:
                    name = f'{name}-{length_suffix}'
            try:
                stride = ds_cfg[f'{mode}_stride']
            except KeyError:
                stride = ds_cfg[f'{mode}_stride'] = ds_cfg[f'stride']

            if stride:
                stride_suffix = f'stride-{stride}'
                if stride_suffix not in name:
                    name = f'{name}-{stride_suffix}'

        suffix = ds_cfg[f'{mode}_suffix']
        """suffix is already in name when the latter is loaded from a trained model config.json"""
        if suffix and not name.endswith(suffix):
            name = f'{name}-{suffix}'

        start_seq_id = ds_cfg[f'{mode}_start_seq_id']
        end_seq_id = ds_cfg[f'{mode}_end_seq_id']
        if start_seq_id > 0 or end_seq_id >= 0:
            assert end_seq_id >= start_seq_id, "end_seq_id must to be >= start_seq_id"
            seq_sufix = f'seq-{start_seq_id}_{end_seq_id}'
            name = f'{name}-{seq_sufix}'

        start_frame_id = ds_cfg[f'{mode}_start_frame_id']
        end_frame_id = ds_cfg[f'{mode}_end_frame_id']
        if start_frame_id > 0 or end_frame_id >= 0:
            frame_suffix = f'{start_frame_id}_{end_frame_id}'
            name = f'{name}-{frame_suffix}'

        json_name = f'{name}.json'
        if ds_cfg.compressed:
            json_name += '.gz'
        json_path = os.path.join(db_root_dir, json_name)
        if ds_cfg.compressed:
            import compress_json
            json_dict = compress_json.load(json_path)
        else:
            import json
            with open(json_path, 'r') as fid:
                json_dict = json.load(fid)

        num_examples = len(json_dict[db_type])

        ds_cfg[f'{mode}_name'] = name
        # cfg[f'{mode}_json_name'] = json_name
        # cfg[f'{mode}_json_path'] = json_path
        ds_cfg[f'{mode}_num_examples'] = num_examples
        ds_cfg[f'{mode}_filename_for_metrics'] = json_name

        if is_video:
            try:
                frame_gaps = ds_cfg[f'{mode}_frame_gaps']
            except KeyError:
                frame_gaps = ds_cfg[f'{mode}_frame_gaps'] = []

            if len(frame_gaps) > 1:
                frame_gaps_suffix = 'fg_' + '_'.join(map(str, frame_gaps))
                if frame_gaps_suffix not in name:
                    name = f'{name}-{frame_gaps_suffix}'

        ds_cfg[f'{mode}_file_pattern'] = os.path.join(db_root_dir, 'tfrecord', name, 'shard*')

    ds_cfg.category_names_path = os.path.join(db_root_dir, ds_cfg.train_filename_for_metrics)
    ds_cfg.coco_annotations_dir_for_metrics = db_root_dir


dataset_configs = {
    'ipsc_object_detection': get_ipsc_data(),
    'ipsc_video_detection': get_ipsc_video_data(),
    # 'coco/2017_object_detection':
    #     D(
    #         name='coco/2017_object_detection',
    #         train_filename_for_metrics='instances_train2017.json',
    #         eval_filename_for_metrics='instances_val2017.json',
    #         category_names_path=os.path.join(
    #             _shared_coco_dataset_config['coco_annotations_dir_for_metrics'],
    #             'instances_val2017.json'),
    #         **_shared_coco_dataset_config
    #     ),
    # 'coco/2017_instance_segmentation':
    #     D(
    #         name='coco/2017_instance_segmentation',
    #         train_filename_for_metrics='instances_train2017.json',
    #         eval_filename_for_metrics='instances_val2017.json',
    #         category_names_path=os.path.join(
    #             _shared_coco_dataset_config['coco_annotations_dir_for_metrics'],
    #             'instances_val2017.json'),
    #         **_shared_coco_dataset_config
    #     ),
    # 'coco/2017_keypoint_detection':
    #     D(
    #         name='coco/2017_keypoint_detection',
    #         train_filename_for_metrics='person_keypoints_train2017.json',
    #         eval_filename_for_metrics='person_keypoints_val2017.json',
    #         category_names_path=os.path.join(
    #             _shared_coco_dataset_config['coco_annotations_dir_for_metrics'],
    #             'person_keypoints_val2017.json'),
    #         **_shared_coco_dataset_config
    #     ),
    # 'coco/2017_captioning':
    #     D(name='coco/2017_captioning',
    #       train_filename_for_metrics='captions_train2017_eval_compatible.json',
    #       eval_filename_for_metrics='captions_val2017_eval_compatible.json',
    #       **_shared_coco_dataset_config),
}
