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
from configs.config_base import D


def get_shared_det_data():
    _transforms_config = D(
        scale_jitter=1,
        fixed_crop=1,
        jitter_scale_min=0.3,
        jitter_scale_max=2.0,
        object_order='random',
    )

    _shared_dataset_config = D(
        buffer_size=300,
        batch_duplicates=1,
        cache_dataset=True,

        target_size=None,

        train_split='train',
        eval_split='validation',

        train_cfg='',
        eval_cfg='',

        category_names_path='',

        transforms=_transforms_config
    )

    for mode in ['train', 'eval']:
        _shared_dataset_config[f'{mode}_name'] = 1
        _shared_dataset_config[f'{mode}_filename_for_metrics'] = ''
        _shared_dataset_config[f'{mode}_suffix'] = ''
        _shared_dataset_config[f'{mode}_start_seq_id'] = 0
        _shared_dataset_config[f'{mode}_end_seq_id'] = -1
        _shared_dataset_config[f'{mode}_start_frame_id'] = 0
        _shared_dataset_config[f'{mode}_end_frame_id'] = -1
        _shared_dataset_config[f'{mode}_num_examples'] = -1

    return _shared_dataset_config


def get_ipsc_data():
    root_dir = './datasets/ipsc/well3/all_frames_roi'

    return D(
        name='ipsc_object_detection',
        root_dir=root_dir,
        label_shift=0,
        compressed=0,
        **get_shared_det_data()
    )


def get_ipsc_video_data():
    root_dir = './datasets/ipsc/well3/all_frames_roi'

    data = D(
        name='ipsc_video_detection',
        root_dir=root_dir,
        label_shift=0,
        compressed=1,
        max_disp=0.01,
        length=2,
        **get_shared_det_data()
    )

    for mode in ['train', 'eval']:
        data[f'{mode}_stride'] = 1
        data[f'{mode}_sample'] = 0
        data[f'{mode}_frame_gaps'] = []

    return data


def get_shared_seg_data():
    root_dir = './datasets/ipsc/well3/all_frames_roi'

    data = D(
        root_dir=root_dir,
        rle_from_json=1,
        label_shift=0,
        compressed=0,
        empty_seg_prob=0.1,

        buffer_size=300,
        batch_duplicates=1,
        cache_dataset=True,

        target_size=None,

        train_name='',
        eval_name='',

        train_split='train',
        eval_split='validation',

        train_cfg='',
        eval_cfg='',

        category_names_path='',

        transforms=D(),

        multi_class=0,
        length_as_class=0,
        starts_2d=0,
        flat_order='C',
    )
    for mode in ['train', 'eval']:
        mode_data = D()
        mode_data[f'suffix'] = ''
        mode_data[f'resize'] = 0
        mode_data[f'start_id'] = 0
        mode_data[f'end_id'] = 0
        mode_data[f'patch_height'] = 0
        mode_data[f'patch_width'] = 0
        mode_data[f'min_stride'] = 0
        mode_data[f'max_stride'] = 0
        mode_data[f'n_rot'] = 0
        mode_data[f'min_rot'] = 0
        mode_data[f'max_rot'] = 0
        mode_data[f'enable_flip'] = 0
        mode_data[f'seq_id'] = -1
        mode_data[f'seq_start_id'] = 0
        mode_data[f'seq_end_id'] = -1
        mode_data[f'subsample'] = 1
        mode_data[f'max_length'] = 0

        data[f'{mode}'] = mode_data
    return data


def get_sem_seg_data():
    data = D(
        name='ipsc_semantic_segmentation',
        **get_shared_seg_data()
    )
    return data


def get_vid_seg_data():
    data = D(
        name='ipsc_video_segmentation',
        time_as_class=0,
        length=2,
        **get_shared_seg_data()
    )
    for mode in ['train', 'eval']:
        data[f'{mode}_stride'] = 1
        data[f'{mode}_sample'] = 0
        data[f'{mode}_frame_gap'] = []
    return data


def ipsc_post_process(ds_cfg, task_cfg, model_cfg, training):
    import os


    if ds_cfg.target_size is not None and not ds_cfg.target_size:
        ds_cfg.target_size = task_cfg.image_size

    # elif isinstance(ds_cfg.target_size, int):
    #     ds_cfg.target_size = (ds_cfg.target_size, ds_cfg.target_size)

    is_seg = 'segmentation' in ds_cfg.name
    is_video = 'video' in ds_cfg.name

    if is_seg and not is_video and ds_cfg.length_as_class:
        ds_cfg.multi_class = 1


    root_dir = ds_cfg.root_dir
    ds_cfg.image_dir = root_dir

    db_type = 'images'

    if is_video:
        if not is_seg:
            root_dir = os.path.join(root_dir, 'ytvis19')
        db_type = 'videos'

    db_root_dir = ds_cfg.db_root_dir = root_dir

    # if not cfg.eval_name:
    #     cfg.eval_name = cfg.train_name

    if training:
        modes = ['train', 'eval']
    else:
        modes = ['eval']

    # modes = ['train', 'eval']

    multi_class = ds_cfg[f'multi_class']
    length_as_class = ds_cfg[f'length_as_class']
    starts_2d = ds_cfg[f'starts_2d']
    flat_order = ds_cfg[f'flat_order']

    for mode in modes:

        db_root_dir = ds_cfg.db_root_dir

        name = ds_cfg[f'{mode}_name']
        if not name:
            print(f'skipping {mode} postprocessing with no name specified')
            continue

        subsample = 0

        if is_seg:

            mode_cfg = ds_cfg[f'{mode}']

            suffix = mode_cfg.suffix

            resize = mode_cfg[f'resize']
            start_id = mode_cfg[f'start_id']
            end_id = mode_cfg[f'end_id']
            patch_height = mode_cfg[f'patch_height']
            patch_width = mode_cfg[f'patch_width']
            min_stride = mode_cfg[f'min_stride']
            max_stride = mode_cfg[f'max_stride']
            n_rot = mode_cfg[f'n_rot']
            min_rot = mode_cfg[f'min_rot']
            max_rot = mode_cfg[f'max_rot']
            enable_flip = mode_cfg[f'enable_flip']
            seq_id = mode_cfg[f'seq_id']
            seq_start_id = mode_cfg[f'seq_start_id']
            seq_end_id = mode_cfg[f'seq_end_id']
            subsample = mode_cfg[f'subsample']

            if not suffix:
                assert end_id >= start_id, f"invalid end_id: {end_id}"

                if patch_width <= 0:
                    mode_cfg[f'patch_width'] = patch_width = patch_height

                if min_stride <= 0:
                    mode_cfg[f'min_stride'] = min_stride = patch_height

                if max_stride <= min_stride:
                    mode_cfg[f'max_stride'] = max_stride = min_stride

                db_suffixes = []
                if resize:
                    db_suffixes.append(f'resize_{resize}')

                db_suffixes += [f'{start_id:d}_{end_id:d}',
                                f'{patch_height:d}_{patch_width:d}',
                                f'{min_stride:d}_{max_stride:d}',
                                ]
                if n_rot > 0:
                    db_suffixes.append(f'rot_{min_rot:d}_{max_rot:d}_{n_rot:d}')

                if enable_flip:
                    db_suffixes.append('flip')

                db_suffix = '-'.join(db_suffixes)
                db_root_dir = f'{db_root_dir}-{db_suffix}'
                name = f'{db_suffix}'

                if seq_id >= 0:
                    seq_start_id = seq_end_id = seq_id

                if seq_start_id > 0 or seq_end_id >= 0:
                    assert seq_end_id >= seq_start_id, "end_seq_id must to be >= start_seq_id"
                    seq_suffix = f'seq_{seq_start_id}_{seq_end_id}'
                    name = f'{name}-{seq_suffix}'
        db_name = name
        if is_video:
            """video specific suffixes"""
            assert ds_cfg.length > 1, "video length must be > 1"

            length_suffix = f'length-{ds_cfg.length}'
            if length_suffix not in db_name:
                db_name = f'{db_name}-{length_suffix}'
            try:
                stride = ds_cfg[f'{mode}_stride']
            except KeyError:
                stride = ds_cfg[f'{mode}_stride'] = ds_cfg[f'stride']

            if stride:
                stride_suffix = f'stride-{stride}'
                if stride_suffix not in db_name:
                    db_name = f'{db_name}-{stride_suffix}'
            try:
                sample = ds_cfg[f'{mode}_sample']
            except KeyError:
                sample = ds_cfg[f'{mode}_sample'] = ds_cfg[f'sample']

            if sample:
                sample_suffix = f'sample-{sample}'
                if sample_suffix not in db_name:
                    db_name = f'{db_name}-{sample_suffix}'

            if not is_seg:
                suffix = ds_cfg[f'{mode}_suffix']
                """suffix is already in name when the latter is loaded from a trained model config.json"""
                if suffix and not name.endswith(suffix):
                    db_name = f'{db_name}-{suffix}'

                start_seq_id = ds_cfg[f'{mode}_start_seq_id']
                end_seq_id = ds_cfg[f'{mode}_end_seq_id']
                if start_seq_id > 0 or end_seq_id >= 0:
                    assert end_seq_id >= start_seq_id, "end_seq_id must to be >= start_seq_id"
                    seq_suffix = f'seq-{start_seq_id}_{end_seq_id}'
                    db_name = f'{db_name}-{seq_suffix}'

                start_frame_id = ds_cfg[f'{mode}_start_frame_id']
                end_frame_id = ds_cfg[f'{mode}_end_frame_id']
                if start_frame_id > 0 or end_frame_id >= 0:
                    frame_suffix = f'{start_frame_id}_{end_frame_id}'
                    db_name = f'{db_name}-{frame_suffix}'

        json_name = db_name
        if is_seg:
            """RLE specific suffixes"""
            if subsample > 1:
                json_name = f'{json_name}-sub_{subsample}'

            if starts_2d:
                json_name = f'{json_name}-2d'

            if is_video:
                time_as_class = ds_cfg[f'time_as_class']
                if time_as_class:
                    if length_as_class:
                        json_name = f'{json_name}-ltac'
                    else:
                        json_name = f'{json_name}-tac'
                elif length_as_class:
                    assert multi_class, "multi_class must be enabled for length_as_class"
                    json_name = f'{json_name}-lac'
                if multi_class:
                    json_name = f'{json_name}-mc'
            else:
                if length_as_class:
                    assert multi_class, "multi_class must be enabled for length_as_class"
                    json_name = f'{json_name}-lac'
                elif multi_class:
                    json_name = f'{json_name}-mc'

            if flat_order != 'C':
                json_name = f'{json_name}-flat_{flat_order}'

        json_name_with_ext = f'{json_name}.json'
        if ds_cfg.compressed:
            json_name_with_ext += '.gz'

        json_path = os.path.join(db_root_dir, json_name_with_ext)

        print(f'reading {mode} json: {json_path}')
        if ds_cfg.compressed:
            import compress_json
            json_dict = compress_json.load(json_path)
        else:
            import json
            with open(json_path, 'r') as fid:
                json_dict = json.load(fid)

        num_examples = len(json_dict[db_type])
        print(f'num_examples: {num_examples}')

        # ds_cfg[f'{mode}_json_dict'] = json_dict

        ds_cfg[f'{mode}_db_root_dir'] = db_root_dir
        # cfg[f'{mode}_json_name'] = json_name
        # cfg[f'{mode}_json_path'] = json_path
        ds_cfg[f'{mode}_num_examples'] = num_examples
        ds_cfg[f'{mode}_filename_for_metrics'] = json_name_with_ext

        tf_name = json_name
        model_name = json_name
        if is_seg:
            params_from_json = json_dict['info']['params']
            model_cfg.coord_vocab_shift = params_from_json['starts_offset']
            model_cfg.len_vocab_shift = params_from_json['lengths_offset']
            model_cfg.class_vocab_shift = params_from_json['class_offset']
            mode_cfg[f'max_length'] = params_from_json['max_length']

            rle_from_json = ds_cfg.rle_from_json
            tf_name = db_name if rle_from_json else json_name

            rle_lens = [str(info['rle_len']) for info in json_dict['videos' if is_video else 'images']]
            mode_cfg.rle_lens = rle_lens

        elif is_video:
            try:
                frame_gaps = ds_cfg[f'{mode}_frame_gaps']
            except KeyError:
                frame_gaps = ds_cfg[f'{mode}_frame_gaps'] = []

            if len(frame_gaps) > 1:
                frame_gaps_suffix = 'fg_' + '_'.join(map(str, frame_gaps))
                if frame_gaps_suffix not in model_name:
                    model_name = f'{model_name}-{frame_gaps_suffix}'

        ds_cfg[f'{mode}_name'] = model_name
        ds_cfg[f'{mode}_file_pattern'] = os.path.join(db_root_dir, 'tfrecord', tf_name, 'shard*')

    if training:
        ds_cfg.category_names_path = os.path.join(ds_cfg.train_db_root_dir, ds_cfg.train_filename_for_metrics)
        # ds_cfg.json_dict = ds_cfg.train_json_dict
    else:
        ds_cfg.category_names_path = os.path.join(ds_cfg.eval_db_root_dir, ds_cfg.eval_filename_for_metrics)
        # ds_cfg.json_dict = ds_cfg.eval_json_dict

    ds_cfg.coco_annotations_dir_for_metrics = db_root_dir


dataset_configs = {
    'ipsc_object_detection': get_ipsc_data(),
    'ipsc_video_detection': get_ipsc_video_data(),
    'ipsc_semantic_segmentation': get_sem_seg_data(),
    'ipsc_video_segmentation': get_vid_seg_data(),
}
