import collections
import json
import os
import sys
import cv2

dproc_path = os.path.join(os.path.expanduser("~"), "ipsc/ipsc_data_processing")

sys.path.append(os.getcwd())
sys.path.append(dproc_path)

import numpy as np
import tensorflow as tf
import paramparse

import vocab
from data.scripts import tfrecord_lib
from tasks.visualization import vis_utils
from tasks import task_utils

from eval_utils import add_suffix


class Params(paramparse.CFG):

    def __init__(self):
        paramparse.CFG.__init__(self, cfg_prefix='p2s_seg_tfrecord')
        self.db_path = ''
        self.db_suffix = ''
        self.vis = 0

        self.n_proc = 0
        self.ann_ext = 'json'
        self.num_shards = 32
        self.output_dir = ''

        self.seq_id = -1
        self.seq_start_id = 0
        self.seq_end_id = -1

        self.n_rot = 3
        self.max_rot = 0
        self.min_rot = 10

        self.resize = 0
        self.start_id = 0
        self.end_id = -1

        self.patch_height = 0
        self.patch_width = 0

        self.max_stride = 0
        self.min_stride = 0

        self.enable_flip = 0

        self.max_length = 0
        self.starts_2d = 0
        self.starts_offset = 1000
        self.lengths_offset = 100


def load_seg_annotations(annotation_path):
    print(f'Reading annotations from {annotation_path}')
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

    assert 0 not in category_id_to_name_map.keys(), "class IDs must to be > 0"

    return image_info, category_id_to_name_map


def generate_annotations(
        params,
        image_infos,
        vid_infos,
):
    for image_info in image_infos:
        seq = image_info['seq']

        yield (
            params,
            image_info,
            vid_infos[seq]
        )


def create_tf_example(
        params,
        image_info,
        vid_info):
    """
    :param Params params:
    """

    image_height = image_info['height']
    image_width = image_info['width']
    filename = image_info['file_name']
    image_id = image_info['img_id']
    seq = image_info['seq']
    frame_id = int(image_info['frame_id'])
    mask_filename = image_info['mask_file_name']
    image_path = os.path.join(params.db_path, filename)
    mask_image_path = os.path.join(params.db_path, mask_filename)

    if not image_id.startswith('seq'):
        image_id = f'{seq}/{image_id}'

    feature_dict = {}

    if vid_info is not None:
        vid_reader, mask_vid_reader, vid_path, mask_vid_path, num_frames, vid_width, vid_height = vid_info
        vid_feature_dict = {
            'image/vid_path': tfrecord_lib.convert_to_feature(vid_path.encode('utf8')),
            'image/mask_vid_path': tfrecord_lib.convert_to_feature(mask_vid_path.encode('utf8')),
        }
        feature_dict.update(vid_feature_dict)

        image = task_utils.read_frame(vid_reader, frame_id - 1, vid_path)
        mask = task_utils.read_frame(mask_vid_reader, frame_id - 1, mask_vid_path)

        encoded_jpg = cv2.imencode('.jpg', image)[1].tobytes()
        # encoded_png = cv2.imencode('.png', mask)[1].tobytes()
    else:
        with tf.io.gfile.GFile(image_path, 'rb') as fid:
            encoded_jpg = fid.read()

        mask = cv2.imread(mask_image_path)

    image_feature_dict = tfrecord_lib.image_info_to_feature_dict(
        image_height, image_width, filename, image_id, encoded_jpg, 'jpg')

    feature_dict.update(image_feature_dict)

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask[mask > 0] = 1

    # mask_h, mask_w = mask.shape

    rle, rle_norm = task_utils.mask_to_rle(
        mask,
        max_length=params.max_length,
        starts_2d=params.starts_2d,
        starts_offset=params.starts_offset,
        lengths_offset=params.lengths_offset,
    )

    seg_feature_dict = {
        'image/rle': tfrecord_lib.convert_to_feature(rle, value_type='int64_list'),
        'image/mask_file_name': tfrecord_lib.convert_to_feature(mask_filename.encode('utf8')),
        'image/frame_id': tfrecord_lib.convert_to_feature(frame_id),
    }
    feature_dict.update(seg_feature_dict)

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example, 0


def main():
    params: Params = paramparse.process(Params)

    assert params.db_path, "db_path must be provided"

    assert params.end_id >= params.start_id, f"invalid end_id: {params.end_id}"

    if params.patch_width <= 0:
        params.patch_width = params.patch_height

    if params.min_stride <= 0:
        params.min_stride = params.patch_height

    if params.max_stride <= params.min_stride:
        params.max_stride = params.min_stride

    if not params.db_suffix:
        db_suffixes = []
        if params.resize:
            db_suffixes.append(f'resize_{params.resize}')

        db_suffixes += [f'{params.start_id:d}_{params.end_id:d}',
                        f'{params.patch_height:d}_{params.patch_width:d}',
                        f'{params.min_stride:d}_{params.max_stride:d}',
                        ]
        if params.n_rot > 0:
            db_suffixes.append(f'rot_{params.min_rot:d}_{params.max_rot:d}_{params.n_rot:d}')

        if params.enable_flip:
            db_suffixes.append('flip')

        params.db_suffix = '-'.join(db_suffixes)

    params.db_path = f'{params.db_path}-{params.db_suffix}'

    json_suffix = params.db_suffix

    if params.seq_id >= 0:
        params.seq_start_id = params.seq_end_id = params.seq_id

    if params.seq_start_id > 0 or params.seq_end_id >= 0:
        assert params.seq_end_id >= params.seq_start_id, "end_seq_id must to be >= start_seq_id"
        seq_suffix = f'seq_{params.seq_start_id}_{params.seq_end_id}'
        json_suffix = f'{json_suffix}-{seq_suffix}'

    output_json_fname = f'{json_suffix}.{params.ann_ext}'
    json_path = os.path.join(params.db_path, output_json_fname)

    image_infos, category_id_to_name_map, = load_seg_annotations(json_path)

    if not params.output_dir:
        params.output_dir = os.path.join(params.db_path, 'tfrecord')

    os.makedirs(params.output_dir, exist_ok=True)

    if params.max_length <= 0:
        params.max_length = params.patch_width

    vid_infos = {}

    # frame_ids = set([int(image_info['frame_id']) for image_info in image_infos])

    for image_info in image_infos:
        seq = image_info['seq']
        mask_filename = image_info['mask_file_name']
        vid_path = os.path.join(params.db_path, f'{seq}.mp4')
        mask_dir = os.path.dirname(mask_filename)
        mask_vid_path = os.path.join(params.db_path, f'{mask_dir}.mp4')

        try:
            vid_info = vid_infos[seq]
        except KeyError:
            vid_reader, vid_width, vid_height, num_frames = task_utils.load_video(vid_path, seq)
            mask_reader, mask_width, mask_height, mask_num_frames = task_utils.load_video(mask_vid_path, seq)

            assert num_frames == mask_num_frames, "num_frames mismatch"
            assert vid_width == mask_width, "vid_width mismatch"
            assert vid_height == mask_height, "vid_height mismatch"

            vid_infos[seq] = vid_reader, mask_reader, vid_path, mask_vid_path, num_frames, vid_width, vid_height
        else:
            vid_reader, mask_reader, vid_path, mask_vid_path, num_frames, vid_width, vid_height = vid_info

        frame_id = int(image_info['frame_id'])
        img_id = image_info['img_id']

        assert frame_id <= num_frames, (f"frame_id {frame_id} for image {img_id} exceeds num_frames {num_frames} for "
                                        f"seq {seq}")

    annotations_iter = generate_annotations(
        params=params,
        image_infos=image_infos,
        vid_infos=vid_infos,
    )
    output_path = os.path.join(params.output_dir, json_suffix)
    os.makedirs(output_path, exist_ok=True)

    tfrecord_pattern = os.path.join(output_path, 'shard')

    tfrecord_lib.write_tf_record_dataset(
        output_path=tfrecord_pattern,
        annotation_iterator=annotations_iter,
        process_func=create_tf_example,
        num_shards=params.num_shards,
        multiple_processes=params.n_proc,
        iter_len=len(image_infos),
    )

    print(f'output_path: {output_path}')


if __name__ == '__main__':
    main()
