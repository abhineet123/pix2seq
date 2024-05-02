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
        paramparse.CFG.__init__(self, cfg_prefix='p2s_tfrecord')
        self.db_path = ''
        self.db_suffix = ''
        self.vis = 0

        self.n_proc = 0
        self.ann_ext = 'json'
        self.num_shards = 32
        self.output_dir = ''
        self.xml_output_file = ''

        self.start_seq_id = 0
        self.end_seq_id = -1

        self.start_frame_id = 0
        self.end_frame_id = -1
        self.frame_stride = 1

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


def load_seg_annotations(annotation_path):
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

    assert 0 not in category_id_to_name_map.keys(), "class IDs must to be > 0"

    return image_info, category_id_to_name_map


def generate_annotations(image_infos,
                         category_id_to_name_map,
                         ):
    for image_info in image_infos:
        yield (
            image_info,
            category_id_to_name_map,
        )


def read_frame(vid_reader, frame_id, vid_path):
    vid_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
    assert vid_reader.get(cv2.CAP_PROP_POS_FRAMES) == frame_id - 1, "Failed to set frame index in video"
    ret, image = vid_reader.read()
    if not ret:
        raise AssertionError(f'Frame {frame_id} could not be read from {vid_path}')
    return image


def load_video(vid_path, seq):
    vid_reader = cv2.VideoCapture()
    if not vid_reader.open(vid_path):
        raise AssertionError(f'Video file could not be opened: {vid_path}')

    num_frames = int(vid_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_height = int(vid_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_width = int(vid_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f'\n{seq}: loaded {vid_width}x{vid_height} video with {num_frames} frames from {vid_path}')

    return vid_reader


def create_tf_example(
        image_info,
        db_path,
        category_id_to_name_map,
        vid_readers
):
    image_height = image_info['height']
    image_width = image_info['width']
    filename = image_info['file_name']
    image_id = image_info['id']
    seq = image_info['seq']
    frame_id = image_info['frame_id']
    mask_filename = image_info['mask_file_name']
    image_path = os.path.join(db_path, filename)
    mask_image_path = os.path.join(db_path, mask_filename)

    vid_path = os.path.join(db_path, f'{seq}.mp4')
    mask_dir = os.path.dirname(mask_filename)
    mask_vid_path = os.path.join(db_path, seq, f'{mask_dir}.mp4')

    if vid_readers is not None:
        try:
            vid_reader, mask_vid_reader = vid_readers[seq]
        except KeyError:
            vid_reader = load_video(vid_path, seq)
            mask_vid_reader = load_video(mask_vid_path, seq)
            vid_readers[seq] = vid_reader, mask_vid_reader

        image = read_frame(vid_reader, frame_id, vid_path)
        mask = read_frame(mask_vid_reader, frame_id, mask_vid_path)

        encoded_jpg = cv2.imencode('.jpg', image)[1].tobytes()
        # encoded_png = cv2.imencode('.png', mask)[1].tobytes()
    else:
        with tf.io.gfile.GFile(image_path, 'rb') as fid:
            encoded_jpg = fid.read()

        mask = cv2.imread(mask_image_path)

    feature_dict = tfrecord_lib.image_info_to_feature_dict(
        image_height, image_width, filename, image_id, encoded_jpg, 'jpg')

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask[mask > 0] = 1

    mask_h, mask_w = mask.shape

    rle, rle_norm = task_utils.mask_to_rle(
        mask, max_length=mask_w, start_2d=False)

    seg_feature_dict = {
        'image/rle': tfrecord_lib.convert_to_feature(rle_norm,
                                                     value_type='float_list'),
    }
    feature_dict.update(seg_feature_dict)

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example, 0


def main():
    params: Params = paramparse.process(Params)

    assert params.db_path, "db_path must be provided"

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

        params.db_suffix = '_'.join(db_suffixes)

    if params.db_suffix:
        params.db_path = f'{params.db_path}_{params.db_suffix}'

    json_suffix = params.db_suffix
    output_json_fname = f'{json_suffix}.{params.ann_ext}'
    json_path = os.path.join(params.db_path, output_json_fname)

    image_info, category_id_to_name_map, = load_seg_annotations(json_path)

    if not params.output_dir:
        params.output_dir = os.path.join(params.db_path, 'tfrecord')

    os.makedirs(params.output_dir, exist_ok=True)

    annotations_iter = generate_annotations(
        image_infos=image_info,
        category_id_to_name_map=category_id_to_name_map,
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
        iter_len=len(image_info),
    )

    print(f'output_path: {output_path}')


if __name__ == '__main__':
    main()
