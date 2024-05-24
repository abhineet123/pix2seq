import collections
import json
import os
import sys
import cv2
from tqdm import tqdm

dproc_path = os.path.join(os.path.expanduser("~"), "ipsc/ipsc_data_processing")
seg_path = os.path.join(os.path.expanduser("~"), "617")

sys.path.append(os.getcwd())
sys.path.append(dproc_path)
sys.path.append(seg_path)

import numpy as np
import tensorflow as tf
import paramparse

import vocab
from data.scripts import tfrecord_lib
from tasks.visualization import vis_utils
from tasks import task_utils

from eval_utils import add_suffix


class Params(paramparse.CFG):
    """
    :ivar subsample_method:
    1: create RLE of full-res mask and sample the starts and lengths thus generated
    2: decrease mask resolution by resizing and create RLE of the low-res mask
    """

    def __init__(self):
        paramparse.CFG.__init__(self, cfg_prefix='p2s_seg_tf')
        self.class_names_path = ''
        self.db_path = ''
        self.db_suffix = ''
        self.vis = 0
        self.stats_only = 0
        self.check = 1

        self.flat_order = 'C'

        self.n_proc = 0
        self.ann_ext = 'json'
        self.num_shards = 32
        self.output_dir = ''

        self.seq_id = -1
        self.seq_start_id = 0
        self.seq_end_id = -1

        self.n_rot = 0
        self.max_rot = 0
        self.min_rot = 0

        self.resize = 0
        self.start_id = 0
        self.end_id = -1

        self.patch_height = 0
        self.patch_width = 0

        self.max_stride = 0
        self.min_stride = 0

        self.enable_flip = 0

        self.length_as_class = 0

        self.max_length = 0
        self.starts_2d = 0
        self.starts_offset = 1000
        self.lengths_offset = 100
        self.class_offset = 0
        self.subsample = 1
        self.subsample_method = 2

        self.show = 0


def append_metrics(metrics, out):
    for metric, val in metrics.items():
        try:
            out[metric].append(val)
        except KeyError:
            out[metric] = [val, ]

    # vis_txt = ' '.join(f'{metric}: {val:.2f}' if isinstance(val, float)
    #                    else f'{metric}: {val}'
    #                    for metric, val in metrics.items())
    # return vis_txt


def eval_mask(pred_mask, gt_mask, rle_len):
    import densenet.evaluation.eval_segm as eval_segm
    class_ids = [0, 1]
    pix_acc = eval_segm.pixel_accuracy(pred_mask, gt_mask, class_ids)
    _acc, mean_acc = eval_segm.mean_accuracy(pred_mask, gt_mask, class_ids, return_acc=1)
    _IU, mean_IU = eval_segm.mean_IU(pred_mask, gt_mask, class_ids, return_iu=1)
    fw_IU = eval_segm.frequency_weighted_IU(pred_mask, gt_mask, class_ids)
    return dict(
        rle_len=rle_len,
        pix_acc=pix_acc,
        mean_acc=mean_acc,
        mean_IU=mean_IU,
        fw_IU=fw_IU,
    )


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

    class_id_to_name = dict(
        (element['id'], element['name']) for element in annotations['categories'])

    try:
        bkg_class = class_id_to_name[0]
    except KeyError:
        pass
    else:
        assert bkg_class == 'background', "class id 0 must be used only for background"

    return image_info


def generate_annotations(
        params,
        class_id_to_col,
        class_id_to_name,
        metrics,
        image_infos,
        vid_infos,
):
    for image_info in image_infos:
        seq = image_info['seq']

        yield (
            params,
            class_id_to_col,
            class_id_to_name,
            metrics,
            image_info,
            vid_infos[seq]
        )


def create_tf_example(
        params: Params,
        class_id_to_col: dict,
        class_id_to_name: dict,
        metrics: dict,
        image_info: dict,
        vid_info: dict
):
    n_classes = len(class_id_to_col)
    multi_class = n_classes > 2

    image_height = image_info['height']
    image_width = image_info['width']
    filename = image_info['file_name']
    image_id = image_info['img_id']
    seq = image_info['seq']
    frame_id = int(image_info['frame_id'])
    mask_filename = image_info['mask_file_name']

    subsample_method = params.subsample_method
    if params.subsample <= 1:
        subsample_method = 0

    image_path = os.path.join(params.db_path, filename)
    mask_image_path = os.path.join(params.db_path, mask_filename)

    if not image_id.startswith('seq'):
        image_id = f'{seq}/{image_id}'

    feature_dict = {}

    image = encoded_jpg = None

    if vid_info is not None:
        vid_reader, mask_vid_reader, vid_path, mask_vid_path, num_frames, vid_width, vid_height = vid_info
        vid_feature_dict = {
            'image/vid_path': tfrecord_lib.convert_to_feature(vid_path.encode('utf8')),
            'image/mask_vid_path': tfrecord_lib.convert_to_feature(mask_vid_path.encode('utf8')),
        }
        feature_dict.update(vid_feature_dict)

        if not params.stats_only:
            image = task_utils.read_frame(vid_reader, frame_id - 1, vid_path)
            # from PIL import Image
            # from io import BytesIO
            # buffer = BytesIO()
            # Image.fromarray(image).save(buffer, format="JPEG")
            # encoded_jpg = buffer.getvalue()

            encoded_jpg = cv2.imencode('.jpg', image)[1].tobytes()
            # encoded_png = cv2.imencode('.png', mask)[1].tobytes()

        mask = task_utils.read_frame(mask_vid_reader, frame_id - 1, mask_vid_path)
    else:
        if not params.stats_only:
            with tf.io.gfile.GFile(image_path, 'rb') as fid:
                encoded_jpg = fid.read()

            image = cv2.imread(image_path)
        mask = cv2.imread(mask_image_path)

    vis_imgs = []

    if not params.stats_only:
        vis_imgs.append(image)
        image_feature_dict = tfrecord_lib.image_info_to_feature_dict(
            image_height, image_width, filename, image_id, encoded_jpg, 'jpg')

        feature_dict.update(image_feature_dict)

    if not multi_class:
        mask = task_utils.mask_to_binary(mask)
    mask = task_utils.mask_to_gs(mask)

    n_rows, n_cols = mask.shape
    max_length = params.max_length

    if subsample_method == 2:
        """        
        decrease mask resolution by resizing and create RLE of the low-res mask
        """
        max_length_sub = int(max_length / params.subsample)
        n_rows_sub, n_cols_sub = int(n_rows / params.subsample), int(n_cols / params.subsample)

        # mask_sub = task_utils.resize_mask(mask, (n_rows_sub, n_cols_sub), n_classes, is_vis=1)
        mask_sub = task_utils.resize_mask_coord(mask, (n_rows_sub, n_cols_sub), n_classes, is_vis=1)
    else:
        mask_sub = np.copy(mask)
        n_rows_sub, n_cols_sub = n_rows, n_cols
        max_length_sub = max_length

    # mask_sub_vis = task_utils.resize_mask(mask_sub, mask.shape, n_classes)
    # mask_sub_vis = np.stack((mask_sub_vis,) * 3, axis=2)
    # mask_vis = np.stack((mask,) * 3, axis=2)
    # file_txt = f'{image_id}'
    # frg_col = (255, 255, 255)
    # concat_img = np.concatenate((image, mask_vis, mask_sub_vis), axis=1)
    # concat_img, _, _ = vis_utils.write_text(concat_img, file_txt, 5, 5, frg_col, font_size=24)
    # cv2.imshow('concat_img', concat_img)
    # cv2.waitKey(0)
    #
    # return

    task_utils.mask_vis_to_id(mask, n_classes=n_classes)
    task_utils.mask_vis_to_id(mask_sub, n_classes=n_classes)

    starts, lengths = task_utils.mask_to_rle(
        mask=mask_sub,
        max_length=max_length_sub,
        n_classes=n_classes,
        order=params.flat_order,
    )

    # starts_bin, lengths_bin = task_utils.mask_to_rle(
    #     mask=mask_sub,
    #     max_length=max_length_sub,
    #     binary=True,
    # )
    # mismatch = False
    # if not np.array_equal(starts, starts_bin):
    #     print('starts mismatch')
    #     mismatch = True
    #
    # if not np.array_equal(lengths, lengths_bin):
    #     print('lengths mismatch')
    #     mismatch = True
    # if mismatch:
    #     mask_bin_vis = (mask > 0).astype(np.uint8) * 255
    #     mask_bin_vis = task_utils.resize_mask(mask_bin_vis, (640, 640), n_classes, is_vis=True)
    #
    #     mask_vis = task_utils.mask_id_to_vis(mask, n_classes, copy=True)
    #     mask_vis = task_utils.resize_mask(mask_vis, (640, 640), n_classes, is_vis=True)
    #
    #     masks = np.concatenate((mask_bin_vis, mask_vis), axis=1)
    #     cv2.imshow('masks', masks)
    #     cv2.waitKey(0)

    if subsample_method == 1:
        """subsample RLE of high-res mask"""
        starts, lengths = task_utils.subsample_rle(
            starts, lengths,
            subsample=params.subsample,
            shape=(n_rows, n_cols),
            max_length=max_length,
            flat_order=params.flat_order,
        )

    rle_cmp = [starts, lengths]

    n_runs = len(starts)

    class_ids = None
    if multi_class:
        assert params.class_offset > 0, "class_offset must be > 0"
        class_ids = task_utils.get_rle_class_ids(mask_sub, starts, lengths, class_id_to_col, order=params.flat_order)
        rle_cmp.append(class_ids)

    if params.vis and n_runs > 0:
        task_utils.vis_rle(
            starts, lengths, class_ids,
            class_id_to_col, class_id_to_name,
            image, mask, mask_sub,
            flat_order=params.flat_order)

    rle_tokens = task_utils.rle_to_tokens(
        rle_cmp, mask_sub.shape,
        params.length_as_class,
        params.starts_offset,
        params.lengths_offset,
        params.class_offset,
        params.starts_2d,
        params.flat_order,
    )
    rle_len = len(rle_tokens)

    if multi_class:
        assert rle_len % 3 == 0, "rle_len must be divisible by 3"
    else:
        assert rle_len % 2 == 0, "rle_len must be divisible by 2"

    if params.check:
        task_utils.check_rle_tokens(
            image, mask, rle_tokens, n_classes,
            params.starts_offset,
            params.lengths_offset,
            params.class_offset,
            max_length,
            params.subsample,
            multi_class,
            class_id_to_col,
            is_vis=True)

    example = None
    if not params.stats_only:
        seg_feature_dict = {
            'image/rle': tfrecord_lib.convert_to_feature(rle_tokens, value_type='int64_list'),
            'image/mask_file_name': tfrecord_lib.convert_to_feature(mask_filename.encode('utf8')),
            'image/frame_id': tfrecord_lib.convert_to_feature(frame_id),
        }
        feature_dict.update(seg_feature_dict)
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    if rle_len > 0:
        metrics_ = dict(rle_len=rle_len)
        append_metrics(metrics_, metrics[f'method_{subsample_method}'])

    if params.show and n_runs > 0:
        rle_rec_cmp = task_utils.rle_from_tokens(
            rle_tokens, mask_sub.shape,
            params.starts_offset,
            params.lengths_offset,
            params.class_offset,
            params.starts_2d,
            multi_class
        )
        starts_rec, lengths_rec = rle_rec_cmp[:2]
        if multi_class:
            class_ids_rec = rle_rec_cmp[2]
        else:
            class_ids_rec = [1, ] * len(starts_rec)

        if subsample_method == 1:
            """reconstruct full-res mask by super sampling / scaling up the starts and lengths"""
            starts_rec, lengths_rec = task_utils.supersample_rle(
                starts_rec, lengths_rec,
                subsample=params.subsample,
                shape=(n_rows, n_cols),
                max_length=max_length,
                flat_order=params.flat_order,
            )

        mask_rec = task_utils.rle_to_mask(
            starts_rec, lengths_rec, class_ids_rec,
            (n_rows_sub, n_cols_sub),
        )

        mask_vis = task_utils.mask_id_to_vis_rgb(mask_sub, class_id_to_col)
        mask_rec_vis = task_utils.mask_id_to_vis_rgb(mask_rec, class_id_to_col)

        if subsample_method == 2:
            """reconstruct low-res mask and resize to scale it up"""
            mask_vis = cv2.resize(mask_vis, (n_cols, n_rows))

            mask_rec_vis = cv2.resize(mask_rec_vis, (n_cols, n_rows))
            # metrics_ = eval_mask(mask_rec, mask, rle_len)
            # vis_txt.append(append_metrics(metrics_, metrics['method_1']))

        vis_imgs.append(mask_vis)
        vis_imgs.append(mask_rec_vis)

        import eval_utils

        vis_imgs = np.concatenate(vis_imgs, axis=1)
        # vis_txt = ' '.join(vis_txt)
        vis_imgs = eval_utils.annotate(vis_imgs, f'{image_id}')
        # cv2.imshow('mask_vis', mask_vis)
        # cv2.imshow('mask_rec_vis', mask_rec_vis)
        cv2.imshow('vis_imgs', vis_imgs)
        k = cv2.waitKey(0)
        if k == 27:
            exit()

    if not params.stats_only:
        return example, 0


def main():
    params: Params = paramparse.process(Params)

    assert params.db_path, "db_path must be provided"

    assert params.end_id >= params.start_id, f"invalid end_id: {params.end_id}"

    class_names, class_id_to_col, class_id_to_name = task_utils.read_class_info(params.class_names_path)[:3]

    if params.stats_only:
        print('running in stats only mode')
        params.vis = params.show = False

    n_classes = len(class_id_to_col)
    multi_class = False
    if n_classes > 2:
        assert params.class_offset > 0, "class_offset must be > 0 for multi_class mode"
        multi_class = True

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

    out_name = json_suffix
    if params.subsample > 1:
        out_name = f'{out_name}-sub_{params.subsample}'

    if multi_class:
        out_name = f'{out_name}-mc'

    image_infos = load_seg_annotations(json_path)

    if not params.output_dir:
        params.output_dir = os.path.join(params.db_path, 'tfrecord')

    os.makedirs(params.output_dir, exist_ok=True)

    output_path = os.path.join(params.output_dir, out_name)
    os.makedirs(output_path, exist_ok=True)

    print(f'output_path: {output_path}')

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
            vid_reader, vid_width, vid_height, num_frames = task_utils.load_video(vid_path)
            mask_reader, mask_width, mask_height, mask_num_frames = task_utils.load_video(mask_vid_path)

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

    metrics = dict(
        method_0={},
        method_1={},
        method_2={},

    )
    annotations_iter = generate_annotations(
        params=params,
        class_id_to_col=class_id_to_col,
        class_id_to_name=class_id_to_name,
        metrics=metrics,
        image_infos=image_infos,
        vid_infos=vid_infos,
    )

    # for idx, annotations_iter_ in tqdm(enumerate(annotations_iter), total=len(image_infos)):
    #     create_tf_example(*annotations_iter_)

    if params.stats_only:
        for idx, annotations_iter_ in tqdm(enumerate(annotations_iter), total=len(image_infos)):
            create_tf_example(*annotations_iter_)
    else:
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

    for method, metrics_ in metrics.items():
        for metric_, val in metrics_.items():
            metrics_path = os.path.join(output_path, f'{method}_{metric_}')
            with open(metrics_path, 'w') as f:
                f.write('\n'.join(map(str, val)))


if __name__ == '__main__':
    main()
