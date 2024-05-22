import os
import sys
import cv2
from tqdm import tqdm


def linux_path(*args, **kwargs):
    return os.path.join(*args, **kwargs).replace(os.sep, '/')


dproc_path = linux_path(os.path.expanduser("~"), "ipsc/ipsc_data_processing")
seg_path = linux_path(os.path.expanduser("~"), "617")

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
        paramparse.CFG.__init__(self, cfg_prefix='p2s_vid_seg_tf')
        self.class_names_path = ''
        self.db_path = ''
        self.db_suffix = ''
        self.vis = 0
        self.stats_only = 0

        self.excluded_src_ids = []

        self.flat_order = 'C'
        self.time_as_class = 1

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

        self.max_length = 0
        self.starts_2d = 0
        self.starts_offset = 1000
        self.lengths_offset = 200
        self.class_offset = 100
        self.subsample = 1
        self.subsample_method = 2

        self.show = 0
        self.vid = Params.Video()

    class Video:
        def __init__(self):
            self.frame_gap = 1
            self.length = 2
            self.stride = 1
            self.sample = 0


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
        assert bkg_class == 'background', "class id 0 can be used only for background"

    return image_info


def generate_patch_vid_infos(
        params: Params,
        image_infos: list[dict],
        vid_infos: dict,
):
    from collections import defaultdict
    patch_vids = defaultdict(list)
    seq_names = list(vid_infos.keys()).sort()
    for image_info in image_infos:
        img_id = image_info['img_id']
        src_id, patch_id = img_id.split('_')
        seq_id = image_info['seq']

        image_info['src_id'] = src_id
        image_info['patch_id'] = patch_id

        patch_seq_id = f'{seq_id}_{patch_id}'
        patch_vids[patch_seq_id].append(image_info)

    all_subseq_img_infos = []
    for patch_seq_id, patch_infos in patch_vids.items():
        sorted(patch_infos, key=lambda x: int(x['frame_id']))

        n_all_files = len(patch_infos)
        subseq_start_ids = list(range(0, n_all_files, params.vid.stride))
        for subseq_id, subseq_start_id in enumerate(subseq_start_ids):
            subseq_end_id = min(subseq_start_id + (params.vid.length - 1) * params.vid.frame_gap, n_all_files - 1)
            if subseq_start_id > subseq_end_id:
                break
            subseq_img_infos = patch_infos[subseq_start_id:subseq_end_id + 1:params.vid.frame_gap]

            src_ids = tuple(image_info['src_id'] for image_info in subseq_img_infos)
            if src_ids in params.excluded_src_ids:
                print(f'Skipping excluded src_ids: {src_ids}')
                continue

            n_subseq_files = len(subseq_img_infos)

            if n_subseq_files < params.vid.length:
                print(f'skipping subseq {subseq_id + 1} - with length {n_subseq_files}')
                continue
            all_subseq_img_infos.append(subseq_img_infos)

    return all_subseq_img_infos


def generate_annotations(
        params,
        class_id_to_col,
        class_id_to_name,
        metrics,
        all_subseq_img_infos,
        vid_infos,
):
    for subseq_img_infos in all_subseq_img_infos:
        seq = subseq_img_infos[0]['seq']

        yield (
            params,
            class_id_to_col,
            class_id_to_name,
            metrics,
            subseq_img_infos,
            seq,
            vid_infos[seq]
        )


def create_tf_example(
        params: Params,
        class_id_to_col,
        class_id_to_name,
        metrics,
        subseq_img_infos,
        seq,
        vid_info):
    n_classes = len(class_id_to_col)
    vid_len = len(subseq_img_infos)
    frame_ids = []
    image_ids = []
    subseq_imgs = []
    subseq_masks = []
    subseq_masks_sub = []

    vid_reader, mask_vid_reader, vid_path, mask_vid_path, num_frames, vid_width, vid_height = vid_info
    if not params.stats_only:
        video_feature_dict = tfrecord_lib.video_seg_info_to_feature_dict(
            vid_height, vid_width, vid_path, mask_vid_path,
            vid_len, seq)

    subsample_method = params.subsample_method
    max_length = params.max_length
    n_rows, n_cols = vid_height, vid_width

    vid = None
    example = None

    multi_class = n_classes > 2

    for _id, image_info in enumerate(subseq_img_infos):

        image_height = image_info['height']
        image_width = image_info['width']

        assert image_height == vid_height, "image_info height mismatch"
        assert image_width == vid_width, "image_info width mismatch"

        filename = image_info['file_name']
        image_id = image_info['img_id']
        seq = image_info['seq']
        frame_id = int(image_info['frame_id'])
        mask_filename = image_info['mask_file_name']

        frame_ids.append(frame_id)
        image_ids.append(image_id)

        if params.subsample <= 1:
            subsample_method = 0

        # image_path = linux_path(params.db_path, filename)
        # mask_image_path = linux_path(params.db_path, mask_filename)

        if not image_id.startswith('seq'):
            image_id = f'{seq}/{image_id}'

        if not params.stats_only:
            image = task_utils.read_frame(vid_reader, frame_id - 1, vid_path)

            img_h, img_w = image.shape[:2]
            assert img_h == vid_height, "img_h mismatch"
            assert img_w == vid_width, "img_w mismatch"

            subseq_imgs.append(image)

            encoded_jpg = cv2.imencode('.jpg', image)[1].tobytes()
            video_frame_feature_dict = tfrecord_lib.video_seg_frame_info_to_feature_dict(
                _id, image_id, frame_id, filename, encoded_jpg, 'jpg')
            video_feature_dict.update(video_frame_feature_dict)

        mask = task_utils.read_frame(mask_vid_reader, frame_id - 1, mask_vid_path)

        if not multi_class:
            mask = task_utils.mask_to_binary(mask)
        mask = task_utils.mask_to_gs(mask)

        mask_h, mask_w = mask.shape
        assert mask_h == vid_height, "mask_h mismatch"
        assert mask_w == vid_width, "mask_w mismatch"

        if subsample_method == 2:
            # mask_sub = task_utils.resize_mask(mask, (n_rows_sub, n_cols_sub), n_classes, is_vis=1)
            n_rows_sub, n_cols_sub = int(n_rows / params.subsample), int(n_cols / params.subsample)
            mask_sub = task_utils.resize_mask_coord(mask, (n_rows_sub, n_cols_sub), n_classes, is_vis=1)
        else:
            mask_sub = np.copy(mask)

        # mask_sub_vis = task_utils.resize_mask(mask_sub, mask.shape, n_classes)
        # mask_sub_vis = np.stack((mask_sub_vis,) * 3, axis=2)
        # mask_vis = np.stack((mask,) * 3, axis=2)
        # file_txt = f'{image_id}'
        # frg_col = (255, 255, 255)
        # concat_img = np.concatenate((image, mask_vis, mask_sub_vis), axis=1)
        # concat_img, _, _ = vis_utils.write_text(concat_img, file_txt, 5, 5, frg_col, font_size=24)
        # cv2.imshow('concat_img', concat_img)
        # cv2.waitKey(0)

        task_utils.mask_vis_to_id(mask, n_classes=n_classes)
        task_utils.mask_vis_to_id(mask_sub, n_classes=n_classes)

        subseq_masks.append(mask)
        subseq_masks_sub.append(mask_sub)
    # return

    if not params.stats_only:
        vid = np.stack(subseq_imgs, axis=0)

    vid_mask = np.stack(subseq_masks, axis=0)
    vid_mask_sub = np.stack(subseq_masks_sub, axis=0)

    if params.time_as_class:
        tac_mask = task_utils.vid_mask_to_tac(vid_mask, n_classes)
        tac_mask_sub = task_utils.vid_mask_to_tac(vid_mask_sub, n_classes)

        vid_mask_rec = task_utils.vid_mask_from_tac(tac_mask, vid_len, n_classes)
        vid_mask_sub_rec = task_utils.vid_mask_from_tac(tac_mask_sub, vid_len, n_classes)

        assert np.array_equal(vid_mask, vid_mask_rec), "vid_mask_rec mismatch"
        assert np.array_equal(vid_mask_sub, vid_mask_sub_rec), "vid_mask_rec mismatch"

        n_rle_classes = int(n_classes ** vid_len)
        rle_id_to_col, rle_id_to_name = task_utils.time_as_class_info(vid_len, class_id_to_name)

        tac_mask_rgb = task_utils.mask_id_to_vis_rgb(tac_mask, rle_id_to_col)
        tac_mask_sub_rgb = task_utils.mask_id_to_vis_rgb(tac_mask_sub, rle_id_to_col)
        tac_mask_sub_rgb = task_utils.resize_mask(tac_mask_sub_rgb, tac_mask_rgb.shape, n_classes)
        tac_mask_cat = np.concatenate((tac_mask_rgb, tac_mask_sub_rgb), axis=1)

        font_size = 24
        text_x = text_y = 5
        for rle_id, rle_col in rle_id_to_col.items():
            if rle_id == 0:
                continue
            rle_name = rle_id_to_name[rle_id]
            # try:
            #     rle_col_name = task_utils.bgr_col[rle_col]
            # except KeyError:
            #     rle_col_name = rle_col
            # text_y = int((rle_id - 1) * font_size + 5)

            tac_mask_cat, text_x, text_y = vis_utils.write_text(
                tac_mask_cat, f'{rle_name} ',text_x, text_y,
                rle_col,
                wait=100, bb=0, show=0, font_size=font_size)
        cv2.imshow('tac_masks', tac_mask_cat)
        cv2.waitKey(10)

        vid_mask = tac_mask
        vid_mask_sub = tac_mask_sub
        # return
    else:
        n_rle_classes = n_classes
        rle_id_to_col, rle_id_to_name = class_id_to_col, class_id_to_name

    if subsample_method == 2:
        max_length_sub = int(max_length / params.subsample)
        n_rows_sub, n_cols_sub = int(n_rows / params.subsample), int(n_cols / params.subsample)
    else:
        n_rows_sub, n_cols_sub = n_rows, n_cols
        max_length_sub = max_length

    starts, lengths = task_utils.mask_to_rle(
        mask=vid_mask_sub,
        max_length=max_length_sub,
        n_classes=n_rle_classes,
        order=params.flat_order,
    )
    rle_cmp = [starts, lengths]

    n_runs = len(starts)

    multi_class = False
    class_ids = None
    if n_rle_classes > 2:
        assert params.class_offset > 0, "class_offset must be > 0"
        class_ids = task_utils.get_rle_class_ids(vid_mask_sub, starts, lengths, rle_id_to_col)
        rle_cmp.append(class_ids)
        multi_class = True

    if params.vis:
        task_utils.vis_video_rle(
            starts, lengths, class_ids,
            class_id_to_col, class_id_to_name,
            image_ids,
            vid, vid_mask, vid_mask_sub,
            params.time_as_class,
            params.flat_order,
            rle_id_to_name,
            rle_id_to_col,
        )

    rle_tokens = task_utils.rle_to_tokens(
        rle_cmp, vid_mask_sub.shape,
        params.starts_offset,
        params.lengths_offset,
        params.class_offset,
        params.starts_2d,
    )
    rle_len = len(rle_tokens)

    if multi_class:
        assert rle_len % 3 == 0, "rle_len must be divisible by 3"
    else:
        assert rle_len % 2 == 0, "rle_len must be divisible by 2"

    if not params.stats_only:
        seg_feature_dict = {
            'image/rle': tfrecord_lib.convert_to_feature(rle_tokens, value_type='int64_list'),
        }
        video_feature_dict.update(seg_feature_dict)
        example = tf.train.Example(features=tf.train.Features(feature=video_feature_dict))

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

    if params.stats_only:
        print('running in stats only mode')
        # params.vis = params.show = False

    class_names, class_id_to_col, class_id_to_name = task_utils.read_class_info(params.class_names_path)[:3]

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
    json_path = linux_path(params.db_path, output_json_fname)

    out_name = json_suffix
    if params.subsample > 1:
        out_name = f'{out_name}-sub_{params.subsample}'

    vid_suffixes = []
    if params.vid.length:
        vid_suffixes.append(f'len_{params.vid.length}')

    if params.vid.stride:
        vid_suffixes.append(f'strd_{params.vid.stride}')

    if params.vid.sample:
        vid_suffixes.append(f'smp_{params.vid.sample}')

    if params.vid.frame_gap:
        vid_suffixes.append(f'fg_{params.vid.frame_gap}')

    if vid_suffixes:
        vid_suffix = '-'.join(vid_suffixes)
        out_name = f'{out_name}-{vid_suffix}'

    if multi_class:
        out_name = f'{out_name}-mc'

    image_infos = load_seg_annotations(json_path)

    if not params.output_dir:
        params.output_dir = linux_path(params.db_path, 'tfrecord')

    os.makedirs(params.output_dir, exist_ok=True)

    output_path = linux_path(params.output_dir, out_name)
    os.makedirs(output_path, exist_ok=True)

    print(f'output_path: {output_path}')

    if params.max_length <= 0:
        params.max_length = params.patch_width * params.vid.length

    vid_infos = {}

    # frame_ids = set([int(image_info['frame_id']) for image_info in image_infos])

    for image_info in image_infos:
        seq = image_info['seq']
        mask_filename = image_info['mask_file_name']
        vid_path = linux_path(params.db_path, f'{seq}.mp4')
        mask_dir = os.path.dirname(mask_filename)
        mask_vid_path = linux_path(params.db_path, f'{mask_dir}.mp4')

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

    all_subseq_img_infos = generate_patch_vid_infos(
        params,
        image_infos,
        vid_infos,
    )

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
        all_subseq_img_infos=all_subseq_img_infos,
        vid_infos=vid_infos,
    )
    # if params.stats_only:

    for idx, annotations_iter_ in tqdm(enumerate(annotations_iter), total=len(image_infos)):
        create_tf_example(*annotations_iter_)
    # else:
    #     tfrecord_pattern = linux_path(output_path, 'shard')
    #     tfrecord_lib.write_tf_record_dataset(
    #         output_path=tfrecord_pattern,
    #         annotation_iterator=annotations_iter,
    #         process_func=create_tf_example,
    #         num_shards=params.num_shards,
    #         multiple_processes=params.n_proc,
    #         iter_len=len(image_infos),
    #     )
    print(f'output_path: {output_path}')

    for method, metrics_ in metrics.items():
        for metric_, val in metrics_.items():
            metrics_path = linux_path(output_path, f'{method}_{metric_}')
            with open(metrics_path, 'w') as f:
                f.write('\n'.join(map(str, val)))


if __name__ == '__main__':
    main()
