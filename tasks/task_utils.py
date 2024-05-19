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
"""Common task utils."""

import json
from typing import Optional, Any, Dict

import cv2

import utils
import vocab
import tensorflow as tf

import numpy as np

from tasks.visualization import vis_utils
from eval_utils import col_bgr


def split_runs(run_ids, starts, lengths, max_length):
    """divide over-long runs into segments"""
    new_starts_ = []
    new_lengths_ = []
    # starts_, lengths_ = list(starts), list(lengths)
    for _id in run_ids:
        start, length = starts[_id], lengths[_id]

        new_starts_.append(start)
        new_lengths_.append(max_length)

        residual_length = length - max_length
        start_ = start
        length_ = max_length
        while True:
            start_ += length_
            new_starts_.append(start_)

            length_ = min(residual_length, max_length)
            new_lengths_.append(length_)

            residual_length -= length_
            if residual_length <= 0:
                break

        # if _id < len(lengths) - 1:
        #     cmb = np.stack((starts, lengths), axis=1)
        #     cmb_new = np.stack((new_starts_, new_lengths_), axis=1)
        #     print()

    valid_starts = [v for i, v in enumerate(starts) if i not in run_ids]
    valid_lengths = [v for i, v in enumerate(lengths) if i not in run_ids]

    valid_starts += new_starts_
    valid_lengths += new_lengths_

    starts, lengths = np.asarray(valid_starts), np.asarray(valid_lengths)

    sort_idx = np.argsort(starts)
    starts = starts[sort_idx]
    lengths = lengths[sort_idx]

    return starts, lengths


def read_frame(vid_reader, frame_id, vid_path):
    vid_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    assert vid_reader.get(cv2.CAP_PROP_POS_FRAMES) == frame_id, "Failed to set frame index in video"
    ret, image = vid_reader.read()
    if not ret:
        raise AssertionError(f'Frame {frame_id} could not be read from {vid_path}')
    return image


def load_video(vid_path, seq=''):
    vid_reader = cv2.VideoCapture()
    if not vid_reader.open(vid_path):
        raise AssertionError(f'Video file could not be opened: {vid_path}')

    num_frames = int(vid_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_height = int(vid_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_width = int(vid_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    if seq:
        print(f'\n{seq}: loaded {vid_width}x{vid_height} video with {num_frames} frames from {vid_path}')

    return vid_reader, vid_width, vid_height, num_frames


def check_rle(
        image, mask, rle, n_classes,
        starts_offset, lengths_offset, class_offset,
        max_length, subsample, multi_class,
        class_to_col, show):
    if len(mask.shape) == 3:
        mask_gt = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gt = np.copy(mask)

    mask_vis_to_id(mask_gt, n_classes)

    n_rows, n_cols = mask_gt.shape

    if subsample > 1:
        max_length = int(max_length / subsample)
        n_rows, n_cols = int(n_rows / subsample), int(n_cols / subsample)

    rle_len = len(rle)

    mask_rec, rle_rec_cmp = mask_from_tokens(
        rle,
        (n_rows, n_cols),
        starts_offset=starts_offset,
        lengths_offset=lengths_offset,
        class_offset=class_offset,
        starts_2d=False,
        multi_class=multi_class,

    )
    if subsample > 1:
        mask_gt_sub = resize_mask_coord(mask_gt, mask_rec.shape, n_classes, is_vis=0)
        # mask_rec_ = mask_id_to_vis(mask_rec, n_classes, copy=True)
        # mask_gt_ = mask_id_to_vis(mask_gt, n_classes, copy=True)
        # cv2.imshow('mask_rec', mask_rec_)
        # cv2.imshow('mask_gt', mask_gt_)
    else:
        mask_gt_sub = mask_gt

    if show and rle_len > 0:
        # if subsample > 1:
        #     mask_rec = resize_mask(mask_rec, mask.shape, n_classes, is_vis=1)

        # import eval_utils
        mask_gt_vis = mask_id_to_vis_rgb(mask_gt, class_to_col)
        mask_rec_vis = mask_id_to_vis_rgb(mask_rec, class_to_col)

        cv2.imshow('mask_rec_vis', mask_rec_vis)

        mask_gt_vis = resize_mask(mask_gt_vis, image.shape, n_classes, is_vis=1)
        mask_rec_vis = resize_mask(mask_rec_vis, image.shape, n_classes, is_vis=1)

        masks_all = np.concatenate([image, mask_gt_vis, mask_rec_vis], axis=1)
        # vis_txt = ' '.join(vis_txt)
        # masks_all = eval_utils.annotate(masks_all, vis_txt)
        # cv2.imshow('mask_gt_vis', mask_gt_vis)
        # cv2.imshow('mask_rec_vis', mask_rec_vis)
        cv2.imshow('masks_all', masks_all)
        k = cv2.waitKey(100)
        if k == 27:
            exit()

    mask_mismatch = np.nonzero(mask_gt_sub != mask_rec)
    assert mask_mismatch[0].size == 0, "mask_rec mismatch"
    print('masks match !')


def interleave_rle(rle_cmps):
    rle = [int(item) for sublist in zip(*rle_cmps) for item in sublist]
    return rle


def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def resize_mask_coord(mask, shape, n_classes, is_vis=1):
    if is_vis:
        mask = np.copy(mask)
        mask_vis_to_id(mask, n_classes)

    mask_out = np.zeros_like(mask, shape=shape)
    mask_rows, mask_cols = mask.shape[:2]
    out_rows, out_cols = mask_out.shape[:2]

    res_x, res_y = float(out_cols) / mask_cols, float(out_rows) / mask_rows

    for class_id in range(1, n_classes):
        y, x = np.nonzero(mask == class_id)
        out_x, out_y = (x * res_x).astype(np.int64), (y * res_y).astype(np.int64)
        mask_out[out_y, out_x] = class_id
    if is_vis:
        mask_id_to_vis(mask_out, n_classes)
    return mask_out


def resize_mask(mask, shape, n_classes, is_vis=1):
    n_rows, n_cols = shape[:2]
    if not is_vis:
        mask = mask_id_to_vis(mask, n_classes, copy=True)
    mask = cv2.resize(mask, (n_cols, n_rows))
    if not is_vis:
        mask_vis_to_id(mask, n_classes)
    return mask


def mask_id_to_vis(mask, n_classes, to_rgb=0, copy=False):
    if to_rgb or copy:
        mask = np.copy(mask)
    if n_classes == 3:
        # labels_img[labels_img == 0] = 0
        mask[mask == 1] = 128
        mask[mask == 2] = 255
    elif n_classes == 2:
        # labels_img[labels_img == 0] = 0
        mask[mask == 1] = 255
    else:
        raise AssertionError('unsupported number of classes: {}'.format(n_classes))
    if to_rgb and len(mask.shape) == 2:
        mask = np.stack((mask,) * 3, axis=2)
    if to_rgb or copy:
        return mask


def mask_vis_to_id(mask, n_classes):
    if n_classes == 3:
        mask[mask < 64] = 0
        mask[np.logical_and(mask >= 64, mask < 192)] = 1
        mask[mask >= 192] = 2
    elif n_classes == 2:
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
    else:
        raise AssertionError('unsupported number of classes: {}'.format(n_classes))


def blend_mask(mask, image, class_to_col):
    n_classes = len(class_to_col)
    """ignore class id 0 for background"""
    vis_image = np.copy(image)

    for class_id in range(1, n_classes):
        class_col = class_to_col[class_id]
        class_col = col_bgr[class_col]
        class_mask_binary = (mask == class_id)
        vis_image[class_mask_binary] = vis_image[class_mask_binary] * 0.5 + np.asarray(class_col) * 0.5
    return vis_image


def mask_id_to_vis_rgb(mask, class_to_col):
    mask_rgb = np.stack((mask,) * 3, axis=2)

    n_classes = len(class_to_col)
    for class_id in range(n_classes):
        class_col = class_to_col[class_id]
        class_col = col_bgr[class_col]
        mask_rgb[mask == class_id] = class_col
    return mask_rgb


def get_cols(n_runs):
    min_col, max_col = 100, 200
    n_col_levels = int(n_runs ** (1. / 3) + 1)
    col_range = max_col - min_col
    assert n_col_levels <= col_range, "n_col_levels exceeds col_range"
    col_levels = [int(x) for x in np.linspace(
        min_col, max_col,
        n_col_levels, dtype=int)]
    import itertools
    cols = list(itertools.product(col_levels, repeat=3))

    return cols


def vis_rle(starts, lengths, class_ids, class_id_to_col, class_id_to_name, image, mask, mask_sub):
    n_runs = len(starts)
    n_classes = len(class_id_to_col)
    # cols = get_cols(n_runs)

    mask_rgb = mask_id_to_vis_rgb(mask, class_id_to_col)
    mask_sub_rgb = mask_id_to_vis_rgb(mask_sub, class_id_to_col)

    mask_sub_vis = mask_id_to_vis(mask_sub, n_classes=n_classes, to_rgb=1, copy=True)
    mask_sub_vis[mask_sub_vis > 0] = 255

    vis_image = blend_mask(mask, image, class_id_to_col)

    text_x = text_y = 5
    # col = (0, 255, 0)
    bkg_col = (0, 0, 0)
    frg_col = (255, 255, 255)
    # mask_col = (0, 255, 0)
    vis_size = 640
    font_size = 16
    n_rows, n_cols = mask_sub.shape
    resize_y, resize_x = float(vis_size) / n_rows, float(vis_size) / n_cols

    mask_sub_vis_ = cv2.resize(mask_sub_vis, (vis_size, vis_size))
    mask_sub_rgb_ = cv2.resize(mask_sub_rgb, (vis_size, vis_size))
    mask_rgb_ = cv2.resize(mask_rgb, (vis_size, vis_size))
    cv2.imshow('mask_sub_vis_', mask_sub_vis_)
    cv2.imshow('mask_sub_rgb', mask_sub_rgb_)
    cv2.imshow('mask_rgb', mask_rgb_)

    # cv2.waitKey(0)
    return

    text_img = np.full((vis_size, vis_size, 3), bkg_col, dtype=np.uint8)
    mask_flat = mask_sub.flatten()
    _pause = 1

    for run_id, (start, length) in enumerate(zip(starts, lengths)):
        mask_bool_flat = np.zeros_like(mask_flat, dtype=bool)
        mask_bool_flat[start:start + length] = True
        mask_bool = np.reshape(mask_bool_flat, (n_rows, n_cols))
        # run_y, run_x = np.unravel_index([start, start + length], (n_rows, n_cols))
        # mask_sub_rgb[run_y[0]:run_y[1], run_x[0]:run_x[1], :] = col

        run_txt = f'{int(start)}, {int(length)}, '
        if class_ids is not None:
            class_id = class_ids[run_id]
            class_name = class_id_to_name[class_id]
            run_txt = f'{run_txt}{class_name}, '
        else:
            class_id = 1

        col = col_bgr[class_id_to_col[class_id]]

        # col_id = run_id % len(cols)
        # col = cols[col_id]
        mask_sub_vis[mask_bool] = col

        text_img, text_x, text_y, text_bb = vis_utils.write_text(text_img, run_txt, text_x, text_y, col,
                                                                 wait=100, bb=1, show=0, font_size=font_size)
        if run_id == n_runs - 1:
            text_img, _, _ = vis_utils.write_text(text_img, 'EOS', text_x, text_y, frg_col,
                                                  show=0, font_size=font_size)

        vis_mask_ = cv2.resize(mask_sub_vis, (vis_size, vis_size))
        vis_image_ = cv2.resize(vis_image, (vis_size, vis_size))

        # vis_mask_, text_bb = vis_utils.write_text(vis_mask_, run_txt, 5, 5, col, font_size=24, show=0, bb=1)

        run_center = int(start + length / 2)
        run_y, run_x = np.unravel_index([run_center, ], (n_rows, n_cols))
        run_y, run_x = int(run_y[0]), int(run_x[0])

        run_x, run_y = int(run_x * resize_x), int(run_y * resize_y)
        """vis_mask_ to the right of vis_image_"""
        run_x += vis_size
        vis_image_cat = np.concatenate((vis_image_, vis_mask_), axis=1)

        left, top, right, bottom = text_bb
        text_bb_x, text_bb_y = int((left + right) / 2), int(bottom)
        """text_img is to the right of vis_mask_"""
        text_bb_x += int(vis_size * 2)
        text_bb_y += 10
        vis_image_cat = np.concatenate((vis_image_cat, text_img), axis=1)

        vis_image_cat = cv2.arrowedLine(vis_image_cat, (run_x, run_y), (text_bb_x, text_bb_y), col, 1, tipLength=0.01)
        cv2.imshow('vis_image_cat', vis_image_cat)
        k = cv2.waitKey(0 if _pause else 250)
        if k == 27:
            exit()
        elif k == 32:
            _pause = 1 - _pause

    cv2.waitKey(0)


def construct_rle(starts_rows, starts_cols, lengths, shape, starts_2d, starts_offset, lengths_offset):
    lengths += lengths_offset
    if starts_2d:
        starts_rows += 1 + starts_offset
        starts_cols += 1 + starts_offset
        rle = [item for sublist in zip(starts_rows, starts_cols, lengths) for item in sublist]
    else:
        starts = np.ravel_multi_index((starts_rows, starts_cols), shape)

        starts += 1 + starts_offset + 1
        rle = [int(item) for sublist in zip(starts, lengths) for item in sublist]
    return rle


def deconstruct_rle(rle, shape, starts_2d, starts_offset, lengths_offset):
    if starts_2d:
        start_rows, start_cols, lengths = [
            np.asarray(x, dtype=int) for x in (rle[0:][::3], rle[1:][::3], rle[2:][::3])]
        start_rows -= 1
        start_cols -= 1

        start_rows -= (starts_offset + 1)
        start_cols -= (starts_offset + 1)
    else:
        starts, lengths = [np.asarray(x, dtype=int) for x in (rle[0:][::2], rle[1:][::2])]
        starts -= (starts_offset + 1)
        start_rows, start_cols = np.unravel_index(starts, shape)

    lengths -= lengths_offset

    return start_rows, start_cols, lengths


def supersample_rle(starts_sub, lengths_sub, subsample, shape, max_length):
    n_rows, n_cols = shape
    n_rows_sub, n_cols_sub = int(n_rows / subsample), int(n_cols / subsample)
    max_length_sub = int(max_length / subsample)

    starts_rows_sub, starts_cols_sub = np.unravel_index(starts_sub, (n_rows_sub, n_cols_sub))

    starts_rows = starts_rows_sub / (n_rows_sub - 1) * (n_rows - 1)
    starts_cols = starts_cols_sub / (n_cols_sub - 1) * (n_cols - 1)

    starts_rows = starts_rows.astype(np.int64)
    starts_cols = starts_cols.astype(np.int64)

    lengths = (lengths_sub - 1).astype(np.float64) / (max_length_sub - 1) * (max_length - 1) + 1
    lengths = lengths.astype(np.int64)

    starts = np.ravel_multi_index((starts_rows, starts_cols), shape)

    return starts, lengths


def subsample_rle(starts, lengths, subsample, shape, max_length):
    if len(starts) == 0:
        return starts, lengths

    n_rows, n_cols = shape
    max_length_sub = int(max_length / subsample)
    n_rows_sub, n_cols_sub = int(n_rows / subsample), int(n_cols / subsample)

    starts_rows, starts_cols = np.unravel_index(starts, (n_rows, n_cols))

    starts_rows_norm, starts_cols_norm = (starts_rows.astype(np.float64) / (n_rows - 1),
                                          starts_cols.astype(np.float64) / (n_cols - 1))

    """
    lengths goes from 1 to max_length so 1 must be subtracted before normalizing so lengths_norm starts from 0
    and un-normalizing works correctly    
    """
    lengths_norm = (lengths.astype(np.float64) - 1) / (max_length - 1)

    lengths = ((lengths_norm * (max_length_sub - 1)) + 1).astype(np.int64)
    starts_rows = (starts_rows_norm * (n_rows_sub - 1)).astype(np.int64)
    starts_cols = (starts_cols_norm * (n_cols_sub - 1)).astype(np.int64)

    rle_all = list(zip(starts_rows, starts_cols, lengths))
    rle_unique = remove_duplicates(rle_all)

    starts_rows, starts_cols, lengths = zip(*rle_unique)

    starts_rows = np.asarray(starts_rows)
    starts_cols = np.asarray(starts_cols)
    lengths = np.asarray(lengths)

    starts = np.ravel_multi_index((starts_rows, starts_cols), (n_rows_sub, n_cols_sub))

    return starts, lengths


def rle_to_tokens(rle_cmp, shape, starts_offset, lengths_offset, class_offset, starts_2d):
    starts, lengths = rle_cmp[:2]

    if len(starts) == 0:
        return []

    starts += (starts_offset + 1)
    lengths += lengths_offset

    if starts_2d:
        starts_rows, starts_cols = np.unravel_index(starts, shape)
        starts_rows += (starts_offset + 1)
        starts_cols += (starts_offset + 1)
        rle_tokens_cmp = [starts_rows, starts_cols, lengths]
    else:
        rle_tokens_cmp = [starts, lengths]

    if len(rle_cmp) == 3:
        class_ids = np.asarray(rle_cmp[2])
        class_ids += class_offset
        rle_tokens_cmp.append(class_ids)
    else:
        assert len(rle_cmp) == 2, "rle_cmp length must be 2 or 3"

    rle_tokens = [int(item) for sublist in zip(*rle_tokens_cmp) for item in sublist]

    return rle_tokens


def mask_from_logits(
        logits_, shape,
        max_length,
        starts_bins,
        n_classes,
        starts_offset, lengths_offset, class_offset,
        starts_2d, multi_class
):
    rle_cmp = rle_from_logits(
        logits_,
        shape,
        max_length=max_length,
        starts_bins=starts_bins,
        n_classes=n_classes,
        starts_offset=starts_offset,
        lengths_offset=lengths_offset,
        class_offset=class_offset,
        starts_2d=starts_2d,
        multi_class=multi_class,
    )
    starts, lengths = rle_cmp[:2]
    if multi_class:
        class_ids = rle_cmp[2]
    else:
        class_ids = [1, ] * len(starts)

    mask = rle_to_mask(
        starts, lengths, class_ids,
        shape,
    )
    return mask, rle_cmp


def selective_argmax(arr, idx_range):
    start_idx, end_idx = idx_range
    return start_idx + np.argmax(arr[:, start_idx:end_idx], axis=1)


def rle_from_logits(
        rle_logits,
        shape,
        max_length,
        starts_bins,
        n_classes,
        starts_offset, lengths_offset, class_offset,
        starts_2d, multi_class):
    assert not starts_2d, "starts_2d is not supported yet"

    rle_tokens_raw = np.argmax(rle_logits, axis=1).squeeze()
    n_tokens_raw = len(rle_tokens_raw)

    """
    index of the first non-zero (non-padding) token from end
    EOS is the next token to this one
    """
    eos_idx = np.nonzero(rle_tokens_raw[::-1])[0]
    eos_idx = n_tokens_raw - eos_idx

    rle_logits_non_padding = rle_logits[:eos_idx, :]
    seq_len, vocab_size = rle_logits_non_padding.shape

    assert vocab_size >= starts_offset + starts_bins, "invalid vocab_size"

    coord_token_range = [starts_offset, starts_offset + starts_bins]
    len_token_range = [lengths_offset, lengths_offset + max_length]
    n_run_tokens = 2 if not multi_class else 3

    if seq_len % n_run_tokens != 0:
        rle_logits_non_padding = rle_logits_non_padding[:-(seq_len % n_run_tokens), :]

    starts_logits = rle_logits_non_padding[0::n_run_tokens, :]
    starts_tokens = selective_argmax(starts_logits, coord_token_range)
    starts = starts_tokens - starts_offset - 1

    len_logits = rle_logits_non_padding[1::n_run_tokens, :]
    len_tokens = selective_argmax(len_logits, len_token_range)
    lengths = len_tokens - lengths_offset

    rle_cmp = [starts, lengths]

    if multi_class:
        class_token_range = [class_offset, class_offset + n_classes]
        class_logits = rle_logits_non_padding[2::n_run_tokens, :]
        class_tokens = selective_argmax(class_logits, class_token_range)
        class_ids = class_tokens - class_offset
        rle_cmp.append(class_ids)
    return rle_cmp


def mask_from_tokens(rle_tokens, shape, starts_offset, lengths_offset, class_offset, starts_2d, multi_class):
    rle_cmp = rle_from_tokens(
        rle_tokens,
        shape,
        starts_offset=starts_offset,
        lengths_offset=lengths_offset,
        class_offset=class_offset,
        starts_2d=starts_2d,
        multi_class=multi_class,
    )
    starts, lengths = rle_cmp[:2]
    if multi_class:
        class_ids = rle_cmp[2]
    else:
        class_ids = [1, ] * len(starts)

    mask = rle_to_mask(
        starts, lengths, class_ids,
        shape,
    )
    return mask, rle_cmp


def rle_from_tokens(rle_tokens, shape, starts_offset, lengths_offset, class_offset, starts_2d, multi_class):
    n_run_tokens = 2
    if starts_2d:
        n_run_tokens += 1
    if multi_class:
        n_run_tokens += 1

    assert len(rle_tokens) % n_run_tokens == 0, f"rle_tokens length must be divisible by {n_run_tokens}"

    if starts_2d:
        starts_rows = np.asarray(rle_tokens[0:][::n_run_tokens], dtype=int)
        starts_cols = np.asarray(rle_tokens[1:][::n_run_tokens], dtype=int)

        starts_rows -= (starts_offset + 1)
        starts_cols -= (starts_offset + 1)

        len_id = 2

        starts = np.ravel_multi_index((starts_rows, starts_cols), shape)
    else:
        starts = np.asarray(rle_tokens[0:][::n_run_tokens], dtype=int)
        starts -= (starts_offset + 1)

        len_id = 1

    lengths = np.asarray(rle_tokens[len_id:][::n_run_tokens], dtype=int)
    lengths -= lengths_offset

    rle_cmp = [starts, lengths]

    if multi_class:
        class_ids = np.asarray(rle_tokens[len_id + 1:][::n_run_tokens], dtype=int)
        class_ids -= class_offset
        rle_cmp.append(class_ids)

    return rle_cmp


def get_rle_class_ids(mask, starts):
    mask_flat = mask.flatten()
    class_ids = [mask_flat[k] for k in starts]

    assert 0 not in class_ids, "class_ids must be non-zero"

    return class_ids


def rle_to_2d(rle, mask):
    n_rows, n_cols = mask.shape[:2]

    starts, lengths = [np.asarray(x, dtype=int) for x in (rle[0:][::2], rle[1:][::2])]

    starts_rows, starts_cols = np.unravel_index(starts, (n_rows, n_cols))

    assert np.all(starts_rows <= n_rows - 1), f"starts_rows cannot be > {n_rows - 1}"
    assert np.all(starts_cols <= n_cols - 1), f"starts_rows cannot be > {n_cols - 1}"

    rle = [int(item) for sublist in zip(starts_rows, starts_cols, lengths) for item in sublist]

    return rle


def mask_to_rle(mask, max_length):
    """
    https://www.kaggle.com/stainsby/fast-tested-rle
    https://ccshenyltw.medium.com/run-length-encode-and-decode-a33383142e6b

    :param mask:
    :param max_length:
    :param starts_2d:
    :return:
    """
    assert len(mask.shape) == 2, "only greyscale masks are supported"

    mask = np.copy(mask)
    mask[mask > 0] = 1

    mask_flat = mask.flatten()
    pixels = np.concatenate([[0], mask_flat, [0]])
    """the +1 in the original code was to convert indices from 0-based to 1-based"""
    runs = np.nonzero(pixels[1:] != pixels[:-1])[0]

    if len(runs) == 0:
        return [], []

    if len(runs) % 2 != 0:
        raise AssertionError("runs must have even length")

    runs[1::2] -= runs[::2]
    starts, lengths = runs[::2], runs[1::2]

    if max_length > 0:
        overlong_runs = np.nonzero(lengths > max_length)[0]
        if len(overlong_runs) > 0:
            starts, lengths = split_runs(overlong_runs, starts, lengths, max_length)

    assert np.all(lengths <= max_length), f"run length cannot be > {max_length}"
    assert np.all(lengths > 0), "run length cannot be 0"
    n_rows, n_cols = mask.shape

    n_pix = n_rows * n_cols

    assert np.all(starts <= n_pix - 1), f"starts cannot be > {n_pix - 1}"

    return starts, lengths


def rle_to_mask(starts, lengths, class_ids, shape):
    if len(starts) == 0:
        mask = np.zeros(tuple(shape), dtype=np.uint8)
        return mask

    """ends are exclusive while starts are inclusive"""
    ends = starts + lengths
    mask_flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi, label in zip(starts, ends, class_ids):
        mask_flat[lo:hi] = label

    mask = mask_flat.reshape(shape)

    return mask


def read_class_info(class_names_path):
    class_info = [k.strip() for k in open(class_names_path, 'r').readlines() if k.strip()]
    class_names, class_cols = zip(*[[m.strip() for m in k.split('\t')] for k in class_info])

    if 'background' not in class_names:
        assert 'black' not in class_cols, "black should only be used for background"
        class_names = ('background',) + class_names
        class_cols = ('black',) + class_cols

    class_id_to_col = {i: x for (i, x) in enumerate(class_cols)}
    class_id_to_name = {i: x for (i, x) in enumerate(class_names)}

    n_classes = len(class_cols)
    palette = []
    for class_id in range(n_classes):
        col = class_cols[class_id]

        col_rgb = col_bgr[col][::-1]

        palette.append(col_rgb)

    palette_flat = [value for color in palette for value in color]

    class_name_to_id = {x: i for (i, x) in enumerate(class_names)}

    return class_names, class_id_to_col, class_id_to_name, class_name_to_id, palette_flat


def get_category_names(
        category_names_path: Optional[str]) -> Dict[int, Dict[str, Any]]:
    """Returns dictionary of category names.

    Args:
      category_names_path: Path to category names json. Expected to be a json file
        with the format {"categories": [{"id": 1, "name": "Person"}, ...]}. If not
        specified, the category id is used as the name.

    Returns:
      Dictionary with the format {1: {"name": "Person"}, ...}
    """
    assert category_names_path, "category_names_path must be provided"

    print(f'Loading category names from {category_names_path}')
    if category_names_path.endswith('.json.gz'):
        import compress_json
        annotations = compress_json.load(category_names_path)
    else:
        with open(category_names_path, 'r') as f:
            annotations = json.load(f)
    category_names = {c['id']: c for c in annotations['categories']}

    # assert 0 not in category_names.keys(), "class IDs must to be > 0"

    try:
        bkg_class = category_names[0]['name']
    except KeyError:
        pass
    else:
        assert bkg_class == 'background', "class id 0 must be used only for background"

    return category_names


def build_instance_prompt_seq(task_vocab_id: int, bbox, label,
                              quantization_bins, coord_vocab_shift):
    """"Build prompt seq for instance tasks like instance segmentation, keypoints.

    Args:
      task_vocab_id: Vocab id for the task.
      bbox: `float` bounding box of shape (bsz, n, 4).
      label: `int` label of shape (bsz, n).
      quantization_bins: `int`.
      coord_vocab_shift: `int`, shifting coordinates by a specified integer.

    Returns:
      discrete prompt sequence of (task_id, bbox, label) with shape (bsz, n, 6).
      tokens are zero'ed if label is padding (0).
    """
    task_id = tf.constant(task_vocab_id)
    quantized_bbox = utils.quantize(bbox, quantization_bins)
    quantized_bbox = quantized_bbox + coord_vocab_shift
    new_label = tf.expand_dims(label + vocab.BASE_VOCAB_SHIFT, -1)
    prompt_seq = tf.concat([quantized_bbox, new_label], axis=-1)
    task_id = tf.zeros_like(prompt_seq[..., :1]) + tf.cast(task_id, label.dtype)
    prompt_seq = tf.concat([task_id, prompt_seq], -1)
    is_padding = tf.expand_dims(tf.equal(label, 0), -1)
    prompt_seq = tf.where(is_padding, tf.zeros_like(prompt_seq), prompt_seq)
    return prompt_seq


def build_instance_response_seq_from_points(points, label, quantization_bins,
                                            coord_vocab_shift):
    """"Build target seq for instance tasks like instance segmentation, keypoints.

    Args:
      points: `float` points of shape (bsz, n, k).
      label: `int` label of shape (bsz, n).
      quantization_bins: `int`.
      coord_vocab_shift: `int`, shifting coordinates by a specified integer.

    Returns:
      discrete target sequence with shape (bsz, n, k). tokens are zero'ed
      if label is padding (0).
    """
    quantized_points = utils.quantize(points, quantization_bins)
    quantized_points = quantized_points + coord_vocab_shift
    response_seq = utils.replace_reserved_tokens(
        quantized_points, points, vocab.FLOAT_TO_TOKEN)
    is_padding = tf.expand_dims(tf.equal(label, 0), -1)
    response_seq = tf.where(is_padding, tf.zeros_like(response_seq), response_seq)
    return response_seq


def build_prompt_seq_from_task_id(task_vocab_id: int,
                                  response_seq=None,
                                  prompt_shape=None):
    """"Build prompt seq just using task id.

    Args:
      task_vocab_id: Vocab id for the task.
      response_seq: an (optional) discrete target sequence with shape (bsz, ..., k).
      prompt_shape: an (optional) tuple for prompt shape. One and only one of
        `response_seq` and `prompt_shape` should be specified.

    Returns:
      discrete input sequence of task id with shape (bsz, ..., 1).
    """
    task_id = tf.constant(task_vocab_id)
    prompt_seq = None
    if response_seq is not None:
        prompt_seq = tf.zeros_like(response_seq[..., :1]) + tf.cast(
            task_id, response_seq.dtype)
    if prompt_shape is not None:
        assert response_seq is None, 'double specification'
        prompt_seq = tf.zeros(prompt_shape, dtype=tf.int64) + tf.cast(task_id, dtype=tf.int64)

    assert prompt_seq is not None, "either response_seq or prompt_shape must be provided"

    return prompt_seq


def decode_instance_seq_to_points(seq, quantization_bins, coord_vocab_shift):
    """Decode points for seq from `build_instance_response_seq_from_points`."""
    assert seq.dtype in (tf.int64, tf.int32)
    points = seq - coord_vocab_shift
    points = utils.dequantize(points, quantization_bins)
    return utils.replace_reserved_tokens(points, seq, vocab.TOKEN_TO_FLOAT)


def decode_video_seq_to_bbox(
        logits,
        seq,
        vid_len,
        quantization_bins,
        coord_vocab_shift,
        seq_mask=None,
):
    _, seqlen, vocab_size = logits.shape

    if seq_mask is not None:
        seq = tf.where(seq_mask, seq, tf.cast(-1, seq.dtype))

    bbox_seq_len = vid_len * 4 + 1

    # truncate out the last few tokens
    if seqlen % bbox_seq_len != 0:
        truncate_len = seqlen % bbox_seq_len
        seq = seq[..., :-truncate_len]
        logits = logits[..., :-truncate_len, :]
        if seq_mask is not None:
            seq_mask = seq_mask[..., :-truncate_len]

    """
    extract probs for all classes - starting from 5th element and extracting every fifth element from there
    """
    probs = tf.nn.softmax(logits)
    class_probs = probs[:, bbox_seq_len - 1::bbox_seq_len]  # (bsz, instances, vocab_size)
    """
    mask-out non-class portions of the vocab
    """
    mask_s1 = [0.] * vocab.BASE_VOCAB_SHIFT  # reserved.
    mask_s2 = [1.] * (coord_vocab_shift - vocab.BASE_VOCAB_SHIFT)  # labels.
    mask_s3 = [0] * (vocab_size - coord_vocab_shift)  # coordinates and others.
    mask = tf.constant(mask_s1 + mask_s2 + mask_s3)
    """
    this is where the claims of automatically learning domain-specific tokens breaks down - we simply select the 
    class with the max prob even if some non-class token has higher prob than the max-prob class
    """
    class_tokens = tf.argmax(class_probs * mask[tf.newaxis, tf.newaxis, :], -1)
    """
    round-about way of selecting the class prob of each bbox as its score
    """
    # scores = tf.reduce_sum(class_probs * tf.one_hot(class_tokens, vocab_size), -1)
    scores = tf.gather_nd(class_probs, class_tokens[:, :, tf.newaxis], batch_dims=2)

    class_ids = tf.maximum(class_tokens - vocab.BASE_VOCAB_SHIFT, 0)
    bboxes = seq_to_video_bbox(seq, quantization_bins, vid_len, coord_vocab_shift)
    return class_ids, bboxes, scores


def seq_to_video_bbox(seq, quantization_bins, vid_len, coord_vocab_shift):
    """Returns [0, 1] normalized yxyx bbox from token sequence."""
    # [batch, 5*num_instances]
    assert seq.shape.rank == 2, f'seq has non-rank 2 shape: {seq.shape.as_list()}'

    bbox_seq_len = vid_len * 4 + 1

    # [batch, num_instances, 1]

    boxes = []

    for _id in range(vid_len):
        bbox_start_id = 4 * _id
        ymin = tf.expand_dims(seq[:, bbox_start_id::bbox_seq_len], -1)
        xmin = tf.expand_dims(seq[:, bbox_start_id + 1::bbox_seq_len], -1)
        ymax = tf.expand_dims(seq[:, bbox_start_id + 2::bbox_seq_len], -1)
        xmax = tf.expand_dims(seq[:, bbox_start_id + 3::bbox_seq_len], -1)
        box_tokens = tf.concat([ymin, xmin, ymax, xmax], axis=-1)

        is_no_box = tf.equal(box_tokens, vocab.NO_BOX_TOKEN)
        is_padding = tf.equal(box_tokens, vocab.PADDING_TOKEN)

        box_quant = box_tokens - coord_vocab_shift
        is_not_coord = tf.less(box_quant, 0)

        box_dequant = utils.dequantize(box_quant, quantization_bins)

        box_clipped = tf.minimum(tf.maximum(box_dequant, 0), 1)

        is_invalid = tf.math.logical_or(is_no_box, is_not_coord)
        is_invalid = tf.math.logical_or(is_invalid, is_padding)

        box_clipped = tf.where(
            is_invalid,
            tf.cast(vocab.NO_BOX_FLOAT, box_clipped.dtype),
            box_clipped)

        boxes.append(box_clipped)

    boxes = tf.concat(boxes, axis=-1)

    return boxes


def decode_object_seq_to_bbox(logits,
                              pred_seq,
                              quantization_bins,
                              coord_vocab_shift):
    """Decode objects (label & bbox) for seq from `build_response_seq_from_bbox`.

    Assume yxyxc format with truncation at the end for any uneven extra tokens.
      Replace class tokens with argmax instead of sampling.

    Args:
      logits: `float` output logits in shape of (bsz, max_seq_len, vocab_size).
      pred_seq: `int` pred sequence in shape of (bsz, max_seq_len).
      quantization_bins: `int` for bins.
      coord_vocab_shift: `int`, shifting coordinates by a specified integer.

    Returns:
      pred_class: `int` of shape (bsz, max_instances_per_image).
      pred_bbox: `float` of shape (bsz, max_instances_per_image, 4).
      pred_score: `float` of shape (bsz, max_instances_per_image).
    """
    _, seqlen, vocab_size = logits.shape
    if seqlen % 5 != 0:  # truncate out the last few tokens.
        pred_seq = pred_seq[..., :-(seqlen % 5)]
        logits = logits[..., :-(seqlen % 5), :]
    """
    extract probs for all classes - starting from 5th element and extracting every fifth element from there
    """
    pred_class_p = tf.nn.softmax(logits)[:, 4::5]  # (bsz, instances, vocab_size)

    """
    mask-out non-class portions of the vocab
    """
    mask_s1 = [0.] * vocab.BASE_VOCAB_SHIFT  # reserved.
    mask_s2 = [1.] * (coord_vocab_shift - vocab.BASE_VOCAB_SHIFT)  # labels.
    mask_s3 = [0] * (vocab_size - coord_vocab_shift)  # coordinates and others.
    mask = tf.constant(mask_s1 + mask_s2 + mask_s3)
    """
    this is where the claims of automatically learning domain-specific tokens breaks down - we simply select the 
    class with the max prob even if some non-class token has higher prob than the max-prob class
    """
    pred_class = tf.argmax(pred_class_p * mask[tf.newaxis, tf.newaxis, :], -1)

    """
    round-about way of selecting the class prob of each bbox as its score
    """
    pred_score = tf.reduce_sum(
        pred_class_p * tf.one_hot(pred_class, vocab_size), -1)
    pred_class = tf.maximum(pred_class - vocab.BASE_VOCAB_SHIFT, 0)
    pred_bbox = seq_to_bbox(pred_seq - coord_vocab_shift, quantization_bins)
    return pred_class, pred_bbox, pred_score


def seq_to_bbox(seq, quantization_bins, seq_format='yxyx_name'):
    """Returns [0, 1] normalized yxyx bbox from token sequence."""
    # [batch, 5*num_instances]
    assert seq.shape.rank == 2, f'seq has non-rank 2 shape: {seq.shape.as_list()}'
    # [batch, num_instances, 1]
    if seq_format.startswith('name'):
        ymin = tf.expand_dims(seq[:, 1::5], -1)
        xmin = tf.expand_dims(seq[:, 2::5], -1)
        ymax = tf.expand_dims(seq[:, 3::5], -1)
        xmax = tf.expand_dims(seq[:, 4::5], -1)
    else:
        ymin = tf.expand_dims(seq[:, 0::5], -1)
        xmin = tf.expand_dims(seq[:, 1::5], -1)
        ymax = tf.expand_dims(seq[:, 2::5], -1)
        xmax = tf.expand_dims(seq[:, 3::5], -1)
    if seq_format in ['name_cycxhw', 'cycxhw_name']:
        ycnt, xcnt, ysize, xsize = ymin, xmin, ymax, xmax
        ymin = ycnt - ysize // 2
        xmin = xcnt - xsize // 2
        ymax = ycnt + ysize // 2
        xmax = xcnt + xsize // 2
    quantized_box = tf.concat([ymin, xmin, ymax, xmax], axis=-1)
    quantized_box = utils.dequantize(quantized_box, quantization_bins)
    return tf.minimum(tf.maximum(quantized_box, 0), 1)


def compute_weighted_scores(bbox_scores, pred_seq, logits,
                            points_score_weight):
    """Computes per instance score as weighted sum of box score and mean pred_seq score."""
    probs = tf.nn.softmax(logits, axis=-1)
    # Set 0 weight for padding tokens.
    token_weight = tf.where(tf.equal(pred_seq, vocab.PADDING_TOKEN), 0.0, 1.0)
    likelihoods = tf.gather(probs, pred_seq, batch_dims=pred_seq.shape.rank)
    points_score = (
            tf.reduce_sum(likelihoods * token_weight, axis=-1) /
            tf.reduce_sum(token_weight, axis=-1))
    num_instances_in_batch = bbox_scores.shape[0]
    num_samples = points_score.shape[0] // num_instances_in_batch
    points_score = tf.reshape(points_score, [num_instances_in_batch, num_samples])
    points_score = tf.reduce_mean(points_score, axis=-1)
    return (points_score_weight * points_score +
            (1 - points_score_weight) * bbox_scores)


def join_if_not_none(args, sep):
    args = [str(arg) for arg in args if arg is not None]
    return sep.join(args)


def integer_map_to_bits(integer_map, n_bits_label, b_scale, num_channels=2):
    """Converts an integer map to analog bits.

    Args:
      integer_map: integer tensor of shape [..., num_channels].
      n_bits_label: integer. Total number of bits of the analog bits.
      b_scale: float. Scaling of the analog bits.
      num_channels: integer. Number of channels in the integer map.

    Returns:
      Analog bits of shape [..., n_bits_label].
    """
    bits = []
    for i in range(num_channels):
        bits.append(utils.int2bits(
            integer_map[..., i], n_bits_label // num_channels, tf.float32))
    bits = tf.concat(bits, -1)
    bits = (bits * 2 - 1) * b_scale
    return bits


def bits_to_panoptic_map(bits, n_bits_label, num_classes,
                         max_instances_per_image):
    """Converts analog bits to a panoptic map.

    Args:
      bits: float tensor of shape [..., n_bits_label].
      n_bits_label: integer. Number of bits of the analog bits.
      num_classes: integer. Number of semantic classes.
      max_instances_per_image: integer. Maximum number of instances in an image.

    Returns:
      The integer panoptic map of [..., 2], where the first channel is the
      semantic map and the second channel is the instance map.
    """
    s_map = utils.bits2int(bits[..., :n_bits_label // 2] > 0, tf.int32)
    s_map = tf.minimum(s_map, num_classes - 1)
    i_map = utils.bits2int(bits[..., n_bits_label // 2:] > 0, tf.int32)
    i_map = tf.minimum(i_map, max_instances_per_image - 1)
    panoptic_map = tf.stack([s_map, i_map], -1)
    return panoptic_map


def get_normalized_weight(id_map, total_num_ids, p=1.0):
    """Returns instance normalized weight given id_map (bsz, h, w)."""
    id_map_hot = tf.one_hot(id_map, total_num_ids)
    weight = 1. / (tf.reduce_sum(id_map_hot, [1, 2]) + 1)
    weight = tf.einsum('bhwk,bk->bhw', id_map_hot, weight)
    weight = tf.pow(weight, p)
    weight /= tf.reduce_sum(weight, [1, 2], keepdims=True)
    weight *= tf.cast(tf.math.reduce_prod(tf.shape(id_map)[1:]), weight.dtype)
    return weight
