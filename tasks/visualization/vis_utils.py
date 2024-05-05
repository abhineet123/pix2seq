import abc
import copy
import collections
import os.path

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

import cv2
import skvideo.io

import utils
import vocab
from tasks.visualization import shape_utils
from tasks.visualization import standard_fields as fields

import six
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

from eval_utils import draw_box, annotate, resize_ar_video, resize_ar

_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def vis_json_ann(video, object_anns, category_id_to_name_map, image_dir, is_video=True):
    from eval_utils import draw_box, annotate

    file_names_to_img = {}
    for object_ann in object_anns:
        if is_video:
            bboxes = object_ann["bboxes"]
            file_names = video['file_names']
        else:
            bboxes = [object_ann["bbox"], ]
            file_names = [video['file_name'], ]

        category_id = object_ann['category_id']
        class_name = category_id_to_name_map[category_id]
        for file_name, bbox in zip(file_names, bboxes):
            try:
                img = file_names_to_img[file_name]
            except KeyError:
                img_path = os.path.join(image_dir, file_name)
                assert os.path.isfile(img_path), f"nonexistent img_path: {img_path}"
                img = cv2.imread(img_path)
                file_names_to_img[file_name] = img

            if bbox is None:
                continue

            cx, cy, w, h = bbox
            draw_box(img, [cx, cy, w, h], _id=class_name,
                     color='green', thickness=1, norm=False, xywh=True)
            file_names_to_img[file_name] = img

    for file_name, img in file_names_to_img.items():
        img = annotate(img, file_name)
        cv2.imshow('img', img)
        cv2.waitKey(100)


def debug_image_pipeline(dataset, train_transforms, batched_examples, model_dir, vis, training):
    batch_size = batched_examples.shape[0]

    from datetime import datetime

    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    vis_img_dir = os.path.join(model_dir, f"debug_transforms_{time_stamp}")

    os.makedirs(vis_img_dir, exist_ok=True)

    batched_examples_ = None
    for single_example in batched_examples:
        single_example = dataset.debug_pipeline(
            single_example, training)
        if batched_examples_ is None:
            batched_examples_ = {
                k: tf.expand_dims(v, 0)
                for k, v in single_example.items()
            }
        else:
            batched_examples_ = {
                k: tf.concat((v, tf.expand_dims(single_example[k], 0)), 0)
                for k, v in batched_examples_.items()
            }

    batched_examples = batched_examples_
    proc_examples = {
        k: [] for k, v in batched_examples.items()
    }

    for i in range(batch_size):
        single_example = {
            k: v[i, ...] for k, v in batched_examples.items()
        }

        # proc_images = dict(
        #     orig=single_example["image"]
        # )
        # proc_bboxes = proc_masks = None
        # if 'bbox' in single_example:
        #     proc_bboxes = dict(
        #         orig=single_example["bbox"]
        #     )
        # elif 'mask' in single_example:
        #     proc_masks = dict(
        #         orig=single_example["mask"]
        #     )
        # else:
        #     raise AssertionError('single_example has neither bbox nor mask')

        import copy
        proc_example = copy.copy(single_example)

        if vis:
            save_image_sample(single_example, 'orig', 0, vis_img_dir)

        for t_id, t in enumerate(train_transforms):
            t_name = t.config.name
            proc_example = t.process_example(proc_example)

            # proc_images[t_name] = proc_example["image"]
            # if proc_bboxes is not None:
            #     proc_bboxes[t_name] = proc_example["bbox"]
            # elif proc_masks is not None:
            #     proc_masks[t_name] = proc_example["mask"]

            if vis:
                save_image_sample(proc_example, t_name, t_id + 1, vis_img_dir)
                # image = proc_example["image"]
                # image = tf.image.convert_image_dtype(image, tf.uint8)

                # video_ids = proc_example["video_id"]
                # file_names = proc_example["file_names"]

                # vis_utils.save_video(image, file_names, t_name, self.config.model_dir)

        for k, v in proc_examples.items():
            v.append(proc_example[k])

    for k, v in proc_examples.items():
        proc_examples[k] = tf.stack(v, axis=0)

    return proc_examples


def debug_video_transforms(transforms, batched_examples, vis, model_dir):
    # bbox_np = batched_examples['bbox'].numpy()
    # class_id_np = batched_examples['class_id'].numpy()
    # class_name_np = batched_examples['class_name'].numpy()
    batch_size = batched_examples["video"].shape[0]
    proc_examples = {
        k: [] for k, v in batched_examples.items()
    }
    from datetime import datetime

    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    vis_img_dir = os.path.join(model_dir, "debug_transforms", f"{time_stamp}")

    os.makedirs(vis_img_dir, exist_ok=True)

    for batch_id in range(batch_size):
        single_example = {
            k: v[batch_id, ...] for k, v in batched_examples.items()
        }
        proc_videos = dict(
            orig=single_example["video"]
        )
        proc_bboxes = dict(
            orig=single_example["bbox"]
        )
        save_video_sample(single_example, batch_id, 'orig', 0, vis_img_dir)

        proc_example = copy.copy(single_example)
        for t_id, t in enumerate(transforms):
            t_name = t.config.name
            proc_example = t.process_example(proc_example)

            proc_videos[t_name] = proc_example["video"]
            proc_bboxes[t_name] = proc_example["bbox"]

            if not vis:
                continue

            save_video_sample(proc_example, batch_id, t_name, t_id + 1, vis_img_dir)

        for k, v in proc_examples.items():
            v.append(proc_example[k])

    for k, v in proc_examples.items():
        proc_examples[k] = tf.stack(v, axis=0)

    return proc_examples


def _force_matplotlib_backend():
    """force the backend of matplotlib to Agg."""
    if matplotlib.get_backend().lower() != 'agg':  # case-insensitive
        matplotlib.use('Agg')  # Set headless-friendly backend.


def _get_multiplier_for_color_randomness():
    """Returns a multiplier to get semi-random colors from successive indices.

    This function computes a prime number, p, in the range [2, 17] that:
    - is closest to len(STANDARD_COLORS) / 10
    - does not divide len(STANDARD_COLORS)

    If no prime numbers in that range satisfy the constraints, p is returned as 1.

    Once p is established, it can be used as a multiplier to select
    non-consecutive colors from STANDARD_COLORS:
    colors = [(p * i) % len(STANDARD_COLORS) for i in range(20)]
    """
    num_colors = len(STANDARD_COLORS)
    prime_candidates = [5, 7, 11, 13, 17]

    # Remove all prime candidates that divide the number of colors.
    prime_candidates = [p for p in prime_candidates if num_colors % p]
    if not prime_candidates:
        return 1

    # Return the closest prime number to num_colors / 10.
    abs_distance = [np.abs(num_colors / 10. - p) for p in prime_candidates]
    num_candidates = len(abs_distance)
    inds = [i for _, i in sorted(zip(abs_distance, range(num_candidates)))]
    return prime_candidates[inds[0]]


def save_image(
        image, vid_writers, out_vis_dir, seq_id, image_id_,
        video_id_=None, unpadded_size=None, orig_size=None,
):
    import cv2
    import eval_utils
    if video_id_ is not None:
        image_name_, image_ext_ = os.path.splitext(image_id_)
        vis_name = f'{image_name_}_{video_id_}{image_ext_}'
    else:
        vis_name = image_id_

    # print(f'seq_id: {seq_id}')
    # print(f'image_id_: {image_id_}')

    if unpadded_size is not None:
        unpadded_h, unpadded_w = map(int, unpadded_size)
        image = image[:unpadded_h, :unpadded_w, :]

    # if orig_size is not None:
    #     orig_h, orig_w = map(int, orig_size)
    #     image = resize_ar(image, height=orig_h, width=orig_w, strict=True)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = eval_utils.annotate(image, vis_name)

    if vid_writers is not None:
        vid_writer = vid_writers[seq_id]
        if vid_writer is None:
            for seq_id_, vid_writer_ in vid_writers.items():
                close_video_writers(vid_writer_)
                vid_writers[seq_id_] = None

            ext = 'mp4'
            seq_vis_path = os.path.join(out_vis_dir, f'{seq_id}.{ext}')
            vid_writer = get_video_writer(seq_vis_path)

            print(f'S{seq_id} :: vis video: {seq_vis_path}')
            vid_writers[seq_id] = vid_writer

        write_frames_to_videos(vid_writer, image)
    else:
        seq_vis_dir = os.path.join(out_vis_dir, seq_id)
        os.makedirs(seq_vis_dir, exist_ok=True)
        vis_path = os.path.join(seq_vis_dir, vis_name)
        cv2.imwrite(vis_path, image)

    # cv2.imshow('image', image)
    # cv2.waitKey(100)


def save_video_sample(proc_example, batch_id, t_name, t_id, vis_img_dir):
    import cv2

    video = proc_example["video"]
    class_names = proc_example["class_name"]
    file_names = proc_example["file_names"]
    bboxes = proc_example["bbox"]
    # video_ids = proc_example["video_id"]

    video = tf.image.convert_image_dtype(video, tf.uint8)

    for img_id, (image, file_name) in enumerate(zip(video, file_names)):
        image_np = image.numpy()
        file_name_np = file_name.numpy()
        file_name_np = file_name_np.decode('utf-8')

        img_name = os.path.basename(file_name_np)
        img_name, img_ext = os.path.splitext(img_name)

        seq_name = os.path.basename(os.path.dirname(file_name_np))

        vis_img_name = f'{seq_name} {batch_id:02d}-{img_id:02d} {img_name} {t_id:04d} {t_name}'

        vis_img_path = os.path.join(vis_img_dir, f'{vis_img_name}{img_ext}')

        bbox_id = img_id * 4
        for bbox, class_name in zip(bboxes, class_names):
            class_name = class_name.numpy().decode('utf-8')
            bbox_np = bbox.numpy()
            bbox_np_ = bbox_np[bbox_id:bbox_id + 4]
            if np.any(np.isnan(bbox_np_)):
                assert np.all(np.isnan(bbox_np_)), "either all or none of the bounding box coordinates can be NAN"
                continue

            ymin, xmin, ymax, xmax = bbox_np_
            # xmin, ymin, xmax, ymax = bbox_np_
            draw_box(image_np, [xmin, ymin, xmax, ymax], _id=class_name,
                     color='green', thickness=1, norm=True, xywh=False)

        # video_np = [img.numpy() for img in video]
        # out_img = np.concatenate(video_np, axis=1)
        cv2.imwrite(vis_img_path, image_np)

        print(f'vis_img_path: {vis_img_path}')
        print()


def save_image_sample(proc_example, t_name, t_id, vis_img_dir):
    import cv2

    image = [proc_example["image"], ]
    file_name = proc_example["image/id"]

    image = tf.image.convert_image_dtype(image, tf.uint8)

    image_np = image.numpy().squeeze()

    image_h, image_w = image_np.shape[:2]

    image_vis = image_np.copy()

    file_name_np = file_name.numpy()
    file_name_np = file_name_np.decode('utf-8')

    img_name = os.path.basename(file_name_np)
    img_name, img_ext = os.path.splitext(img_name)
    if not img_ext:
        img_ext = '.jpg'

    seq_name = os.path.basename(os.path.dirname(file_name_np))
    vis_img_name = f'{seq_name} {img_name} {t_id:04d} {t_name}'

    if 'bbox' in proc_example:
        class_ids = proc_example["label"]
        bboxes = proc_example["bbox"]
        for bbox, class_id in zip(bboxes, class_ids):
            class_id = class_id.numpy()
            bbox_np_ = bbox.numpy()

            ymin, xmin, ymax, xmax = bbox_np_
            draw_box(image_vis, [xmin, ymin, xmax, ymax], _id=class_id,
                     color='green', thickness=1, norm=True, xywh=False)

    elif 'mask' in proc_example:
        mask = proc_example["mask"]
        mask = tf.image.convert_image_dtype(mask, tf.uint8)
        mask = mask.numpy().squeeze()
        mask[mask > 0] = 255
        vis_mask_dir = os.path.join(vis_img_dir, f'masks')
        os.makedirs(vis_mask_dir, exist_ok=True)
        vis_mask_path = os.path.join(vis_mask_dir, f'{vis_img_name}.png')
        print(f'vis_mask_path: {vis_mask_path}')

        cv2.imwrite(vis_mask_path, mask)
    else:
        raise AssertionError('proc_example has neither bbox nor mask')

    vis_img_path = os.path.join(vis_img_dir, f'{vis_img_name}{img_ext}')
    image_vis = annotate(image_vis, vis_img_name)
    cv2.imwrite(vis_img_path, image_vis)

    # image_show = resize_ar(image_vis, width=900)
    # cv2.imshow('image_vis', image_show)
    # k = cv2.waitKey(0)
    # if k == 27:
    #     exit()

    print(f'vis_img_path: {vis_img_path}')
    print()


def save_image_array_as_png(image, output_path):
    """Saves an image (represented as a numpy array) to PNG.

    Args:
      image: a numpy array with shape [height, width, 3].
      output_path: path to which image should be written.
    """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    with tf.gfile.Open(output_path, 'w') as fid:
        image_pil.save(fid, 'PNG')


def encode_image_array_as_png_str(image):
    """Encodes a numpy array into a PNG string.

    Args:
      image: a numpy array with shape [height, width, 3].

    Returns:
      PNG encoded image string.
    """
    image_pil = Image.fromarray(np.uint8(image))
    output = six.BytesIO()
    image_pil.save(output, format='PNG')
    png_string = output.getvalue()
    output.close()
    return png_string


def draw_bounding_box_on_image_array(
        image,
        ymin,
        xmin,
        ymax,
        xmax,
        color='red',
        thickness=4,
        display_str_list=(),
        use_normalized_coordinates=True):
    """Adds a bounding box to an image (numpy array).

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Args:
      image: a numpy array with shape [height, width, 3].
      ymin: ymin of bounding box.
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list: list of strings to display in box (each to be shown on its
        own line).
      use_normalized_coordinates: If True (default), treat coordinates ymin, xmin,
        ymax, xmax as relative to the image.  Otherwise treat coordinates as
        absolute.
    """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                               thickness, display_str_list,
                               use_normalized_coordinates)
    image_vis = np.array(image_pil)
    np.copyto(image, image_vis)

    return image_vis


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    if thickness > 0:
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                   (left, top)],
                  width=thickness,
                  fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getbbox(ds)[3] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getbbox(display_str)[2:]
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill='black',
                  font=font)
        text_bottom -= text_height - 2 * margin


def draw_bounding_boxes_on_image_array(image,
                                       boxes,
                                       color='red',
                                       thickness=4,
                                       display_str_list_list=()):
    """Draws bounding boxes on image (numpy array).

    Args:
      image: a numpy array object.
      boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax). The
        coordinates are in normalized format between [0, 1].
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list_list: list of list of strings. a list of strings for each
        bounding box. The reason to pass a list of strings for a bounding box is
        that it might contain multiple labels.

    Raises:
      ValueError: if boxes is not a [N, 4] array
    """
    image_pil = Image.fromarray(image)
    draw_bounding_boxes_on_image(image_pil, boxes, color, thickness,
                                 display_str_list_list)
    np.copyto(image, np.array(image_pil))


def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 color='red',
                                 thickness=4,
                                 display_str_list_list=()):
    """Draws bounding boxes on image.

    Args:
      image: a PIL.Image object.
      boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax). The
        coordinates are in normalized format between [0, 1].
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list_list: list of list of strings. a list of strings for each
        bounding box. The reason to pass a list of strings for a bounding box is
        that it might contain multiple labels.

    Raises:
      ValueError: if boxes is not a [N, 4] array
    """
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('Input must be of size [N, 4]')
    for i in range(boxes_shape[0]):
        display_str_list = ()
        if display_str_list_list:
            display_str_list = display_str_list_list[i]
        draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],
                                   boxes[i, 3], color, thickness, display_str_list)


def create_visualization_fn(category_index,
                            include_masks=False,
                            include_keypoints=False,
                            include_track_ids=False,
                            **kwargs):
    """Constructs a visualization function that can be wrapped in a py_func.

    py_funcs only accept positional arguments. This function returns a suitable
    function with the correct positional argument mapping. The positional
    arguments in order are:
    0: image
    1: boxes
    2: classes
    3: scores
    [4-6]: masks (optional)
    [4-6]: keypoints (optional)
    [4-6]: track_ids (optional)

    -- Example 1 --
    vis_only_masks_fn = create_visualization_fn(category_index,
      include_masks=True, include_keypoints=False, include_track_ids=False,
      **kwargs)
    image = tf.py_func(vis_only_masks_fn,
                       inp=[image, boxes, classes, scores, masks],
                       Tout=tf.uint8)

    -- Example 2 --
    vis_masks_and_track_ids_fn = create_visualization_fn(category_index,
      include_masks=True, include_keypoints=False, include_track_ids=True,
      **kwargs)
    image = tf.py_func(vis_masks_and_track_ids_fn,
                       inp=[image, boxes, classes, scores, masks, track_ids],
                       Tout=tf.uint8)

    Args:
      category_index: a dict that maps integer ids to category dicts. e.g.
        {1: {1: 'dog'}, 2: {2: 'cat'}, ...}
      include_masks: Whether masks should be expected as a positional argument in
        the returned function.
      include_keypoints: Whether keypoints should be expected as a positional
        argument in the returned function.
      include_track_ids: Whether track ids should be expected as a positional
        argument in the returned function.
      **kwargs: Additional kwargs that will be passed to
        visualize_boxes_and_labels_on_image_array.

    Returns:
      Returns a function that only takes tensors as positional arguments.
    """

    def visualization_py_func_fn(*args):
        """Visualization function that can be wrapped in a tf.py_func.

        Args:
          *args: First 4 positional arguments must be: image - uint8 numpy array
            with shape (img_height, img_width, 3). boxes - a numpy array of shape
            [N, 4]. classes - a numpy array of shape [N]. scores - a numpy array of
            shape [N] or None. -- Optional positional arguments -- instance_masks -
            a numpy array of shape [N, image_height, image_width]. keypoints - a
            numpy array of shape [N, num_keypoints, 2]. track_ids - a numpy array of
            shape [N] with unique track ids.

        Returns:
          uint8 numpy array with shape (img_height, img_width, 3) with overlaid
          boxes.
        """
        image = args[0]
        boxes = args[1]
        classes = args[2]
        scores = args[3]
        masks = keypoints = track_ids = None
        pos_arg_ptr = 4  # Positional argument for first optional tensor (masks).
        if include_masks:
            masks = args[pos_arg_ptr]
            pos_arg_ptr += 1
        if include_keypoints:
            keypoints = args[pos_arg_ptr]
            pos_arg_ptr += 1
        if include_track_ids:
            track_ids = args[pos_arg_ptr]

        return visualize_boxes_and_labels_on_image_array(
            image,
            boxes,
            classes,
            scores,
            category_index=category_index,
            instance_masks=masks,
            keypoints=keypoints,
            track_ids=track_ids,
            **kwargs)

    return visualization_py_func_fn


def _resize_original_image(image, image_shape):
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_images(
        image,
        image_shape,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        align_corners=True)
    return tf.cast(tf.squeeze(image, 0), tf.uint8)


def draw_bounding_boxes_on_image_tensors(images,
                                         boxes,
                                         classes,
                                         scores,
                                         category_index,
                                         original_image_spatial_shape=None,
                                         true_image_shape=None,
                                         instance_masks=None,
                                         keypoints=None,
                                         keypoint_edges=None,
                                         track_ids=None,
                                         max_boxes_to_draw=20,
                                         min_score_thresh=0.2,
                                         use_normalized_coordinates=True):
    """Draws bounding boxes, masks, and keypoints on batch of image tensors.

    Args:
      images: A 4D uint8 image tensor of shape [N, H, W, C]. If C > 3, additional
        channels will be ignored. If C = 1, then we convert the images to RGB
        images.
      boxes: [N, max_detections, 4] float32 tensor of detection boxes.
      classes: [N, max_detections] int tensor of detection classes. Note that
        classes are 1-indexed.
      scores: [N, max_detections] float32 tensor of detection scores.
      category_index: a dict that maps integer ids to category dicts. e.g.
        {1: {1: 'dog'}, 2: {2: 'cat'}, ...}
      original_image_spatial_shape: [N, 2] tensor containing the spatial size of
        the original image.
      true_image_shape: [N, 3] tensor containing the spatial size of unpadded
        original_image.
      instance_masks: A 4D uint8 tensor of shape [N, max_detection, H, W] with
        instance masks.
      keypoints: A 4D float32 tensor of shape [N, max_detection, num_keypoints, 2]
        with keypoints.
      keypoint_edges: A list of tuples with keypoint indices that specify which
        keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
        edges from keypoint 0 to 1 and from keypoint 2 to 4.
      track_ids: [N, max_detections] int32 tensor of unique tracks ids (i.e.
        instance ids for each object). If provided, the color-coding of boxes is
        dictated by these ids, and not classes.
      max_boxes_to_draw: Maximum number of boxes to draw on an image. Default 20.
      min_score_thresh: Minimum score threshold for visualization. Default 0.2.
      use_normalized_coordinates: Whether to assume boxes and kepoints are in
        normalized coordinates (as opposed to absolute coordiantes). Default is
        True.

    Returns:
      4D image tensor of type uint8, with boxes drawn on top.
    """
    # Additional channels are being ignored.
    if images.shape[3] > 3:
        images = images[:, :, :, 0:3]
    elif images.shape[3] == 1:
        images = tf.image.grayscale_to_rgb(images)
    visualization_keyword_args = {
        'use_normalized_coordinates': use_normalized_coordinates,
        'max_boxes_to_draw': max_boxes_to_draw,
        'min_score_thresh': min_score_thresh,
        'agnostic_mode': False,
        'line_thickness': 4,
        'keypoint_edges': keypoint_edges
    }
    if true_image_shape is None:
        true_shapes = tf.constant(-1, shape=[images.shape.as_list()[0], 3])
    else:
        true_shapes = true_image_shape
    if original_image_spatial_shape is None:
        original_shapes = tf.constant(-1, shape=[images.shape.as_list()[0], 2])
    else:
        original_shapes = original_image_spatial_shape

    visualize_boxes_fn = create_visualization_fn(
        category_index,
        include_masks=instance_masks is not None,
        include_keypoints=keypoints is not None,
        include_track_ids=track_ids is not None,
        **visualization_keyword_args)

    elems = [true_shapes, original_shapes, images, boxes, classes, scores]
    if instance_masks is not None:
        elems.append(instance_masks)
    if keypoints is not None:
        elems.append(keypoints)
    if track_ids is not None:
        elems.append(track_ids)

    def draw_boxes(image_and_detections):
        """Draws boxes on image."""
        true_shape = image_and_detections[0]
        original_shape = image_and_detections[1]
        if true_image_shape is not None:
            image = shape_utils.pad_or_clip_nd(image_and_detections[2],
                                               [true_shape[0], true_shape[1], 3])
        if original_image_spatial_shape is not None:
            image_and_detections[2] = _resize_original_image(image, original_shape)

        image_with_boxes = tf.py_func(visualize_boxes_fn, image_and_detections[2:],
                                      tf.uint8)
        return image_with_boxes

    images = tf.map_fn(draw_boxes, elems, fn_output_signature=tf.uint8, back_prop=False)
    return images


def draw_side_by_side_evaluation_image(eval_dict,
                                       category_index,
                                       max_boxes_to_draw=20,
                                       min_score_thresh=0.2,
                                       use_normalized_coordinates=True,
                                       keypoint_edges=None):
    """Creates a side-by-side image with detections and groundtruth.

    Bounding boxes (and instance masks, if available) are visualized on both
    subimages.

    Args:
      eval_dict: The evaluation dictionary returned by
        eval_util.result_dict_for_batched_example() or
        eval_util.result_dict_for_single_example().
      category_index: A category index (dictionary) produced from a labelmap.
      max_boxes_to_draw: The maximum number of boxes to draw for detections.
      min_score_thresh: The minimum score threshold for showing detections.
      use_normalized_coordinates: Whether to assume boxes and keypoints are in
        normalized coordinates (as opposed to absolute coordinates). Default is
        True.
      keypoint_edges: A list of tuples with keypoint indices that specify which
        keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
        edges from keypoint 0 to 1 and from keypoint 2 to 4.

    Returns:
      A list of [1, H, 2 * W, C] uint8 tensor. The subimage on the left
        corresponds to detections, while the subimage on the right corresponds to
        groundtruth.
    """
    detection_fields = fields.DetectionResultFields()
    input_data_fields = fields.InputDataFields()

    images_with_detections_list = []

    # Add the batch dimension if the eval_dict is for single example.
    if len(eval_dict[detection_fields.detection_classes].shape) == 1:
        for key in eval_dict:
            if key != input_data_fields.original_image and key != input_data_fields.image_additional_channels:
                eval_dict[key] = tf.expand_dims(eval_dict[key], 0)

    for indx in range(eval_dict[input_data_fields.original_image].shape[0]):
        instance_masks = None
        if detection_fields.detection_masks in eval_dict:
            instance_masks = tf.cast(
                tf.expand_dims(
                    eval_dict[detection_fields.detection_masks][indx], axis=0),
                tf.uint8)
        keypoints = None
        if detection_fields.detection_keypoints in eval_dict:
            keypoints = tf.expand_dims(
                eval_dict[detection_fields.detection_keypoints][indx], axis=0)
        groundtruth_instance_masks = None
        if input_data_fields.groundtruth_instance_masks in eval_dict:
            groundtruth_instance_masks = tf.cast(
                tf.expand_dims(
                    eval_dict[input_data_fields.groundtruth_instance_masks][indx],
                    axis=0), tf.uint8)

        images_with_detections = draw_bounding_boxes_on_image_tensors(
            tf.expand_dims(
                eval_dict[input_data_fields.original_image][indx], axis=0),
            tf.expand_dims(
                eval_dict[detection_fields.detection_boxes][indx], axis=0),
            tf.expand_dims(
                eval_dict[detection_fields.detection_classes][indx], axis=0),
            tf.expand_dims(
                eval_dict[detection_fields.detection_scores][indx], axis=0),
            category_index,
            original_image_spatial_shape=tf.expand_dims(
                eval_dict[input_data_fields.original_image_spatial_shape][indx],
                axis=0),
            true_image_shape=tf.expand_dims(
                eval_dict[input_data_fields.true_image_shape][indx], axis=0),
            instance_masks=instance_masks,
            keypoints=keypoints,
            keypoint_edges=keypoint_edges,
            max_boxes_to_draw=max_boxes_to_draw,
            min_score_thresh=min_score_thresh,
            use_normalized_coordinates=use_normalized_coordinates)
        images_with_groundtruth = draw_bounding_boxes_on_image_tensors(
            tf.expand_dims(
                eval_dict[input_data_fields.original_image][indx], axis=0),
            tf.expand_dims(
                eval_dict[input_data_fields.groundtruth_boxes][indx], axis=0),
            tf.expand_dims(
                eval_dict[input_data_fields.groundtruth_classes][indx], axis=0),
            tf.expand_dims(
                tf.ones_like(
                    eval_dict[input_data_fields.groundtruth_classes][indx],
                    dtype=tf.float32),
                axis=0),
            category_index,
            original_image_spatial_shape=tf.expand_dims(
                eval_dict[input_data_fields.original_image_spatial_shape][indx],
                axis=0),
            true_image_shape=tf.expand_dims(
                eval_dict[input_data_fields.true_image_shape][indx], axis=0),
            instance_masks=groundtruth_instance_masks,
            keypoints=None,
            keypoint_edges=None,
            max_boxes_to_draw=None,
            min_score_thresh=0.0,
            use_normalized_coordinates=use_normalized_coordinates)
        images_to_visualize = tf.concat(
            [images_with_detections, images_with_groundtruth], axis=2)

        if input_data_fields.image_additional_channels in eval_dict:
            images_with_additional_channels_groundtruth = (
                draw_bounding_boxes_on_image_tensors(
                    tf.expand_dims(
                        eval_dict[input_data_fields.image_additional_channels][indx],
                        axis=0),
                    tf.expand_dims(
                        eval_dict[input_data_fields.groundtruth_boxes][indx], axis=0),
                    tf.expand_dims(
                        eval_dict[input_data_fields.groundtruth_classes][indx],
                        axis=0),
                    tf.expand_dims(
                        tf.ones_like(
                            eval_dict[input_data_fields.groundtruth_classes][indx],
                            dtype=tf.float32),
                        axis=0),
                    category_index,
                    original_image_spatial_shape=tf.expand_dims(
                        eval_dict[input_data_fields.original_image_spatial_shape]
                        [indx],
                        axis=0),
                    true_image_shape=tf.expand_dims(
                        eval_dict[input_data_fields.true_image_shape][indx], axis=0),
                    instance_masks=groundtruth_instance_masks,
                    keypoints=None,
                    keypoint_edges=None,
                    max_boxes_to_draw=None,
                    min_score_thresh=0.0,
                    use_normalized_coordinates=use_normalized_coordinates))
            images_to_visualize = tf.concat(
                [images_to_visualize, images_with_additional_channels_groundtruth],
                axis=2)
        images_with_detections_list.append(images_to_visualize)

    return images_with_detections_list


def draw_keypoints_on_image_array(image,
                                  keypoints,
                                  color='red',
                                  radius=2,
                                  use_normalized_coordinates=True,
                                  keypoint_edges=None,
                                  keypoint_edge_color='green',
                                  keypoint_edge_width=2):
    """Draws keypoints on an image (numpy array).

    Args:
      image: a numpy array with shape [height, width, 3].
      keypoints: a numpy array with shape [num_keypoints, 2].
      color: color to draw the keypoints with. Default is red.
      radius: keypoint radius. Default value is 2.
      use_normalized_coordinates: if True (default), treat keypoint values as
        relative to the image.  Otherwise treat them as absolute.
      keypoint_edges: A list of tuples with keypoint indices that specify which
        keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
        edges from keypoint 0 to 1 and from keypoint 2 to 4.
      keypoint_edge_color: color to draw the keypoint edges with. Default is red.
      keypoint_edge_width: width of the edges drawn between keypoints. Default
        value is 2.
    """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw_keypoints_on_image(image_pil, keypoints, color, radius,
                            use_normalized_coordinates, keypoint_edges,
                            keypoint_edge_color, keypoint_edge_width)
    np.copyto(image, np.array(image_pil))


def draw_keypoints_on_image(image,
                            keypoints,
                            color='red',
                            radius=2,
                            use_normalized_coordinates=True,
                            keypoint_edges=None,
                            keypoint_edge_color='green',
                            keypoint_edge_width=2):
    """Draws keypoints on an image.

    Args:
      image: a PIL.Image object.
      keypoints: a numpy array with shape [num_keypoints, 2].
      color: color to draw the keypoints with. Default is red.
      radius: keypoint radius. Default value is 2.
      use_normalized_coordinates: if True (default), treat keypoint values as
        relative to the image.  Otherwise treat them as absolute.
      keypoint_edges: A list of tuples with keypoint indices that specify which
        keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
        edges from keypoint 0 to 1 and from keypoint 2 to 4.
      keypoint_edge_color: color to draw the keypoint edges with. Default is red.
      keypoint_edge_width: width of the edges drawn between keypoints. Default
        value is 2.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    keypoints_x = [k[1] for k in keypoints]
    keypoints_y = [k[0] for k in keypoints]
    if use_normalized_coordinates:
        keypoints_x = tuple([im_width * x for x in keypoints_x])
        keypoints_y = tuple([im_height * y for y in keypoints_y])
    for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y):
        draw.ellipse([(keypoint_x - radius, keypoint_y - radius),
                      (keypoint_x + radius, keypoint_y + radius)],
                     outline=color,
                     fill=color)
    if keypoint_edges is not None:
        for keypoint_start, keypoint_end in keypoint_edges:
            if (keypoint_start < 0 or keypoint_start >= len(keypoints) or
                    keypoint_end < 0 or keypoint_end >= len(keypoints)):
                continue
            edge_coordinates = [
                keypoints_x[keypoint_start], keypoints_y[keypoint_start],
                keypoints_x[keypoint_end], keypoints_y[keypoint_end]
            ]
            draw.line(
                edge_coordinates, fill=keypoint_edge_color, width=keypoint_edge_width)


def draw_mask_on_image_array(image, mask, color='red', alpha=0.4):
    """Draws mask on an image.

    Args:
      image: uint8 numpy array with shape (img_height, img_height, 3)
      mask: a uint8 numpy array of shape (img_height, img_height) with values
        between either 0 or 1.
      color: color to draw the keypoints with. Default is red.
      alpha: transparency value between 0 and 1. (default: 0.4)

    Raises:
      ValueError: On incorrect data type for image or masks.
    """
    if image.dtype != np.uint8:
        raise ValueError('`image` not of type np.uint8')
    if mask.dtype != np.uint8:
        raise ValueError('`mask` not of type np.uint8')
    if np.any(np.logical_and(mask != 1, mask != 0)):
        raise ValueError('`mask` elements should be in [0, 1]')
    if image.shape[:2] != mask.shape:
        raise ValueError('The image has spatial dimensions %s but the mask has '
                         'dimensions %s' % (image.shape[:2], mask.shape))
    rgb = ImageColor.getrgb(color)
    pil_image = Image.fromarray(image)

    solid_color = np.expand_dims(
        np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
    pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
    pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert('L')
    pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
    np.copyto(image, np.array(pil_image.convert('RGB')))


def write_frames_to_videos(vid_writers, frames):
    if not isinstance(vid_writers, (list, tuple)):
        vid_writers = [vid_writers, ]
        frames = [frames, ]

    assert len(vid_writers) == len(frames), "vid_writers and frames must have matching lengths"

    for vid_writer, frame in zip(vid_writers, frames):
        if isinstance(vid_writer, cv2.VideoWriter):
            vid_writer.write(frame)
        elif isinstance(vid_writer, skvideo.io.FFmpegWriter):
            vid_writer.writeFrame(frame)
        else:
            raise AssertionError(f'invalid vid_writer type: {type(vid_writer)}')


def close_video_writers(vid_writers):
    if not isinstance(vid_writers, (list, tuple)):
        vid_writers = [vid_writers, ]

    for vid_writer in vid_writers:
        if vid_writer is None:
            continue
        if isinstance(vid_writer, cv2.VideoWriter):
            vid_writer.release()
        elif isinstance(vid_writer, skvideo.io.FFmpegWriter):
            vid_writer.close()
        else:
            raise AssertionError(f'invalid vid_writer type: {type(vid_writer)}')


def get_video_writer(vid_path, codec='mp4v', crf=0, fps=5, cv=False, shape=None):
    if cv:
        assert shape is not None, "shape must be provided for OpenCV video writer"

        # codec, ext = 'hfyu', 'avi'
        # codec, ext = 'mp4v', 'mp4'
        # codec, ext = 'mjpg', 'avi'

        fourcc = cv2.VideoWriter_fourcc(*codec)
        video_h, video_w = shape[:2]
        vid_writer = cv2.VideoWriter(vid_path, fourcc, fps, (video_w, video_h))
        if vid_writer is None:
            raise IOError(f'Output video file could not be opened: {vid_path}')
    else:
        inputdict = {
            '-r': str(fps),
        }

        outputdict = {
            '-r': str(fps),
            '-vcodec': 'libx264',
            '-crf': str(crf),
            '-preset': 'medium'
        }

        vid_writer = skvideo.io.FFmpegWriter(
            vid_path, inputdict=inputdict, outputdict=outputdict)

    return vid_writer


def visualize_mask(
        image_id,
        image,
        mask,
        gt_mask,
        category_index,
        out_mask_dir,
        out_vis_dir,
        img_ext='.jpg',
        vid_writers=None,
        csv_data=None,
        orig_size=None,
        video_id=None,
):
    seq_id = 'generic'
    if isinstance(image_id, bytes):
        image_id_ = image_id.decode('utf-8')
    else:
        image_id = image_id.astype(str)
        image_id_ = str(image_id.item())

    if '/' in image_id_:
        seq_id, image_id_ = image_id_.split('/')

    if not image_id_.endswith(img_ext):
        image_id_ += img_ext

    import cv2
    import eval_utils
    if video_id is not None:
        image_name_, image_ext_ = os.path.splitext(image_id_)
        vis_name = f'{image_name_}_{video_id}{image_ext_}'
    else:
        vis_name = image_id_

    image = cv2.resize(image, orig_size)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    vis_mask = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    vis_gt_mask = cv2.cvtColor(gt_mask * 255, cv2.COLOR_GRAY2BGR)

    pred_img = np.asarray(Image.blend(Image.fromarray(image), Image.fromarray(vis_mask), 0.5))
    gt_img = np.asarray(Image.blend(Image.fromarray(image), Image.fromarray(vis_gt_mask), 0.5))
    vis_img = np.concatenate((gt_img, pred_img), axis=1)
    vis_mask_all = np.concatenate((vis_gt_mask, vis_mask), axis=1)
    vis_img = np.concatenate((vis_img, vis_mask_all), axis=0)
    vis_img = cv2.resize(vis_img, (800, 800))

    vis_img = eval_utils.annotate(vis_img, vis_name)

    cv2.imshow('vis_img', vis_img)
    cv2.waitKey(100)

    if vid_writers is not None:
        if vid_writers[seq_id] is not None:
            mask_writer, vis_writer = vid_writers[seq_id]
        else:
            """close video writers for all other sequences assuming that images are 
            arranged sequence by sequence"""
            for seq_id_, writers in vid_writers.items():
                close_video_writers(writers)
                vid_writers[seq_id_] = None

            ext = 'mp4'

            seq_mask_path = os.path.join(out_mask_dir, f'{seq_id}.{ext}')
            seq_vis_path = os.path.join(out_vis_dir, f'{seq_id}.{ext}')

            mask_writer = get_video_writer(seq_mask_path)
            vis_writer = get_video_writer(seq_vis_path, crf=20)

            print(f'{seq_id} :: mask video: {seq_mask_path}')
            print(f'{seq_id} :: vis video: {seq_vis_path}')

            vid_writers[seq_id] = mask_writer, vis_writer

        write_frames_to_videos([mask_writer, vis_writer], (mask, vis_img))
    else:
        seq_mask_dir = os.path.join(out_mask_dir, seq_id)
        os.makedirs(seq_mask_dir, exist_ok=True)
        mask_path = os.path.join(seq_mask_dir, vis_name)
        cv2.imwrite(mask_path, mask)

        seq_vis_dir = os.path.join(out_vis_dir, seq_id)
        os.makedirs(seq_vis_dir, exist_ok=True)
        vis_path = os.path.join(seq_vis_dir, vis_name)
        cv2.imwrite(vis_path, blended_img)


def visualize_boxes_and_labels_on_image_array(
        image_id,
        image,
        bboxes_rescaled,
        boxes,
        classes,
        scores,
        category_index,
        img_ext='.jpg',
        vid_cap=None,
        out_vis_dir=None,
        csv_data=None,
        instance_masks=None,
        instance_boundaries=None,
        keypoints=None,
        keypoint_edges=None,
        track_ids=None,
        use_normalized_coordinates=False,
        max_boxes_to_draw=20,
        min_score_thresh=.5,
        agnostic_mode=False,
        line_thickness=4,
        groundtruth_box_visualization_color='black',
        skip_boxes=False,
        skip_scores=False,
        skip_labels=False,
        skip_track_ids=False,
        unpadded_size=None,
        orig_size=None,
):
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_rescaled_map = {}
    box_to_class_map = {}
    box_to_scores_map = {}
    box_to_instance_masks_map = {}
    box_to_instance_boundaries_map = {}
    box_to_keypoints_map = collections.defaultdict(list)
    box_to_track_ids_map = {}
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    """
    Collect supplementary information for each bounding box including instance mask, key points, track IDs, colour
    """
    for i in range(boxes.shape[0]):
        if max_boxes_to_draw == len(box_to_color_map):
            break
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if instance_boundaries is not None:
                box_to_instance_boundaries_map[box] = instance_boundaries[i]
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            if track_ids is not None:
                box_to_track_ids_map[box] = track_ids[i]

            if classes[i] in six.viewkeys(category_index):
                class_name = category_index[classes[i]]['name']
            else:
                class_name = 'N/A'
            box_to_class_map[box] = class_name
            box_to_rescaled_map[box] = tuple(bboxes_rescaled[i].tolist())

            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
                box_to_scores_map[box] = 1.0
            else:
                box_to_scores_map[box] = scores[i]
                display_str = ''
                if not skip_labels:
                    if not agnostic_mode:
                        display_str = str(class_name)
                if not skip_scores:
                    if not display_str:
                        display_str = '{}%'.format(int(100 * scores[i]))
                    else:
                        display_str = '{}: {}%'.format(display_str, int(100 * scores[i]))
                if not skip_track_ids and track_ids is not None:
                    if not display_str:
                        display_str = 'ID {}'.format(track_ids[i])
                    else:
                        display_str = '{}: ID {}'.format(display_str, track_ids[i])
                box_to_display_str_map[box].append(display_str)
                if agnostic_mode:
                    box_to_color_map[box] = 'DarkOrange'
                elif track_ids is not None:
                    prime_multipler = _get_multiplier_for_color_randomness()
                    box_to_color_map[box] = STANDARD_COLORS[(prime_multipler *
                                                             track_ids[i]) %
                                                            len(STANDARD_COLORS)]
                else:
                    box_to_color_map[box] = STANDARD_COLORS[classes[i] %
                                                            len(STANDARD_COLORS)]

    seq_id = 'generic'
    if isinstance(image_id, bytes):
        image_id_ = image_id.decode('utf-8')
    else:
        image_id = image_id.astype(str)
        image_id_ = str(image_id.item())

    if '/' in image_id_:
        seq_id, image_id_ = image_id_.split('/')

    if not image_id_.endswith(img_ext):
        image_id_ += img_ext

    # Draw all boxes onto image.
    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box

        if csv_data is not None:
            score = box_to_scores_map[box]
            label = box_to_class_map[box]

            box_rescaled = box_to_rescaled_map[box]
            ymin_, xmin_, ymax_, xmax_ = box_rescaled
            row = {
                "ImageID": image_id_,
                "LabelName": label,
                "XMin": xmin_,
                "XMax": xmax_,
                "YMin": ymin_,
                "YMax": ymax_,
                "Confidence": score,
            }
            csv_data[seq_id].append(row)

        if instance_masks is not None:
            draw_mask_on_image_array(
                image, box_to_instance_masks_map[box], color=color)

        if instance_boundaries is not None:
            draw_mask_on_image_array(
                image, box_to_instance_boundaries_map[box], color='red', alpha=1.0)

        draw_bounding_box_on_image_array(
            image,
            ymin,
            xmin,
            ymax,
            xmax,
            color=color,
            thickness=0 if skip_boxes else line_thickness,
            display_str_list=box_to_display_str_map[box],
            use_normalized_coordinates=use_normalized_coordinates)
        if keypoints is not None:
            draw_keypoints_on_image_array(
                image,
                box_to_keypoints_map[box],
                color=color,
                radius=line_thickness / 2,
                use_normalized_coordinates=use_normalized_coordinates,
                keypoint_edges=keypoint_edges,
                keypoint_edge_color=color,
                keypoint_edge_width=line_thickness // 2)

    if out_vis_dir:
        save_image(image, vid_cap, out_vis_dir, seq_id, image_id_,
                   unpadded_size=unpadded_size, orig_size=orig_size)

    return image


def visualize_image(config, examples, logits, tokens, label, category_names, mask, vis_out_dir):
    from tasks import task_utils

    images = examples['image']
    tconfig = config.task
    mconfig = config.model

    classes, bboxes, scores = task_utils.decode_object_seq_to_bbox(
        logits, tokens, tconfig.quantization_bins,
        mconfig.coord_vocab_shift)

    image_size = images.shape[1:3].as_list()
    scale = utils.tf_float32(image_size)

    bboxes_rescaled = utils.scale_points(bboxes, scale)
    image_ids = examples['image/id']

    image_ids_, bboxes_, bboxes_rescaled_, classes_, scores_, images_ = (
        image_ids.numpy(),
        bboxes.numpy(),
        bboxes_rescaled.numpy(),
        classes.numpy(),
        scores.numpy(),
        tf.image.convert_image_dtype(images, tf.uint8).numpy(),
    )
    vis_images = add_image_summary_with_bbox(
        images_, bboxes_, bboxes_rescaled_, classes_, scores_,
        category_names, image_ids_,
        min_score_thresh=0
    )
    import cv2
    for img_id, img in enumerate(vis_images):
        img_path = image_ids_[img_id].decode('utf-8')

        img_name = os.path.basename(img_path)
        seq_name = os.path.basename(os.path.dirname(img_path))

        vis_img_name = f'{label} {seq_name} {img_name}'

        import eval_utils
        img = eval_utils.annotate(img, vis_img_name)
        vis_img_path = os.path.join(vis_out_dir, f'{vis_img_name}.jpg')

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(vis_img_path, img)

    return bboxes_, bboxes_rescaled_, classes_, scores


def visualize_video(config, examples, logits, tokens, label, category_names, mask, vis_out_dir):
    from tasks import task_utils

    videos = examples['video']
    orig_video_size = examples['orig_video_size']
    unpadded_video_size = examples['unpadded_video_size']

    tconfig = config.task
    mconfig = config.model
    vid_len = config.dataset.length

    classes, bboxes, scores = task_utils.decode_video_seq_to_bbox(
        logits, tokens, vid_len, tconfig.quantization_bins,
        mconfig.coord_vocab_shift, mask)

    image_size = videos.shape[2:4].as_list()
    scale = utils.tf_float32(image_size)

    bboxes_rescaled = utils.scale_points(bboxes, scale)

    video_ids = examples['video_id'].numpy()
    file_ids = examples['file_ids'].numpy()
    file_names = examples['file_names'].numpy()

    bboxes_, bboxes_rescaled_, classes_, scores_, videos_ = (
        bboxes.numpy(),
        bboxes_rescaled.numpy(),
        classes.numpy(),
        scores.numpy(),
        tf.image.convert_image_dtype(videos, tf.uint8).numpy(),
    )
    videos_vis = add_video_summary_with_bbox(
        videos_, bboxes_, bboxes_rescaled_, classes_, scores_,
        vid_len=vid_len,
        filenames=file_names,
        file_ids=file_ids,
        video_ids=video_ids,
        category_names=category_names,
        orig_size=orig_video_size,
        unpadded_size=unpadded_video_size,
        min_score_thresh=0
    )
    import cv2
    for vid_id, video in enumerate(videos_vis):
        for img_id in range(vid_len):
            img = video[img_id, ...]

            img_path = file_names[vid_id][img_id].decode('utf-8')

            img_name = os.path.basename(img_path)
            seq_name = os.path.basename(os.path.dirname(img_path))

            vis_img_name = f'{label} {seq_name} {img_name}'

            import eval_utils
            img = eval_utils.annotate(img, vis_img_name)
            vis_img_path = os.path.join(vis_out_dir, vis_img_name)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cv2.imwrite(vis_img_path, img)

            # print()
    return bboxes_, bboxes_rescaled_, classes_, scores


def add_image_summary_with_bbox(
        images, bboxes, bboxes_rescaled, classes, scores, category_names,
        image_ids,
        unpadded_size,
        orig_size,
        vid_cap=None,
        out_vis_dir=None,
        csv_data=None,
        min_score_thresh=0.1,

):
    k = 0
    new_images = []
    for image_id_, image_, boxes_, bboxes_rescaled_, scores_, classes_, unpadded_size_, orig_size_ in zip(
            image_ids, images, bboxes, bboxes_rescaled, scores, classes, unpadded_size, orig_size):
        keep_indices = np.where(classes_ > 0)[0]

        image = visualize_boxes_and_labels_on_image_array(
            out_vis_dir=out_vis_dir,
            vid_cap=vid_cap,
            csv_data=csv_data,
            image_id=image_id_,
            image=image_,
            bboxes_rescaled=bboxes_rescaled_[keep_indices],
            boxes=boxes_[keep_indices],
            classes=classes_[keep_indices],
            scores=scores_[keep_indices],
            category_index=category_names,
            use_normalized_coordinates=True,
            min_score_thresh=min_score_thresh,
            max_boxes_to_draw=100,
            unpadded_size=unpadded_size_,
            orig_size=orig_size_,
        )
        new_images.append(image)

        # new_images.append(tf.image.convert_image_dtype(image, tf.float32))
        k += 1
        # if max_images_shown >= 0 and k >= max_images_shown:
        #     break
    # tf.summary.image(tag, new_images, step=step, max_outputs=max_images_shown)
    return new_images


def add_video_summary_with_bbox(
        videos, bboxes, bboxes_rescaled, classes, scores, category_names,
        video_ids, vid_len,
        filenames,
        file_ids,
        unpadded_size=None,
        orig_size=None,
        out_vis_dir=None,
        vid_cap=None,
        csv_data=None,
        min_score_thresh=0.1,

):
    k = 0
    vis_videos = []
    for iter_data in zip(
            video_ids, videos, filenames, file_ids, bboxes,
            bboxes_rescaled, scores, classes, unpadded_size, orig_size):
        (video_id_, video, filenames_, file_ids_, boxes_, bboxes_rescaled_,
         scores_, classes_, unpadded_size_, orig_size_) = iter_data

        keep_indices = np.where(classes_ > 0)[0]

        vis_video = visualize_boxes_and_labels_on_video(
            out_vis_dir=out_vis_dir,
            vid_cap=vid_cap,
            csv_data=csv_data,
            video_id=video_id_,
            video=video,
            file_names=filenames_,
            file_ids=file_ids_,
            vid_len=vid_len,
            bboxes_rescaled=bboxes_rescaled_[keep_indices],
            boxes=boxes_[keep_indices],
            classes=classes_[keep_indices],
            scores=scores_[keep_indices],
            category_index=category_names,
            use_normalized_coordinates=True,
            min_score_thresh=min_score_thresh,
            unpadded_size=unpadded_size_,
            orig_size=orig_size_,
            max_boxes_to_draw=100,

        )
        vis_videos.append(vis_video)
        k += 1
    return vis_videos


def visualize_boxes_and_labels_on_video(
        video_id,
        video,
        vid_len,
        file_names,
        file_ids,
        bboxes_rescaled,
        boxes,
        classes,
        scores,
        category_index,
        out_vis_dir=None,
        vid_cap=None,
        csv_data=None,
        use_normalized_coordinates=False,
        max_boxes_to_draw=20,
        min_score_thresh=.5,
        agnostic_mode=False,
        line_thickness=4,
        groundtruth_box_visualization_color='black',
        skip_boxes=False,
        skip_scores=False,
        skip_labels=False,
        unpadded_size=None,
        orig_size=None,
):
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_rescaled_map = {}
    box_to_class_map = {}
    box_to_scores_map = {}

    file_names = [str(filename.decode("utf-8")) for filename in file_names]

    seg_dir_path = os.path.dirname(file_names[0])
    seq_name = os.path.basename(seg_dir_path)

    video_id = video_id.astype(str)
    video_id_ = str(video_id.item())
    if '/' in video_id_:
        seq_id, video_id_ = video_id_.split('/')
    else:
        seq_id = seq_name

    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    """
    Collect supplementary information for each bounding box including colour, class
    """
    for box_id in range(boxes.shape[0]):
        if max_boxes_to_draw == len(box_to_color_map):
            break
        if scores is None or scores[box_id] > min_score_thresh:
            box = tuple(boxes[box_id].tolist())

            if classes[box_id] in six.viewkeys(category_index):
                class_name = category_index[classes[box_id]]['name']
            else:
                class_name = 'N/A'
            box_to_class_map[box] = class_name
            box_to_rescaled_map[box] = tuple(bboxes_rescaled[box_id].tolist())

            display_str = f'{video_id_}'
            if not skip_labels:
                if not agnostic_mode:
                    display_str = f'{display_str} {str(class_name)}'

            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
                box_to_scores_map[box] = 1.0
            else:
                box_to_scores_map[box] = scores[box_id]

                if not skip_scores:
                    conf = int(100 * scores[box_id])
                    display_str = f'{display_str}: {conf:d}%'

            box_to_display_str_map[box].append(display_str)
            if agnostic_mode:
                box_to_color_map[box] = 'DarkOrange'
            else:
                box_to_color_map[box] = STANDARD_COLORS[classes[box_id] %
                                                        len(STANDARD_COLORS)]

    video_vis = []
    for frame_id in range(vid_len):
        start_id = frame_id * 4
        image_vis = np.copy(video[frame_id, ...])

        image_path = str(file_names[frame_id])
        image_name = os.path.basename(image_path)

        image_id = f'{image_name}'
        seg_dir_path_ = os.path.dirname(str(file_names[frame_id]))

        assert seg_dir_path_ == seg_dir_path, f"seg_dir_path_ mismatch: {seg_dir_path}, {seg_dir_path_}"

        for box, color in box_to_color_map.items():
            box_rescaled = box_to_rescaled_map[box]
            score = box_to_scores_map[box]
            label = box_to_class_map[box]

            ymin, xmin, ymax, xmax = box[start_id:start_id + 4]

            if vocab.NO_BOX_FLOAT in [ymin, xmin, ymax, xmax]:
                continue
            # image_id = f'{seq_name}/{image_name}'

            if csv_data is not None:
                ymin_, xmin_, ymax_, xmax_ = box_rescaled[start_id:start_id + 4]
                row = {
                    "ImageID": image_id,
                    "VideoID": video_id_,
                    "LabelName": label,
                    "XMin": xmin_,
                    "XMax": xmax_,
                    "YMin": ymin_,
                    "YMax": ymax_,
                    "Confidence": score,
                }
                csv_data[seq_id].append(row)

            """box_rescaled not actually used for visualization which is why they seem correct in spite of not 
            cropping and resizing the 
            transformed image into the original shape"""
            image_vis = draw_bounding_box_on_image_array(
                image_vis,
                ymin,
                xmin,
                ymax,
                xmax,
                color=color,
                thickness=0 if skip_boxes else line_thickness,
                display_str_list=box_to_display_str_map[box],
                use_normalized_coordinates=use_normalized_coordinates)

        video_vis.append(image_vis)

        if out_vis_dir:
            save_image(image_vis, vid_cap, out_vis_dir, seq_id, image_name, video_id_,
                       unpadded_size=unpadded_size, orig_size=orig_size)

    video_vis = np.stack(video_vis, axis=0)
    return video_vis


def add_cdf_image_summary(values, name):
    """Adds a tf.summary.image for a CDF plot of the values.

    Normalizes `values` such that they sum to 1, plots the cumulative distribution
    function and creates a tf image summary.

    Args:
      values: a 1-D float32 tensor containing the values.
      name: name for the image summary.
    """
    _force_matplotlib_backend()

    def cdf_plot(values):
        """Numpy function to plot CDF."""
        normalized_values = values / np.sum(values)
        sorted_values = np.sort(normalized_values)
        cumulative_values = np.cumsum(sorted_values)
        fraction_of_examples = (
                np.arange(cumulative_values.size, dtype=np.float32) /
                cumulative_values.size)
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(fraction_of_examples, cumulative_values)
        ax.set_ylabel('cumulative normalized values')
        ax.set_xlabel('fraction of examples')
        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(
            fig.canvas.tostring_rgb(),
            dtype='uint8').reshape(1, int(height), int(width), 3)
        return image

    cdf_plot = tf.py_func(cdf_plot, [values], tf.uint8)
    tf.summary.image(name, cdf_plot)


def add_hist_image_summary(values, bins, name):
    """Adds a tf.summary.image for a histogram plot of the values.

    Plots the histogram of values and creates a tf image summary.

    Args:
      values: a 1-D float32 tensor containing the values.
      bins: bin edges which will be directly passed to np.histogram.
      name: name for the image summary.
    """
    _force_matplotlib_backend()

    def hist_plot(values, bins):
        """Numpy function to plot hist."""
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(1, 1, 1)
        y, x = np.histogram(values, bins=bins)
        ax.plot(x[:-1], y)
        ax.set_ylabel('count')
        ax.set_xlabel('value')
        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(
            fig.canvas.tostring_rgb(),
            dtype='uint8').reshape(1, int(height), int(width), 3)
        return image

    hist_plot = tf.py_func(hist_plot, [values, bins], tf.uint8)
    tf.summary.image(name, hist_plot)


class EvalMetricOpsVisualization(six.with_metaclass(abc.ABCMeta, object)):
    """Abstract base class responsible for visualizations during evaluation.

    Currently, summary images are not run during evaluation. One way to produce
    evaluation images in Tensorboard is to provide tf.summary.image strings as
    `value_ops` in tf.estimator.EstimatorSpec's `eval_metric_ops`. This class is
    responsible for accruing images (with overlaid detections and groundtruth)
    and returning a dictionary that can be passed to `eval_metric_ops`.
    """

    def __init__(self,
                 category_index,
                 max_examples_to_draw=5,
                 max_boxes_to_draw=20,
                 min_score_thresh=0.2,
                 use_normalized_coordinates=True,
                 summary_name_prefix='evaluation_image',
                 keypoint_edges=None):
        """Creates an EvalMetricOpsVisualization.

        Args:
          category_index: A category index (dictionary) produced from a labelmap.
          max_examples_to_draw: The maximum number of example summaries to produce.
          max_boxes_to_draw: The maximum number of boxes to draw for detections.
          min_score_thresh: The minimum score threshold for showing detections.
          use_normalized_coordinates: Whether to assume boxes and keypoints are in
            normalized coordinates (as opposed to absolute coordinates). Default is
            True.
          summary_name_prefix: A string prefix for each image summary.
          keypoint_edges: A list of tuples with keypoint indices that specify which
            keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
            edges from keypoint 0 to 1 and from keypoint 2 to 4.
        """

        self._category_index = category_index
        self._max_examples_to_draw = max_examples_to_draw
        self._max_boxes_to_draw = max_boxes_to_draw
        self._min_score_thresh = min_score_thresh
        self._use_normalized_coordinates = use_normalized_coordinates
        self._summary_name_prefix = summary_name_prefix
        self._keypoint_edges = keypoint_edges
        self._images = []

    def clear(self):
        self._images = []

    def add_images(self, images):
        """Store a list of images, each with shape [1, H, W, C]."""
        if len(self._images) >= self._max_examples_to_draw:
            return

        # Store images and clip list if necessary.
        self._images.extend(images)
        if len(self._images) > self._max_examples_to_draw:
            self._images[self._max_examples_to_draw:] = []

    def get_estimator_eval_metric_ops(self, eval_dict):
        # pyformat: disable
        """Returns metric ops for use in tf.estimator.EstimatorSpec.

        Args:
          eval_dict: A dictionary that holds an image, groundtruth, and detections
            for a batched example. Note that, we use only the first example for
            visualization. See eval_util.result_dict_for_batched_example() for a
            convenient method for constructing such a dictionary. The dictionary
            contains
            fields.InputDataFields.original_image: [batch_size, H, W, 3] image.
            fields.InputDataFields.original_image_spatial_shape: [batch_size, 2]
              tensor containing the size of the original image.
            fields.InputDataFields.true_image_shape: [batch_size, 3]
              tensor containing the spatial size of the upadded original image.
            fields.InputDataFields.groundtruth_boxes - [batch_size, num_boxes, 4]
              float32 tensor with groundtruth boxes in range [0.0, 1.0].
            fields.InputDataFields.groundtruth_classes - [batch_size, num_boxes]
              int64 tensor with 1-indexed groundtruth classes.
            fields.InputDataFields.groundtruth_instance_masks - (optional)
              [batch_size, num_boxes, H, W] int64 tensor with instance masks.
            fields.DetectionResultFields.detection_boxes - [batch_size,
              max_num_boxes, 4] float32 tensor with detection boxes in range [0.0,
              1.0].
            fields.DetectionResultFields.detection_classes - [batch_size,
              max_num_boxes] int64 tensor with 1-indexed detection classes.
            fields.DetectionResultFields.detection_scores - [batch_size,
              max_num_boxes] float32 tensor with detection scores.
            fields.DetectionResultFields.detection_masks - (optional) [batch_size,
              max_num_boxes, H, W] float32 tensor of binarized masks.
            fields.DetectionResultFields.detection_keypoints - (optional)
              [batch_size, max_num_boxes, num_keypoints, 2] float32 tensor with
              keypoints.

        Returns:
          A dictionary of image summary names to tuple of (value_op, update_op). The
          `update_op` is the same for all items in the dictionary, and is
          responsible for saving a single side-by-side image with detections and
          groundtruth. Each `value_op` holds the tf.summary.image string for a given
          image.
        """
        # pyformat: enable
        if self._max_examples_to_draw == 0:
            return {}
        images = self.images_from_evaluation_dict(eval_dict)

        def get_images():
            """Returns a list of images, padded to self._max_images_to_draw."""
            images = self._images
            while len(images) < self._max_examples_to_draw:
                images.append(np.array(0, dtype=np.uint8))
            self.clear()
            return images

        def image_summary_or_default_string(summary_name, image):
            """Returns image summaries for non-padded elements."""
            return tf.cond(
                tf.equal(tf.size(tf.shape(image)), 4),  # pyformat: disable
                lambda: tf.summary.image(summary_name, image),
                lambda: tf.constant(''))

        if tf.executing_eagerly():
            update_op = self.add_images([[images[0]]])  # pylint: disable=assignment-from-none
            image_tensors = get_images()
        else:
            update_op = tf.py_func(self.add_images, [[images[0]]], [])
            image_tensors = tf.py_func(get_images, [],
                                       [tf.uint8] * self._max_examples_to_draw)
        eval_metric_ops = {}
        for i, image in enumerate(image_tensors):
            summary_name = self._summary_name_prefix + '/' + str(i)
            value_op = image_summary_or_default_string(summary_name, image)
            eval_metric_ops[summary_name] = (value_op, update_op)
        return eval_metric_ops

    @abc.abstractmethod
    def images_from_evaluation_dict(self, eval_dict):
        """Converts evaluation dictionary into a list of image tensors.

        To be overridden by implementations.

        Args:
          eval_dict: A dictionary with all the necessary information for producing
            visualizations.

        Returns:
          A list of [1, H, W, C] uint8 tensors.
        """
        raise NotImplementedError


class VisualizeSingleFrameDetections(EvalMetricOpsVisualization):
    """Class responsible for single-frame object detection visualizations."""

    def __init__(self,
                 category_index,
                 max_examples_to_draw=5,
                 max_boxes_to_draw=20,
                 min_score_thresh=0.2,
                 use_normalized_coordinates=True,
                 summary_name_prefix='Detections_Left_Groundtruth_Right',
                 keypoint_edges=None):
        super(VisualizeSingleFrameDetections, self).__init__(
            category_index=category_index,
            max_examples_to_draw=max_examples_to_draw,
            max_boxes_to_draw=max_boxes_to_draw,
            min_score_thresh=min_score_thresh,
            use_normalized_coordinates=use_normalized_coordinates,
            summary_name_prefix=summary_name_prefix,
            keypoint_edges=keypoint_edges)

    def images_from_evaluation_dict(self, eval_dict):
        return draw_side_by_side_evaluation_image(eval_dict, self._category_index,
                                                  self._max_boxes_to_draw,
                                                  self._min_score_thresh,
                                                  self._use_normalized_coordinates,
                                                  self._keypoint_edges)
