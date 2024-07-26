import collections
import os
import sys
import pickle

import cv2
import numpy as np

dproc_path = os.path.join(os.path.expanduser("~"), "ipsc/ipsc_data_processing")

sys.path.append(os.getcwd())
sys.path.append(dproc_path)

# import numpy as np


import paramparse
from data.scripts import tfrecord_lib
from tasks.visualization import vis_utils

from eval_utils import col_bgr


class Params(paramparse.CFG):
    def __init__(self):
        paramparse.CFG.__init__(self, cfg_prefix='tf_vid')
        self.ann_file = ''
        self.ann_suffix = ''
        self.ann_ext = 'json.gz'

        self.frame_gaps = []
        self.length = 0
        self.stride = 0
        self.sample = 0

        self.add_stride_info = 1

        self.image_dir = ''
        self.n_proc = 0
        self.save_json = 0
        self.num_shards = 32
        self.output_dir = ''

        self.start_seq_id = 0
        self.end_seq_id = -1

        self.start_frame_id = 0
        self.end_frame_id = -1
        self.frame_stride = -1

        self.vis = 0


def save_ytvis_annotations(json_dict, json_path):
    print(f'saving ytvis annotations to {json_path}')
    json_kwargs = dict(
        indent=4
    )
    if json_path.endswith('.json.gz'):
        import compress_json
        compress_json.dump(json_dict, json_path, json_kwargs=json_kwargs)
    elif json_path.endswith('.json'):
        import json
        output_json = json.dumps(json_dict, **json_kwargs)
        with open(json_path, 'w') as f:
            f.write(output_json)
    else:
        raise AssertionError(f'Invalid json_path: {json_path}')


def load_ytvis_annotations(annotation_path, vid_id_offset):
    assert os.path.exists(annotation_path), f"ytvis json does not exist: {annotation_path}"

    print(f'Reading ytvis annotations from {annotation_path}')
    if annotation_path.endswith('.json'):
        import json
        with open(annotation_path, 'r') as f:
            json_data = json.load(f)
    elif annotation_path.endswith('.json.gz'):
        import compress_json
        json_data = compress_json.load(annotation_path)
    else:
        raise AssertionError(f'Invalid annotation_path: {annotation_path}')

    video_info = json_data['videos']
    annotations = json_data['annotations']
    categories = json_data['categories']

    if vid_id_offset > 0:
        for vid in video_info:
            vid["id"] += vid_id_offset

    category_id_to_name_map = dict(
        (category['id'], category['name']) for category in categories)

    # assert 0 not in category_id_to_name_map.keys(), "class IDs must to be > 0"

    try:
        bkg_class = category_id_to_name_map[0]
    except KeyError:
        pass
    else:
        assert bkg_class == 'background', "class id 0 can be used only for background"

    vid_to_ann = collections.defaultdict(list)
    for ann in annotations:
        ann['video_id'] += vid_id_offset
        vid_id = ann['video_id']
        vid_to_ann[vid_id].append(ann)

    return video_info, category_id_to_name_map, vid_to_ann, json_data


def show_img_rgb(frame, cu_z, mlab):
    from tvtk.api import tvtk

    h, w = frame.shape[:2]

    frame_gs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gs = np.transpose(frame_gs)
    mlab_im = mlab.imshow(frame_gs, extent=[0, w, 0, h, cu_z, cu_z], opacity=0.75)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    colors = tvtk.UnsignedCharArray()
    frame_ = frame
    # frame_ = frame_.transpose((1, 0, 2))
    colors.from_array(frame_.reshape(-1, 3))
    # mlab_im = mlab.imshow(np.ones(frame.shape[:2]),  extent=[0, w, 0, h, cu_z, cu_z])
    mlab_im.actor.input.point_data.scalars = colors


def to_rgb(col, norm=1):
    if isinstance(col, str):
        col = col_bgr[col]

    col_rgb = col[::-1]
    if norm:
        col_rgb = [k / 255. for k in col_rgb]
    return tuple(col_rgb)


def get_camera_view(mlab):
    azimuth, elevation, distance, focalpoint = mlab.view()
    roll = mlab.roll()

    return dict(
        azimuth=azimuth,
        elevation=elevation,
        distance=distance,
        focalpoint=focalpoint,
        roll=roll,
    )


def set_camera_view(mlab, params_dict):
    mlab.view(**params_dict)
    # roll = mlab.roll()


def show_vid_objs(video, image_dir, obj_annotations, id_to_name_map, fig):
    from mayavi import mlab

    file_ids = video['file_ids']
    file_names = video['file_names']
    file_paths = [os.path.join(image_dir, filename) for filename in file_names]

    file_names = [os.path.splitext(os.path.basename(file_path))[0] for file_path in file_paths]

    # z_gap = 500
    # title_font_size=48

    z_gap = 2000
    title_font_size = 64

    cols = (
        'green', 'red', 'deep_sky_blue',
        'yellow', 'forest_green', 'cyan',
        'magenta', 'purple', 'orange',
        'maroon', 'peach_puff', 'dark_orange',
        'slate_gray', 'pale_turquoise', 'green_yellow',
    )

    bkg_col = 'black'
    # bkg_col = 'white'
    frg_col = 'white' if bkg_col == 'black' else 'black'
    token_win_name = 'Token Sequence'

    bkg_col = col_bgr[bkg_col]
    frg_col = col_bgr[frg_col]

    bkg_col_rgb = to_rgb(bkg_col)
    frg_col_rgb = to_rgb(frg_col)

    frames = [cv2.imread(file_path) for file_path in file_paths]
    h, w = frames[0].shape[:2]

    if fig is None:
        fig = mlab.figure(
            'Video Clip',
            size=(1320, 1000),
            bgcolor=bkg_col_rgb
        )
        first_fig = True
        try:
            with open('view_params.pkl', 'rb') as f:
                view_params = pickle.load(f)
        except FileNotFoundError:
            view_params = None
        else:
            set_camera_view(mlab, view_params)
    else:
        mlab.clf(figure=fig)
        first_fig = False
        view_params = get_camera_view(mlab)

    # vid_len = len(file_paths)
    # z = (vid_len - 1) * z_gap
    # containing_box = (
    #     (0, 0, 0), (0, h, 0), (w, h, 0), (w, 0, 0), (0, 0, 0),
    #     (0, 0, z), (0, h, z), (w, h, z), (w, 0, z), (0, 0, z),
    #     (0, h, z), (0, h, 0), (w, h, 0), (w, h, z), (w, 0, z), (w, 0, 0),
    # )
    #
    # box_x = [k[0] for k in containing_box]
    # box_y = [k[1] for k in containing_box]
    # box_z = [k[2] for k in containing_box]

    # mlab.plot3d(box_x, box_y, box_z, color=(1, 1, 1), line_width=2.0, tube_radius=1.5)
    text_img = np.full(((1000, 600, 3)), bkg_col, dtype=np.uint8)

    for frame_id, frame in enumerate(frames):
        cu_z = (frame_id + 1) * z_gap

        file_id = int(file_ids[frame_id])
        file_name = file_names[frame_id]
        # mlab.text(w/2, 0, file_name, z=cu_z, figure=fig, color=frg_col_rgb, line_width=1.0)
        file_txt = f'image {file_id + 1}'

        frame, _, _ = vis_utils.write_text(frame, file_txt, 5, 5, frg_col, font_size=title_font_size)

        show_img_rgb(frame, cu_z, mlab)

    # mlab.axes(figure=fig, color=frg_col_rgb,
    #           x_axis_visibility=True,
    #           y_axis_visibility=True,
    #           z_axis_visibility=True,
    #           extent=[0, w, 0, h, 0, cu_z],
    #           # ranges=[0, w, 0, h, cu_z, cu_z]
    #           )

    # mlab.draw()
    # mlab.show()

    if first_fig:
        cv2.imshow(token_win_name, text_img)
        k = cv2.waitKey(0)
        if k == 27:
            exit()
        view_params = get_camera_view(mlab)
        with open('view_params.pkl', 'wb') as f:
            pickle.dump(view_params, f, pickle.HIGHEST_PROTOCOL)
    else:
        if view_params is not None:
            set_camera_view(mlab, view_params)

    text_x = text_y = 5

    for ann_id, ann in enumerate(obj_annotations):
        bboxes = ann['bboxes']
        col_id = ann_id % len(cols)
        col = col_bgr[cols[col_id]]

        prev_bbox = None

        col_rgb = to_rgb(col)

        for bbox_id, bbox in enumerate(bboxes):
            if bbox is not None:

                (x, y, width, height) = tuple(bbox)
                xmin, ymin, xmax, ymax = int(x), int(y), int(x + width), int(y + height)

                assert ymax > ymin and xmax > xmin, f"invalid bbox: {bbox}"

                cu_z = (bbox_id + 1) * z_gap

                xs = [xmin, xmin, xmax, xmax, xmin]
                ys = [ymin, ymax, ymax, ymin, ymin]
                zs = [cu_z, ] * len(xs)

                mlab.plot3d(xs, ys, zs, color=col_rgb, line_width=3.0, tube_radius=3.0)
                set_camera_view(mlab, view_params)

                if prev_bbox is not None:
                    xmin_, ymin_, xmax_, ymax_, pre_z = prev_bbox
                    xs_ = [xmin, xmin_, xmin_, xmin, xmax, xmax_, xmax_, xmax]
                    ys_ = [ymin, ymin_, ymax_, ymax, ymax, ymax_, ymin_, ymin]
                    zs_ = [cu_z, pre_z, pre_z, cu_z, cu_z, pre_z, pre_z, cu_z]
                    mlab.plot3d(xs_, ys_, zs_, color=col_rgb, line_width=2.0, tube_radius=2.0)
                    set_camera_view(mlab, view_params)

                # bbox_txts.append(bbox_txt)
                prev_bbox = [xmin, ymin, xmax, ymax, cu_z]

                bbox_txt = f'{int(xmin)}, {int(ymin)}, {int(xmax)}, {int(ymax)}, '
            else:
                prev_bbox = None
                bbox_txt = f'NA, NA, NA, NA, '

            text_img, text_x, text_y = vis_utils.write_text(
                text_img, bbox_txt, text_x, text_y, col, show=1, win_name=token_win_name)

            k = cv2.waitKey(100)
            if k == 27:
                sys.exit()

        class_id = int(ann['category_id'])
        class_name = id_to_name_map[class_id]

        text_img, text_x, text_y = vis_utils.write_text(text_img, f'{class_name}, ', text_x, text_y, col,
                                                        show=1, win_name=token_win_name)

        k = cv2.waitKey(500)
        if k == 27:
            sys.exit()

    text_img, text_x, text_y = vis_utils.write_text(text_img, 'EOS', text_x, text_y, frg_col,
                                                    show=1, win_name=token_win_name)

    k = cv2.waitKey(0)
    if k == 27:
        sys.exit()

    return fig


def ytvis_annotations_to_lists(obj_annotations: dict, id_to_name_map: dict, vid_len: int):
    """
    Converts YTVIS annotations to feature lists.
    """

    data = dict((k, list()) for k in [
        'is_crowd', 'category_id',
        'category_names', 'target_id'])

    for _id in range(vid_len):
        frame_data = dict((k, list()) for k in [
            f'xmin-{_id}', f'xmax-{_id}', f'ymin-{_id}', f'ymax-{_id}', f'area-{_id}'
        ])
        data.update(frame_data)

    for ann_id, ann in enumerate(obj_annotations):
        bboxes = ann['bboxes']
        areas = ann['areas']
        assert len(bboxes) == vid_len, \
            f"number of boxes: {len(bboxes)} does not match the video length: {vid_len}"
        assert len(areas) == vid_len, \
            f"number of areas: {len(areas)} does not match the video length: {vid_len}"

        data['is_crowd'].append(ann['iscrowd'])
        data['target_id'].append(int(ann['id']))
        category_id = int(ann['category_id'])
        data['category_id'].append(category_id)
        data['category_names'].append(id_to_name_map[category_id].encode('utf8'))

        valid_box_exists = False

        for bbox_id, bbox in enumerate(bboxes):
            area = areas[bbox_id]

            if bbox is None:
                assert area is None, "null bbox for non-null area"
                # xmin, ymin, xmax, ymax = -1, -1, -1, -1
                # area = -1
                # xmin = ymin = xmax = ymax = area = -1
                xmin = ymin = xmax = ymax = area = None
            else:
                assert area is not None, "null area for non-null bbox"
                (x, y, width, height) = tuple(bbox)
                xmin, ymin, xmax, ymax = float(x), float(y), float(x + width), float(y + height)

                assert ymax > ymin and xmax > xmin, f"invalid bbox: {bbox}"

                valid_box_exists = True

            # if bbox_id == 0:
            #     xmin = ymin = xmax = ymax = area = None

            data[f'xmin-{bbox_id}'].append(xmin)
            data[f'xmax-{bbox_id}'].append(xmax)
            data[f'ymin-{bbox_id}'].append(ymin)
            data[f'ymax-{bbox_id}'].append(ymax)
            data[f'area-{bbox_id}'].append(area)

        if not valid_box_exists:
            raise AssertionError('all boxes for an object cannot be None')

    return data


def obj_annotations_to_feature_dict(obj_annotations, id_to_name_map, vid_len):
    """Convert COCO annotations to an encoded feature dict.

    Args:
      obj_annotations: a list of object annotations.
      id_to_name_map: category id to category name map.

    Returns:
      a dict of tf features.
    """

    data = ytvis_annotations_to_lists(obj_annotations, id_to_name_map, vid_len)
    feature_dict = {
        'video/object/class/text':
            tfrecord_lib.convert_to_feature(data['category_names']),
        'video/object/class/label':
            tfrecord_lib.convert_to_feature(data['category_id']),
        'video/object/is_crowd':
            tfrecord_lib.convert_to_feature(data['is_crowd']),
        'video/object/target_id':
            tfrecord_lib.convert_to_feature(data['target_id']),
    }

    for _id in range(vid_len):
        frame_feature_dict = {
            f'video/object/bbox/xmin-{_id}':
                tfrecord_lib.convert_to_feature(data[f'xmin-{_id}']),
            f'video/object/bbox/xmax-{_id}':
                tfrecord_lib.convert_to_feature(data[f'xmax-{_id}']),
            f'video/object/bbox/ymin-{_id}':
                tfrecord_lib.convert_to_feature(data[f'ymin-{_id}']),
            f'video/object/bbox/ymax-{_id}':
                tfrecord_lib.convert_to_feature(data[f'ymax-{_id}']),
            f'video/object/area-{_id}':
                tfrecord_lib.convert_to_feature(data[f'area-{_id}'], value_type='float_list'),
        }
        feature_dict.update(frame_feature_dict)

    return feature_dict


def generate_video_annotations(
        params: Params,
        videos,
        category_id_to_name_map,
        vid_to_obj_ann,
        image_dir,

):
    fig = None
    for video in videos:
        object_anns = vid_to_obj_ann.get(video['id'], {})
        if params.vis == 1:
            from tasks.visualization import vis_utils
            vis_utils.vis_json_ann(video, object_anns, category_id_to_name_map, image_dir)

        if params.vis == 2:
            fig = show_vid_objs(video, image_dir, object_anns, category_id_to_name_map, fig)
        else:
            yield (
                video,
                category_id_to_name_map,
                object_anns,
                image_dir,
            )


def create_video_tf_example(
        video,
        category_id_to_name_map,
        object_ann,
        image_dir,
):
    file_ids = video['file_ids']

    assert all(i < j for i, j in zip(file_ids, file_ids[1:])), \
        "file_ids should be strictly increasing"

    video_height = video['height']
    video_width = video['width']
    file_names = video['file_names']
    video_id = video['id']

    vid_len = len(file_names)

    file_paths = [os.path.join(image_dir, filename) for filename in file_names]
    # file_paths = [os.path.realpath(file_path) for file_path in file_paths]

    feature_dict = tfrecord_lib.video_info_to_feature_dict(
        video_height, video_width, file_ids, video_id, file_paths)

    if object_ann:
        # Bbox, area, etc.
        obj_feature_dict = obj_annotations_to_feature_dict(
            object_ann,
            category_id_to_name_map,
            vid_len)
        feature_dict.update(obj_feature_dict)

    import tensorflow as tf
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example, 0


def main():
    params = Params()
    paramparse.process(params)

    assert params.image_dir, "image_dir must be provided"
    assert params.ann_file, "ann_file must be provided"

    assert os.path.exists(params.image_dir), f"image_dir does not exist: {params.image_dir}"

    if params.start_frame_id > 0 or params.end_frame_id >= 0 or params.frame_stride > 1:
        frame_suffix = f'{params.start_frame_id}_{params.end_frame_id}'
        if params.frame_stride > 1:
            frame_suffix = f'{frame_suffix}_{params.frame_stride}'
        params.ann_file = f'{params.ann_file}-{frame_suffix}'

    if params.length:
        params.ann_file = f'{params.ann_file}-length-{params.length}'

    if params.stride:
        params.ann_file = f'{params.ann_file}-stride-{params.stride}'

    if params.sample:
        params.ann_file = f'{params.ann_file}-sample-{params.sample}'

    if params.frame_gaps:
        ann_files = [f'{params.ann_file}-frame_gap-{frame_gap}' if frame_gap > 1 else params.ann_file
                     for frame_gap in params.frame_gaps]
    else:
        ann_files = [params.ann_file, ]

    if params.ann_suffix:
        ann_files = [f'{ann_file}-{params.ann_suffix}' for ann_file in ann_files]

    if params.start_seq_id > 0 or params.end_seq_id >= 0:
        assert params.end_seq_id >= params.start_seq_id, "end_seq_id must to be >= start_seq_id"
        ann_files = [f'{ann_file}-seq-{params.start_seq_id}_{params.end_seq_id}' for ann_file in ann_files]

    # params.ann_file = None

    ann_files = [os.path.join(params.image_dir, 'ytvis19', f'{ann_file}.{params.ann_ext}') for ann_file in ann_files]
    vid_id_offset = 0
    n_all_vid = 0
    n_all_ann = 0
    video_info = []
    annotations_all = {}
    category_id_to_name_map = {}
    vid_to_obj_ann = collections.defaultdict(list)

    stride_to_video_ids = {}

    for ann_file in ann_files:
        video_info_, category_id_to_name_map_, vid_to_obj_ann_, annotations_ = load_ytvis_annotations(
            ann_file, vid_id_offset)

        if params.add_stride_info:
            filenames_to_vid_id = dict(
                (tuple(video_['file_names']), video_['id']) for video_ in video_info_
            )
            for _stride in range(params.stride + 1, params.length + 1):
                _stride_ann_file = ann_file.replace(f'stride-{params.stride}', f'stride-{_stride}')
                _stride_video_info_, _, _, _ = load_ytvis_annotations(_stride_ann_file, vid_id_offset=0)
                stride_to_video_ids[_stride] = [filenames_to_vid_id[tuple(video_['file_names'])]
                                                for video_ in _stride_video_info_]

                print()
            annotations_['stride_to_video_ids'] = stride_to_video_ids

        new_vid_ids = set(vid_to_obj_ann_.keys())
        n_new_vid_ids = len(new_vid_ids)
        counts_ = annotations_['info']['counts'][0]
        n_new_vids = len(annotations_['videos'])
        n_new_anns = len(annotations_['annotations'])

        assert n_new_vids == counts_['videos'], "videos counts mismatch"
        assert n_new_vids == n_new_vid_ids, "n_new_vid_ids mismatch"
        assert n_new_anns == counts_['annotations'], "annotations counts mismatch"

        existing_vid_ids = set(vid_to_obj_ann.keys())
        overlapped_vid_ids = existing_vid_ids.intersection(new_vid_ids)

        assert not overlapped_vid_ids, f"overlapped_vid_ids found: {overlapped_vid_ids}"

        video_info += video_info_
        vid_to_obj_ann.update(vid_to_obj_ann_)
        category_id_to_name_map.update(category_id_to_name_map_)

        vid_id_offset = max(vid_to_obj_ann.keys())

        if not params.save_json:
            continue

        if not annotations_all:
            annotations_all = annotations_
        else:
            annotations_all['videos'] += annotations_['videos']
            annotations_all['annotations'] += annotations_['annotations']
            # annotations_all['categories'] += annotations_['categories']
            if params.add_stride_info:
                annotations_all['stride_to_video_ids'].update(annotations_['stride_to_video_ids'])

            counts = annotations_all['info']['counts'][0]

            counts['videos'] += n_new_vids
            counts['annotations'] += n_new_anns

        n_all_vid += n_new_vids
        n_all_ann += n_new_anns

        counts = annotations_all['info']['counts'][0]

        assert n_all_vid == counts['videos'], "n_all_vid mismatch"
        assert n_all_ann == counts['annotations'], "n_all_ann mismatch"

        assert n_all_vid == len(annotations_all['videos']), "annotations_all videos mismatch"
        assert n_all_ann == len(annotations_all['annotations']), "annotations_all annotations mismatch"

    # categories_all = annotations_all['categories']
    # categories_unique = [dict(t) for t in {tuple(d.items()) for d in categories_all}]
    # annotations_all['categories'] = categories_unique

    if not params.output_dir:
        output_dir = os.path.join(params.image_dir, 'ytvis19', 'tfrecord')
        print(f'setting output_dir to {output_dir}')
        params.output_dir = output_dir

    os.makedirs(params.output_dir, exist_ok=True)

    out_name = os.path.basename(ann_files[0]).split(os.extsep)[0]

    if len(params.frame_gaps) > 1:
        frame_gaps_suffix = 'fg_' + '_'.join(map(str, params.frame_gaps))
        if frame_gaps_suffix not in out_name:
            out_name = f'{out_name}-{frame_gaps_suffix}'

    if params.save_json:
        out_json_path = os.path.join(params.image_dir, 'ytvis19', f'{out_name}.{params.ann_ext}')
        annotations_all['info']['description'] = out_name
        save_ytvis_annotations(annotations_all, out_json_path)

    if params.save_json != 2:
        annotations_iter = generate_video_annotations(
            params=params,
            videos=video_info,
            category_id_to_name_map=category_id_to_name_map,
            vid_to_obj_ann=vid_to_obj_ann,
            image_dir=params.image_dir,
        )
        output_path = os.path.join(params.output_dir, out_name)
        os.makedirs(output_path, exist_ok=True)

        tfrecord_pattern = os.path.join(output_path, 'shard')

        tfrecord_lib.write_tf_record_dataset(
            output_path=tfrecord_pattern,
            annotation_iterator=annotations_iter,
            process_func=create_video_tf_example,
            num_shards=params.num_shards,
            multiple_processes=params.n_proc,
            iter_len=len(video_info),
        )
        print(f'output_path: {output_path}')

    print(f'out_name: {out_name}')


if __name__ == '__main__':
    main()
