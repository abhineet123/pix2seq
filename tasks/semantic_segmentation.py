import ml_collections
import numpy as np
import utils
import vocab
from tasks import task as task_lib
from tasks import task_utils
from tasks.visualization import vis_utils
import tensorflow as tf


@task_lib.TaskRegistry.register('semantic_segmentation')
class TaskSemanticSegmentation(task_lib.Task):
    def __init__(self,
                 config: ml_collections.ConfigDict):
        super().__init__(config)

        json_dict = self.config.dataset.json_dict

        self._category_names = task_utils.get_category_names(json_dict)

        self.frame_id_to_img_info = {
            img['frame_id']: img for img in json_dict['images']
        }

        class_id_to_col, class_id_to_name = task_utils.get_class_info(self._category_names)

        # class_info_path = config.dataset.get('class_info_path')
        # assert class_info_path, "class_info_path must be provided"
        # class_id_to_col, class_id_to_name = task_utils.read_class_info(
        #     class_info_path)

        # n_classes = len(self._category_names)
        # class_names_from_json = tuple(self._category_names[i]['name'] for i in range(n_classes))
        # assert class_names_from_json == class_names, "class_names mismatch"

        self.class_id_to_col = class_id_to_col
        self.class_id_to_name = class_id_to_name

    def preprocess_single(self, dataset, batch_duplicates, training, validation):
        if self.config.debug != 2:
            """apply transforms"""
            dataset = dataset.map(
                lambda x: self.preprocess_single_example(
                    x, training, validation, batch_duplicates),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return dataset

    def check_rle(self, mask_vid_paths, images, img_ids, frame_ids, rles, training):

        if isinstance(mask_vid_paths, str):
            images = np.expand_dims(images, axis=0)
            img_ids = np.expand_dims(img_ids, axis=0)
            frame_ids = np.expand_dims(frame_ids, axis=0)
            rles = np.expand_dims(rles, axis=0)

            mask_vid_paths = [mask_vid_paths, ]

        batch_size = frame_ids.shape[0]
        if training:
            mode_cfg = self.config.dataset.train
        else:
            mode_cfg = self.config.dataset.eval

        max_length = mode_cfg.max_length
        subsample = mode_cfg.subsample

        assert max_length > 0, "max_length must be > 0"
        assert subsample >= 1, "subsample must be >= 1"

        multi_class = self.config.dataset.multi_class
        class_id_to_col = self.class_id_to_col

        n_classes = len(self.class_id_to_col)

        if not multi_class:
            assert n_classes == 2, "n_classes must be 2 for no multi_class"
        else:
            assert n_classes > 2, "n_classes must be > 2 for multi_class"

        flat_order = self.config.dataset.flat_order
        length_as_class = self.config.dataset.length_as_class

        starts_offset = self.config.model.coord_vocab_shift
        lengths_offset = self.config.model.len_vocab_shift
        class_offset = self.config.model.class_vocab_shift

        if length_as_class:
            n_lac_classes = int(max_length / subsample) * (n_classes - 1)
            if starts_offset < n_lac_classes:
                print(f'setting starts_offset to {n_lac_classes}')
                starts_offset = n_lac_classes
        masks = []
        masks_sub = []
        for batch_id in range(batch_size):
            mask_vid_path = mask_vid_paths[batch_id].decode('utf-8')
            rle = rles[batch_id]
            img_id = img_ids[batch_id]
            image = images[batch_id]
            frame_id = frame_ids[batch_id]
            vid_reader, vid_width, vid_height, num_frames = task_utils.load_video(mask_vid_path)

            n_rows, n_cols = vid_height, vid_width

            mask = task_utils.read_frame(vid_reader, frame_id - 1, mask_vid_path)
            if not multi_class:
                mask = task_utils.mask_to_binary(mask)
            mask = task_utils.mask_to_gs(mask)

            if subsample > 1:
                n_rows_sub, n_cols_sub = int(n_rows / subsample), int(n_cols / subsample)
                mask_sub = task_utils.resize_mask_coord(mask, (n_rows_sub, n_cols_sub), n_classes, is_vis=1)
            else:
                mask_sub = np.copy(mask)

            masks.append(mask)
            masks_sub.append(mask_sub)

            rle_stripped = rle[rle != vocab.PADDING_TOKEN]
            if rle_stripped.size == 0:
                print(f'\n{img_id}: mask is empty\n')

            task_utils.check_rle_tokens(
                image, mask, mask_sub,
                rle_stripped,
                n_classes=n_classes,
                length_as_class=length_as_class,
                starts_offset=starts_offset,
                lengths_offset=lengths_offset,
                class_offset=class_offset,
                max_length=max_length,
                subsample=subsample,
                class_to_col=class_id_to_col,
                multi_class=multi_class,
                flat_order=flat_order,
                is_vis=True,
            )
        return masks, masks_sub

    def preprocess_batched(self, batched_examples, training):
        config = self.config.task
        if self.config.debug == 2:
            batched_examples = vis_utils.debug_image_pipeline(
                self.dataset,
                self.train_transforms,
                batched_examples,
                vis=1,
                model_dir=self.config.model_dir,
                training=training)

        response_seq = batched_examples['rle']
        token_weights = tf.ones_like(response_seq, dtype=tf.float32)

        if self.config.debug:
            mask_vid_paths = batched_examples['mask_vid_path'].numpy()
            images = batched_examples['image'].numpy()
            img_ids = batched_examples['image/id'].numpy()
            frame_ids = batched_examples['frame_id'].numpy()
            rles = batched_examples['rle'].numpy()

            self.check_rle(mask_vid_paths, images, img_ids, frame_ids, rles, training=True)

        # response_seq, token_weights = build_response_seq_from_rle(
        #     batched_examples['rle'],
        #     config.starts_bins,
        #     config.lengths_bins,
        #     mconfig.coord_vocab_shift,
        # )

        prompt_seq = task_utils.build_prompt_seq_from_task_id(
            task_vocab_id=self.task_vocab_id,
            response_seq=response_seq)  # (bsz, 1)
        input_seq = tf.concat([prompt_seq, response_seq], -1)
        target_seq = tf.concat([prompt_seq, response_seq], -1)

        """rle seq is already padded"""
        # input_seq = utils.pad_to_max_len(input_seq, config.max_seq_len + 1,
        #                                  dim=-1, padding_token=vocab.PADDING_TOKEN)
        # target_seq = utils.pad_to_max_len(target_seq, config.max_seq_len + 1,
        #                                   dim=-1, padding_token=vocab.PADDING_TOKEN)

        """
        right shift the target_seq and left-shift the input_seq
        """
        input_seq, target_seq = input_seq[..., :-1], target_seq[..., 1:]

        """
        token_weights should already be config.max_seq_len since it is created from response_seq
        """
        # token_weights = utils.pad_to_max_len(token_weights, config.max_seq_len,
        #                                      dim=-1, padding_token=vocab.PADDING_TOKEN)

        """
        Assign lower weights for ending/padding tokens.
        eos_token_weight = 0.1
        """
        token_weights = tf.where(
            target_seq == vocab.PADDING_TOKEN,
            tf.zeros_like(token_weights) + config.eos_token_weight,
            token_weights)

        return batched_examples, input_seq, target_seq, token_weights

    def infer(self, model, preprocessed_outputs):
        """Perform inference given the model and preprocessed outputs."""
        config = self.config.task
        mconfig = self.config.model
        examples, input_seq, target_seq, token_weights = preprocessed_outputs
        image = examples["image"]
        bsz = tf.shape(image)[0]
        prompt_seq = task_utils.build_prompt_seq_from_task_id(
            self.task_vocab_id, prompt_shape=(bsz, 1))
        pred_seq, logits, _ = model.infer(
            image, prompt_seq, encoded=None,
            max_seq_len=mconfig.max_seq_len + 1,
            temperature=config.temperature, top_k=config.top_k, top_p=config.top_p)

        return examples, pred_seq, logits

    def postprocess_tpu(self, batched_examples, pred_rle, logits, training=False):
        example = batched_examples
        images, image_ids = example['image'], example['image/id']
        orig_image_size = example['orig_image_size']

        gt_rle = example['rle']

        return images, image_ids, pred_rle, logits, gt_rle, orig_image_size

    def postprocess_cpu(self,
                        outputs,
                        train_step,
                        out_vis_dir,
                        out_mask_dir,
                        out_mask_logits_dir,
                        json_vid_info=None,
                        vid_cap=None,
                        eval_step=None,
                        training=False,
                        show=False,
                        summary_tag='eval',
                        ret_results=False,
                        **kwargs
                        ):

        # Copy outputs to cpu.
        new_outputs = []
        for i in range(len(outputs)):
            new_outputs.append(
                tf.identity(outputs[i]).numpy()
            )

        images, image_ids, frame_ids, rles, logits, gt_rles, orig_sizes, \
            vid_paths, mask_vid_paths, seqs = new_outputs

        image_ids = image_ids.flatten().astype(str)
        vid_paths = vid_paths.flatten().astype(str)
        mask_vid_paths = mask_vid_paths.flatten().astype(str)
        seqs = seqs.flatten().astype(str)

        images = np.copy(tf.image.convert_image_dtype(images, tf.uint8))

        max_length = self.config.dataset.train.max_length
        subsample = self.config.dataset.train.subsample
        multi_class = self.config.dataset.multi_class

        length_as_class = self.config.dataset.length_as_class
        flat_order = self.config.dataset.flat_order

        starts_offset = self.config.model.coord_vocab_shift
        lengths_offset = self.config.model.len_vocab_shift
        class_offset = self.config.model.class_vocab_shift

        n_classes = len(self.class_id_to_col)
        max_seq_len = self.config.model.max_seq_len
        vocab_size = self.config.model.vocab_size

        if subsample > 1:
            max_length = int(max_length / subsample)

        for (image_id_, image_, frame_id_, rle_, logits_,
             orig_size_, gt_rle_, seq, vid_path, mask_vid_path) in zip(
            image_ids, images, frame_ids, rles, logits,
            orig_sizes, gt_rles, seqs, vid_paths, mask_vid_paths):

            orig_size_ = tuple(orig_size_)
            n_rows, n_cols = orig_size_

            if subsample > 1:
                n_rows, n_cols = int(n_rows / subsample), int(n_cols / subsample)

            rle_ = rle_[rle_ != vocab.PADDING_TOKEN]

            mask_from_file = mask_from_file_sub = None
            if self.config.debug:
                vid_masks, vid_masks_sub = self.check_rle(
                    mask_vid_path, image_, image_id_, frame_id_, gt_rle_, training=False)
                mask_from_file = vid_masks[0]
                mask_from_file_sub = vid_masks_sub[0]

            mask_logits, rle_logits_cmp = task_utils.mask_from_logits(
                rle_logits=logits_,
                shape=(n_rows, n_cols),
                max_length=max_length,
                n_classes=n_classes,
                starts_offset=starts_offset,
                lengths_offset=lengths_offset,
                class_offset=class_offset,
                length_as_class=length_as_class,
                starts_2d=False,
                multi_class=multi_class,
                max_seq_len=max_seq_len,
                vocab_size=vocab_size,
            )

            mask_rec, rle_rec_cmp = task_utils.mask_from_tokens(
                rle_,
                (n_rows, n_cols),
                starts_offset=starts_offset,
                lengths_offset=lengths_offset,
                class_offset=class_offset,
                starts_2d=False,
                multi_class=multi_class,
                max_length=max_length,
                length_as_class=length_as_class,
                flat_order=flat_order,
            )

            mask_gt, rle_gt_cmp = task_utils.mask_from_tokens(
                gt_rle_,
                (n_rows, n_cols),
                starts_offset=starts_offset,
                lengths_offset=lengths_offset,
                class_offset=class_offset,
                starts_2d=False,
                multi_class=multi_class,
                max_length=max_length,
                length_as_class=length_as_class,
                flat_order=flat_order,
            )

            if self.config.debug:
                # mask_from_file = task_utils.mask_vis_to_id(mask_from_file, n_classes, copy=True)
                mask_from_file_sub = task_utils.mask_vis_to_id(mask_from_file_sub, n_classes, copy=True)
                if not np.array_equal(mask_from_file_sub, mask_gt):
                    raise AssertionError("vid_mask_gt mismatch")

            n_classes = len(self.class_id_to_col)

            seq_img_infos = json_vid_info[seq]
            if seq_img_infos:
                out_frame_id = seq_img_infos[-1]['out_frame_id']
            else:
                out_frame_id = 0

            out_frame_id += 1
            img_info = dict(
                seq=str(seq),
                vid_id=-1,
                image_id=str(image_id_),
                src_frame_id=int(frame_id_),
                out_frame_id=int(out_frame_id),
                vid_path=str(vid_path),
                mask_vid_path=str(mask_vid_path),
            )

            # if subsample > 1:
            #     mask_rec = task_utils.resize_mask(mask_rec, orig_size_, n_classes)
            #     mask_gt = task_utils.resize_mask(mask_gt, orig_size_, n_classes)

            vis_utils.visualize_mask(
                image_id_,
                image_,
                mask_rec,
                mask_logits,
                mask_gt,
                self.class_id_to_col,
                seq_id=seq,
                img_info=img_info,
                out_mask_dir=out_mask_dir,
                out_mask_logits_dir=out_mask_logits_dir,
                out_vis_dir=out_vis_dir,
                vid_writers=vid_cap,
                orig_size=orig_size_,
                show=show,
            )
            seq_img_infos.append(
                img_info
            )

    def compute_scalar_metrics(self, step):
        raise AssertionError('not implemented')

    def reset_metrics(self):
        raise AssertionError('not implemented')

#
# def build_response_seq_from_rle(
#         rle_norm,
#         starts_bins,
#         lengths_bins,
#         coord_vocab_shift
# ):
#     batch_size, seq_len = rle_norm.shape
#     n_elem = batch_size * seq_len
#     is_padding = tf.equal(rle_norm, 0)
#
#     rle_norm_flat = tf.reshape(rle_norm, [-1])
#     starts = rle_norm_flat[::2]
#     lengths = rle_norm_flat[1::2]
#     quantized_starts = utils.quantize(starts, starts_bins)
#     quantized_lengths = utils.quantize(lengths, lengths_bins)
#
#     quantized_starts = quantized_starts + coord_vocab_shift
#     quantized_lengths = quantized_lengths + vocab.BASE_VOCAB_SHIFT
#
#     even_indices = [[k, ] for k in range(0, n_elem, 2)]
#     odd_indices = [[k, ] for k in range(1, n_elem, 2)]
#
#     even_indices_tf = tf.constant(even_indices)
#     odd_indices_tf = tf.constant(odd_indices)
#
#     # even_indices_tf = tf.reshape(even_indices_tf, [1, -1])
#     # odd_indices_tf = tf.reshape(odd_indices_tf, [1, -1])
#
#     quantized_rle_flat = tf.zeros_like(rle_norm_flat, dtype=tf.int64)
#     quantized_rle_flat = tf.tensor_scatter_nd_update(quantized_rle_flat, even_indices_tf, quantized_starts)
#     quantized_rle_flat = tf.tensor_scatter_nd_update(quantized_rle_flat, odd_indices_tf, quantized_lengths)
#
#     # is_even = np.zeros((n_elem,), dtype=bool)
#     # is_even[even_indices] = True
#     # is_even_tf = tf.constant(is_even)
#     # quantized_rle_flat = tf.where(is_even_tf, quantized_starts, quantized_starts)
#
#     quantized_rle = tf.reshape(quantized_rle_flat, rle_norm.shape)
#
#     quantized_rle = tf.where(is_padding,
#                              tf.zeros_like(quantized_rle), quantized_rle)
#
#     token_weights = tf.ones_like(quantized_rle)
#
#     return quantized_rle, token_weights
