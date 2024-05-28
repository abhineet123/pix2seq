import ml_collections
import numpy as np
import utils
import vocab
from tasks import task as task_lib
from tasks import task_utils
from tasks.visualization import vis_utils
import tensorflow as tf


@task_lib.TaskRegistry.register('video_segmentation')
class TaskVideoSegmentation(task_lib.Task):
    def __init__(self,
                 config: ml_collections.ConfigDict):
        super().__init__(config)

        self._category_names = task_utils.get_category_names(
            config.dataset.get('category_names_path'))

        class_id_to_col, class_id_to_name = task_utils.get_class_info(self._category_names)


        # class_info_path = config.dataset.get('class_info_path')
        # assert class_info_path, "class_info_path must be provided"
        # class_id_to_col, class_id_to_name = task_utils.read_class_info(class_info_path)

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

    def check_video_rle(self, batched_examples, show):
        mask_vid_paths = batched_examples['mask_vid_path'].numpy()
        videos = batched_examples['video'].numpy()
        img_ids_all = batched_examples['image_ids'].numpy()
        frame_ids_all = batched_examples['frame_ids'].numpy()
        rles = batched_examples['rle'].numpy()
        rle_lens = batched_examples['rle_len'].numpy()

        batch_size = frame_ids_all.shape[0]
        max_length = self.config.dataset.train.max_length
        subsample = self.config.dataset.train.subsample
        multi_class = self.config.dataset.multi_class
        time_as_class = self.config.dataset.time_as_class
        length_as_class = self.config.dataset.length_as_class
        flat_order = self.config.dataset.flat_order

        starts_offset = self.config.model.coord_vocab_shift
        lengths_offset = self.config.model.len_vocab_shift
        class_offset = self.config.model.class_vocab_shift

        # starts_bins = self.config.task.starts_bins
        # lengths_bins = self.config.task.lengths_bins

        class_id_to_col = self.class_id_to_col
        class_id_to_name = self.class_id_to_name

        max_seq_len = self.config.model.max_seq_len

        n_classes = len(self.class_id_to_col)

        vocab_size = self.config.model.vocab_size

        for batch_id in range(batch_size):
            rle_len = rle_lens[batch_id]
            rle = rles[batch_id]
            rle_tokens = rle[rle != vocab.PADDING_TOKEN]
            if rle_len > max_seq_len:
                assert rle_tokens.size == max_seq_len, "curtailed RLE length mismatch"
                print(f'skipping curtailed rle with original length {rle_len}')
                continue

            mask_vid_path = mask_vid_paths[batch_id].decode('utf-8')
            img_ids = img_ids_all[batch_id]
            video = videos[batch_id]
            frame_ids = frame_ids_all[batch_id]
            vid_reader, vid_width, vid_height, num_frames = task_utils.load_video(mask_vid_path)
            vid_mask = []
            for frame_id in frame_ids:
                mask = task_utils.read_frame(vid_reader, frame_id - 1, mask_vid_path)
                if not multi_class:
                    mask[mask > 0] = 255

                vid_mask.append(mask)
            vid_mask = np.stack(vid_mask, axis=0)


            assert rle_tokens.size == rle_len, "rle_len mismatch"

            if rle_len:
                n_run_tokens = 2
                if (multi_class or time_as_class) and not length_as_class:
                    n_run_tokens += 1
                assert len(rle_tokens) % n_run_tokens == 0, f"rle_tokens length must be divisible by {n_run_tokens}"
                starts_tokens = np.array(rle_tokens[0:][::n_run_tokens], dtype=np.int64)
                max_starts = np.amax(starts_tokens)
                min_starts = np.amin(starts_tokens)
                assert max_starts < vocab_size, "max_starts exceeds vocab_size"
                assert min_starts > starts_offset, "starts_offset exceeds min_starts"
                # assert max_starts < starts_bins + starts_offset, "max_starts exceeds starts_bins + starts_offset"

                lengths_tokens = np.array(rle_tokens[1:][::n_run_tokens], dtype=np.int64)
                max_lengths_tokens = np.amax(lengths_tokens)
                min_lengths_tokens = np.amin(lengths_tokens)
                assert max_lengths_tokens < starts_offset, "max_lengths_tokens exceeds starts_offset"
                assert min_lengths_tokens > lengths_offset, "lengths_offset exceeds min_lengths_tokens"
                # assert max_lengths_tokens < lengths_bins + lengths_offset, "max_lengths_tokens exceeds lengths_bins + lengths_offset"

                if (multi_class or time_as_class) and not length_as_class:
                    class_tokens = np.array(rle_tokens[2:][::n_run_tokens], dtype=np.int64)
                    max_class_tokens = np.amax(class_tokens)
                    min_class_tokens = np.amin(class_tokens)
                    assert max_class_tokens < starts_offset, "max_class_tokens exceeds starts_offset"
                    assert min_class_tokens > class_offset, "class_offset exceeds min_class_tokens"

                    if not time_as_class:
                        assert max_class_tokens < lengths_offset, "max_class_tokens exceeds lengths_offset"

            task_utils.check_video_rle_tokens(
                video, vid_mask, rle_tokens,
                n_classes=n_classes,
                length_as_class=length_as_class,
                starts_offset=starts_offset,
                time_as_class=time_as_class,
                lengths_offset=lengths_offset,
                class_offset=class_offset,
                max_length=max_length,
                subsample=subsample,
                class_id_to_name=class_id_to_name,
                class_id_to_col=class_id_to_col,
                multi_class=multi_class,
                flat_order=flat_order,
                is_vis=1,
                tac_mask_sub=None,
                tac_id_to_col=None,
            )

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
            self.check_video_rle(batched_examples, show=1)

        prompt_seq = task_utils.build_prompt_seq_from_task_id(
            task_vocab_id=self.task_vocab_id,
            response_seq=response_seq)
        input_seq = tf.concat([prompt_seq, response_seq], -1)
        target_seq = tf.concat([prompt_seq, response_seq], -1)

        """
        right shift the target_seq and left-shift the input_seq
        """
        input_seq, target_seq = input_seq[..., :-1], target_seq[..., 1:]

        """
        Assign lower weights for ending/padding tokens.
        eos_token_weight = 0.1
        """
        token_weights = tf.where(
            target_seq == vocab.PADDING_TOKEN,
            tf.zeros_like(token_weights) + config.eos_token_weight,
            token_weights)

        """goes into video_ar_model.compute_loss"""
        return batched_examples, input_seq, target_seq, token_weights

    def infer(self, model, preprocessed_outputs):
        config = self.config.task
        mconfig = self.config.model
        examples, input_seq, target_seq, token_weights = preprocessed_outputs
        video = examples["video"]
        bsz = tf.shape(video)[0]
        prompt_seq = task_utils.build_prompt_seq_from_task_id(
            self.task_vocab_id, prompt_shape=(bsz, 1))
        pred_seq, logits, _ = model.infer(
            video, prompt_seq, encoded=None,
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
            new_outputs.append(tf.identity(outputs[i]))

        images, image_ids, rles, logits, gt_rles, orig_sizes = new_outputs

        orig_sizes = orig_sizes.numpy()
        gt_rles = gt_rles.numpy()
        rles = rles.numpy()
        logits = logits.numpy()

        image_ids_ = image_ids.numpy().flatten().astype(str)
        image_ids = list(image_ids_)
        images = np.copy(tf.image.convert_image_dtype(images, tf.uint8))
        multi_class = self.config.dataset.multi_class
        for image_id_, image_, rle_, logits_, orig_size_, gt_rle_ in zip(
                image_ids, images, rles, logits, orig_sizes, gt_rles):
            orig_size_ = tuple(orig_size_)
            n_rows, n_cols = orig_size_

            max_length = self.config.dataset.train.max_length
            subsample = self.config.dataset.train.subsample

            if subsample > 1:
                max_length = int(max_length / subsample)
                n_rows, n_cols = int(n_rows / subsample), int(n_cols / subsample)

            rle_ = rle_[rle_ != vocab.PADDING_TOKEN]

            mask_rec, rle_rec_cmp = task_utils.mask_from_tokens(
                rle_,
                (n_rows, n_cols),
                starts_offset=self.config.model.coord_vocab_shift,
                lengths_offset=self.config.model.len_vocab_shift,
                class_offset=self.config.model.class_vocab_shift,
                starts_2d=False,
                multi_class=multi_class,
            )

            mask_gt, rle_gt_cmp = task_utils.mask_from_tokens(
                gt_rle_,
                (n_rows, n_cols),
                starts_offset=self.config.model.coord_vocab_shift,
                lengths_offset=self.config.model.len_vocab_shift,
                class_offset=self.config.model.class_vocab_shift,
                starts_2d=False,
                multi_class=multi_class,
            )
            n_classes = len(self.class_id_to_col)

            if subsample > 1:
                mask_rec = task_utils.resize_mask(mask_rec, orig_size_, n_classes)
                mask_gt = task_utils.resize_mask(mask_gt, orig_size_, n_classes)

            vis_utils.visualize_mask(
                image_id_,
                image_,
                mask_rec,
                mask_gt,
                self._category_names,
                out_mask_dir=out_mask_dir,
                out_vis_dir=out_vis_dir,
                vid_writers=vid_cap,
                orig_size=orig_size_,
                show=show,
            )

    def compute_scalar_metrics(self, step):
        raise AssertionError('not implemented')

    def reset_metrics(self):
        raise AssertionError('not implemented')
