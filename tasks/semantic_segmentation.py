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

        self._category_names = task_utils.get_category_names(
            config.dataset.get('category_names_path'))

    def preprocess_single(self, dataset, batch_duplicates, training, validation):
        if self.config.debug != 2:
            """apply transforms"""
            dataset = dataset.map(
                lambda x: self.preprocess_single_example(
                    x, training, validation, batch_duplicates),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return dataset

    def check_rle(self, batched_examples):
        mask_vid_paths = batched_examples['mask_vid_path'].numpy()
        frame_ids = batched_examples['frame_id'].numpy()
        rles = batched_examples['rle'].numpy()
        batch_size = frame_ids.shape[0]
        for batch_id in range(batch_size):
            mask_vid_path = mask_vid_paths[batch_id].decode('utf-8')
            rle = rles[batch_id]
            frame_id = frame_ids[batch_id]
            vid_reader, vid_width, vid_height, num_frames = task_utils.load_video(mask_vid_path)
            mask = task_utils.read_frame(vid_reader, frame_id - 1, mask_vid_path)
            rle_stripped = rle[rle != vocab.PADDING_TOKEN]
            task_utils.check_rle(mask, rle_stripped, self.config.model.coord_vocab_shift, vocab.BASE_VOCAB_SHIFT)

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

        # self.check_rle(batched_examples)

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
                        vid_cap=None,
                        csv_data=None,
                        eval_step=None,
                        training=False,
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

        for image_id_, image_, rle_, logits_, orig_size_, gt_rle_ in zip(
                image_ids, images, rles, logits, orig_sizes, gt_rles):
            orig_size_ = tuple(orig_size_)
            rle_ = rle_[rle_ != vocab.PADDING_TOKEN]
            mask = task_utils.rle_to_mask(
                rle_,
                shape=orig_size_,
                starts_offset=self.config.model.coord_vocab_shift,
                lengths_offset=vocab.BASE_VOCAB_SHIFT,
                starts_2d=False)

            gt_mask = task_utils.rle_to_mask(
                gt_rle_,
                shape=orig_size_,
                starts_offset=self.config.model.coord_vocab_shift,
                lengths_offset=vocab.BASE_VOCAB_SHIFT,
                starts_2d=False)

            vis_utils.visualize_mask(
                image_id_,
                image_,
                mask,
                gt_mask,
                self._category_names,
                out_mask_dir=out_mask_dir,
                out_vis_dir=out_vis_dir,
                vid_writers=vid_cap,
                csv_data=csv_data,
                orig_size=orig_size_,
            )

    def compute_scalar_metrics(self, step):
        raise AssertionError('not implemented')

    def reset_metrics(self):
        raise AssertionError('not implemented')


def build_response_seq_from_rle(
        rle_norm,
        starts_bins,
        lengths_bins,
        coord_vocab_shift
):
    batch_size, seq_len = rle_norm.shape
    n_elem = batch_size * seq_len
    is_padding = tf.equal(rle_norm, 0)

    rle_norm_flat = tf.reshape(rle_norm, [-1])
    starts = rle_norm_flat[::2]
    lengths = rle_norm_flat[1::2]
    quantized_starts = utils.quantize(starts, starts_bins)
    quantized_lengths = utils.quantize(lengths, lengths_bins)

    quantized_starts = quantized_starts + coord_vocab_shift
    quantized_lengths = quantized_lengths + vocab.BASE_VOCAB_SHIFT

    even_indices = [[k, ] for k in range(0, n_elem, 2)]
    odd_indices = [[k, ] for k in range(1, n_elem, 2)]

    even_indices_tf = tf.constant(even_indices)
    odd_indices_tf = tf.constant(odd_indices)

    # even_indices_tf = tf.reshape(even_indices_tf, [1, -1])
    # odd_indices_tf = tf.reshape(odd_indices_tf, [1, -1])

    quantized_rle_flat = tf.zeros_like(rle_norm_flat, dtype=tf.int64)
    quantized_rle_flat = tf.tensor_scatter_nd_update(quantized_rle_flat, even_indices_tf, quantized_starts)
    quantized_rle_flat = tf.tensor_scatter_nd_update(quantized_rle_flat, odd_indices_tf, quantized_lengths)

    # is_even = np.zeros((n_elem,), dtype=bool)
    # is_even[even_indices] = True
    # is_even_tf = tf.constant(is_even)
    # quantized_rle_flat = tf.where(is_even_tf, quantized_starts, quantized_starts)

    quantized_rle = tf.reshape(quantized_rle_flat, rle_norm.shape)

    quantized_rle = tf.where(is_padding,
                             tf.zeros_like(quantized_rle), quantized_rle)

    token_weights = tf.ones_like(quantized_rle)

    return quantized_rle, token_weights