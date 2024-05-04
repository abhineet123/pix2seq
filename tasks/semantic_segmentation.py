from typing import Any, Dict, List

import ml_collections
import numpy as np
import utils
import vocab
from metrics import metric_utils
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
        mconfig = self.config.model

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
        examples, input_seq, target_seq, token_weights = preprocessed_outputs  # response_seq unused by
        # default
        image = examples["image"]
        bsz = tf.shape(image)[0]
        prompt_seq = task_utils.build_prompt_seq_from_task_id(
            self.task_vocab_id, prompt_shape=(bsz, 1))
        pred_seq, logits, _ = model.infer(
            image, prompt_seq, encoded=None,
            max_seq_len=(config.max_instances_per_image_test * 5 + 1),
            temperature=config.temperature, top_k=config.top_k, top_p=config.top_p)

        return examples, pred_seq, logits

    def postprocess_tpu(self, batched_examples, pred_seq, logits,
                        training=False):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
        """Organizing results after fitting the batched examples in graph.

        Such as updating metrics, putting together results for computing metrics in
          CPU/numpy mode.

        Note: current implementation only support eval mode where gt are given in
          metrics as they are not constructed here from input_seq/target_seq.

        Args:
          batched_examples: a tuple of features (`dict`) and labels (`dict`),
            containing images and labels.
          pred_seq: `int` sequence of shape (bsz, seqlen').
          logits: `float` sequence of shape (bsz, seqlen', vocab_size).
          training: `bool` indicating training or inference mode.

        Returns:
          results for passing to `postprocess_cpu` which runs in CPU mode.
        """
        config = self.config.task
        mconfig = self.config.model
        example = batched_examples
        images, image_ids = example['image'], example['image/id']
        orig_image_size = example['orig_image_size']
        unpadded_image_size = example['unpadded_image_size']

        # Decode sequence output.
        pred_classes, pred_bboxes, scores = task_utils.decode_object_seq_to_bbox(
            logits, pred_seq, config.quantization_bins, mconfig.coord_vocab_shift)

        # Compute coordinate scaling from [0., 1.] to actual pixels in orig image.
        image_size = images.shape[1:3].as_list()
        if training:
            # scale points to whole image size during train.
            scale = utils.tf_float32(image_size)
        else:
            # scale points to original image size during eval.
            scale = (
                    utils.tf_float32(image_size)[tf.newaxis, :] /
                    utils.tf_float32(unpadded_image_size))
            scale = scale * utils.tf_float32(orig_image_size)
            scale = tf.expand_dims(scale, 1)
        pred_bboxes_rescaled = utils.scale_points(pred_bboxes, scale)

        gt_classes, gt_bboxes = example['label'], example['bbox']
        gt_bboxes_rescaled = utils.scale_points(gt_bboxes, scale)
        area, is_crowd = example['area'], example['is_crowd']

        return (images, image_ids, pred_bboxes, pred_bboxes_rescaled, pred_classes,
                scores, gt_classes, gt_bboxes, gt_bboxes_rescaled, area,
                orig_image_size, unpadded_image_size,
                # is_crowd
                )

    def postprocess_cpu(self,
                        outputs,
                        train_step,
                        # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
                        out_vis_dir=None,
                        vid_cap=None,
                        csv_data=None,
                        eval_step=None,
                        training=False,
                        summary_tag='eval',
                        ret_results=False,
                        min_score_thresh=0.1,
                        ):
        # Copy outputs to cpu.
        new_outputs = []
        for i in range(len(outputs)):
            # logging.info('Copying output at index %d to cpu for cpu post-process', i)
            new_outputs.append(tf.identity(outputs[i]))
        (images, image_ids, pred_bboxes, pred_bboxes_rescaled, pred_classes,
         scores, gt_classes, gt_bboxes, gt_bboxes_rescaled, area,
         orig_image_size, unpadded_image_size,
         # is_crowd
         ) = new_outputs

        unpadded_image_size = unpadded_image_size.numpy()
        orig_image_size = orig_image_size.numpy()

        # Image summary.
        image_ids_ = image_ids.numpy().flatten().astype(str)
        image_ids__ = list(image_ids_)
        ret_images = []
        bboxes_, bboxes_rescaled_, classes_, scores_ = (
            pred_bboxes.numpy(), pred_bboxes_rescaled.numpy(), pred_classes.numpy(), scores.numpy())
        images_ = np.copy(tf.image.convert_image_dtype(images, tf.uint8))
        ret_images += vis_utils.add_image_summary_with_bbox(
            images_, bboxes_, bboxes_rescaled_, classes_, scores_,
            self._category_names,
            image_ids__,
            # train_step, tag,
            # max_images_shown=(-1 if ret_results else 3)
            out_vis_dir=out_vis_dir,
            vid_cap=vid_cap,
            csv_data=csv_data,
            min_score_thresh=min_score_thresh,
            unpadded_size=unpadded_image_size,
            orig_size=orig_image_size,
        )

        if ret_results:
            return ret_images

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


def build_annotations(image_ids, category_ids, boxes, scores,
                      counter) -> List[Dict[str, Any]]:
    """Builds annotations."""
    annotations = []
    for image_id, category_id_list, box_list, score_list in zip(
            image_ids, category_ids, boxes, scores):
        for category_id, box, score in zip(category_id_list, box_list, score_list):
            category_id = int(category_id)
            if category_id:
                annotations.append({
                    'id': counter,
                    'image_id': int(image_id),
                    'category_id': category_id,
                    'bbox': metric_utils.yxyx_to_xywh(box.tolist()),
                    'iscrowd': False,
                    'score': float(score)
                })
                counter += 1
    return annotations
