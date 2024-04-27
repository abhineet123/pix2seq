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

    def preprocess_batched(self, batched_examples, training):
        config = self.config.task
        mconfig = self.config.model

        if self.config.debug == 2:
            batched_examples = vis_utils.debug_image_pipeline(
                self.dataset,
                self.train_transforms, batched_examples,
                vis=1, model_dir=self.config.model_dir, training=training)

        response_seq, token_weights = build_response_seq_from_mask(
            batched_examples['mask'],
            config.quantization_bins,
            mconfig.coord_vocab_shift,
        )
        prompt_seq = task_utils.build_prompt_seq_from_task_id(
            task_vocab_id=self.task_vocab_id,
            response_seq=response_seq)  # (bsz, 1)
        input_seq = tf.concat([prompt_seq, response_seq], -1)
        target_seq = tf.concat([prompt_seq, response_seq], -1)

        # Pad sequence to a unified maximum length.
        """
        max_seq_len=512 for object_detection
        """
        input_seq = utils.pad_to_max_len(input_seq, config.max_seq_len + 1,
                                         dim=-1, padding_token=vocab.PADDING_TOKEN)
        target_seq = utils.pad_to_max_len(target_seq, config.max_seq_len + 1,
                                          dim=-1, padding_token=vocab.PADDING_TOKEN)
        """
        right shift the target_seq and left-shift the input_seq
        """
        input_seq, target_seq = input_seq[..., :-1], target_seq[..., 1:]
        token_weights = utils.pad_to_max_len(token_weights, config.max_seq_len,
                                             dim=-1, padding_token=vocab.PADDING_TOKEN)

        # Assign lower weights for ending/padding tokens.
        """
        eos_token_weight = 0.1 for object_detection
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
        """Returns a dict containing scalar metrics to log."""
        if self._coco_metrics:
            return self._coco_metrics.result(step)
        else:
            return {}

    def reset_metrics(self):
        """Reset states of metrics accumulators."""
        if self._coco_metrics:
            self._coco_metrics.reset_states()


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def mask_to_rle(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # runs[1::2] -= runs[::2]
    # starts, lengths = runs[::2], runs[1::2]
    row, col = np.unravel_index(runs, img.shape)

    rle = [item for sublist in zip(row, col) for item in sublist]
    rle_str = ' '.join(f'{r} {c}' for r, c in zip(row, col))
    rle_str2 = ' '.join(rle)
    assert rle_str == rle_str2, "rle_str mismatch"

    n_rows, n_cols = img.shape
    row_norm, col_norm = row.astype(np.float32) / n_rows, col.astype(np.float32) / n_cols
    rle_norm = [item for sublist in zip(row_norm, col_norm) for item in sublist]
    return rle_norm


def rle2mask(mask_rle, label, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = label
    return img.reshape(shape)  # Needed to align to RLE direction


def build_response_seq_from_mask(
        mask,
        quantization_bins,
        coord_vocab_shift):
    rle_norm = mask_to_rle(mask)
    quantized_rle = utils.quantize(rle_norm, quantization_bins)
    quantized_rle = quantized_rle + coord_vocab_shift

    # quantized_rle = utils.flatten_non_batch_dims(quantized_rle, 2)
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
