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
"""Object detection task via COCO metric evaluation."""
import json
import os
import pickle
from typing import Any, Dict, List

from absl import logging
import ml_collections
import numpy as np
import utils
import vocab
from metrics import metric_registry
from metrics import metric_utils
from tasks import task as task_lib
from tasks import task_utils
from tasks.visualization import vis_utils
import tensorflow as tf


@task_lib.TaskRegistry.register('video_detection')
class TaskVideoDetection(task_lib.Task):
    """
    video detection task with coco metric evaluation.
    """

    def __init__(self,
                 config: ml_collections.ConfigDict):
        super().__init__(config)

        self.max_seq_len = config.task.get('max_seq_len', 'auto')
        self.max_seq_len_test = config.task.get('max_seq_len_test', 'auto')

        if self.max_seq_len == 'auto':
            self.max_seq_len = config.task.max_instances_per_image * (config.dataset.length * 4 + 1)

        if self.max_seq_len_test == 'auto':
            self.max_seq_len_test = config.task.max_instances_per_image_test * (config.dataset.length * 4 + 1) + 1

        self.config.task.max_seq_len = self.max_seq_len
        self.config.task.max_seq_len_test = self.max_seq_len_test

        self._category_names = task_utils.get_category_names(
            config.dataset.get('category_names_path'))
        metric_config = config.task.get('metric')
        if metric_config and metric_config.get('name'):
            self._coco_metrics = metric_registry.MetricRegistry.lookup(
                metric_config.name)(config)
        else:
            self._coco_metrics = None
        if self.config.task.get('eval_outputs_json_path', None):
            self.eval_output_annotations = []

    def preprocess_single(self, dataset, batch_duplicates, training):
        """Task-specific preprocessing of individual example in the dataset.

        Args:
          dataset: A tf.data.Dataset.
          batch_duplicates: `int`, enlarge a batch by augmenting it multiple times
            (as specified) and concating the augmented examples.
          training: bool.

        Returns:
          A dataset.
        """

        def _convert_video_to_image_features(example):
            new_example = dict(
                orig_video_size=tf.shape(example['video/frames'])[1:3],
                video_id=example['video/id'],
                num_frames=example['video/num_frames'],
                video=example['video/frames'],
                bbox=example['bbox'],
                class_name=example['class_name'],
                class_id=example['class_id'],
                area=example['area'],
                is_crowd=example['is_crowd'],
            )
            return new_example

        dataset = dataset.map(_convert_video_to_image_features,
                              # num_parallel_calls=tf.data.experimental.AUTOTUNE
                              )
        dataset = dataset.map(
            lambda x: self.preprocess_single_example(
                x, training,
                batch_duplicates),
            # num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        return dataset

    def preprocess_batched(self, batched_examples, training):
        """Task-specific preprocessing of batched examples on accelerators (TPUs).

        Typical operations in this preprocessing step for detection task:
          - Quantization and serialization of object instances.
          - Creating the input sequence, target sequence, and token weights.

        Args:
          batched_examples: tuples of feature and label tensors that are
            preprocessed, batched, and stored with `dict`.
          training: bool.

        Returns:
          images: `float` of shape (bsz, h, w, c)
          input_seq: `int` of shape (bsz, seqlen).
          target_seq: `int` of shape (bsz, seqlen).
          token_weights: `float` of shape (bsz, seqlen).
        """
        config = self.config.task
        mconfig = self.config.model
        dconfig = self.config.dataset

        # bbox_np = batched_examples['bbox'].numpy()
        # class_id_np = batched_examples['class_id'].numpy()
        # class_name_np = batched_examples['class_name'].numpy()

        # Create input/target seq.
        """coord_vocab_shift needed to accomodate class tokens before the coord tokens"""
        ret = build_response_seq_from_video_bboxes(
            bboxes=batched_examples['bbox'],
            label=batched_examples['class_id'],
            quantization_bins=config.quantization_bins,
            noise_bbox_weight=config.noise_bbox_weight,
            coord_vocab_shift=mconfig.coord_vocab_shift,
            vid_len=dconfig.length,
            class_label_corruption=config.class_label_corruption)

        """response_seq_cm has random and noise labels by default"""
        response_seq, response_seq_cm, token_weights = ret

        """
        vocab_id=10 for object_detection
        prompt_seq is apparently just a [bsz, 1] vector containing vocab_id and serves as the start token        
        """
        prompt_seq = task_utils.build_prompt_seq_from_task_id(
            task_vocab_id=self.task_vocab_id,
            response_seq=response_seq)  # (bsz, 1)
        input_seq = tf.concat([prompt_seq, response_seq_cm], -1)
        target_seq = tf.concat([prompt_seq, response_seq], -1)

        # Pad sequence to a unified maximum length.
        """
        max_seq_len=512 for object_detection
        """
        assert input_seq.shape[-1] <= config.max_seq_len + 1, \
            f"input_seq length {input_seq.shape[-1]} exceeds max_seq_len {config.max_seq_len + 1}"
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
            # stupid way of assigning eos_token_weight to padding tokens,
            # just eos_token_weight should work here too since tf.where is supposed to broadcast
            tf.zeros_like(token_weights) + config.eos_token_weight,
            token_weights)

        if training:
            return batched_examples['video'], input_seq, target_seq, token_weights
        else:
            return batched_examples['video'], response_seq, batched_examples

    def infer(self, model, preprocessed_outputs):
        """Perform inference given the model and preprocessed outputs."""
        config = self.config.task
        image, _, examples = preprocessed_outputs  # response_seq unused by default
        bsz = tf.shape(image)[0]
        prompt_seq = task_utils.build_prompt_seq_from_task_id(
            self.task_vocab_id, prompt_shape=(bsz, 1))
        pred_seq, logits, _ = model.infer(
            image, prompt_seq, encoded=None,
            max_seq_len=config.max_seq_len_test,
            temperature=config.temperature, top_k=config.top_k, top_p=config.top_p)
        # if True:  # Sanity check by using gt response_seq as pred_seq.
        #   pred_seq = preprocessed_outputs[1]
        #   logits = tf.one_hot(pred_seq, mconfig.vocab_size)
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
        images, image_ids = example['video'], example['video/id']
        orig_video_size = example['orig_video_size']
        unpadded_video_size = example['unpadded_video_size']

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
                    utils.tf_float32(unpadded_video_size))
            scale = scale * utils.tf_float32(orig_video_size)
            scale = tf.expand_dims(scale, 1)
        pred_bboxes_rescaled = utils.scale_points(pred_bboxes, scale)

        gt_classes, gt_bboxes = example['label'], example['bbox']
        gt_bboxes_rescaled = utils.scale_points(gt_bboxes, scale)
        area, is_crowd = example['area'], example['is_crowd']

        return (images, image_ids, pred_bboxes, pred_bboxes_rescaled, pred_classes,
                scores, gt_classes, gt_bboxes, gt_bboxes_rescaled, area,
                # is_crowd
                )

    def postprocess_cpu(self,
                        outputs,
                        train_step,
                        # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
                        out_vis_dir=None,
                        csv_data=None,
                        eval_step=None,
                        training=False,
                        summary_tag='eval',
                        ret_results=False):
        """CPU post-processing of outputs.

        Such as computing the metrics, log image summary.

        Note: current implementation only support eval mode where gt are given in
          metrics as they are not given here in outputs.

        Args:
          outputs: a tuple of tensor passed from `postprocess_tpu`.
          train_step: `int` scalar indicating training step of current model or
            the checkpoint.
          eval_step: `int` scalar indicating eval step for the given checkpoint.
          training: `bool` indicating training or inference mode.
          summary_tag: `string` of name scope for result summary.
          ret_results: whether to return visualization images.

        Returns:
          A dict of visualization images if ret_results, else None.
        """
        # Copy outputs to cpu.
        new_outputs = []
        for i in range(len(outputs)):
            # logging.info('Copying output at index %d to cpu for cpu post-process', i)
            new_outputs.append(tf.identity(outputs[i]))
        (images, image_ids, pred_bboxes, pred_bboxes_rescaled, pred_classes,
         # pylint: disable=unbalanced-tuple-unpacking
         scores, gt_classes, gt_bboxes, gt_bboxes_rescaled, area,
         # is_crowd
         ) = new_outputs

        if self.config.task.get('eval_outputs_json_path', None):
            annotations = build_annotations(image_ids.numpy(),
                                            pred_classes.numpy(),
                                            pred_bboxes_rescaled.numpy(),
                                            scores.numpy(),
                                            len(self.eval_output_annotations))
            self.eval_output_annotations.extend(annotations)

        # Log/accumulate metrics.
        # if self._coco_metrics:
        #     self._coco_metrics.record_prediction(
        #         image_ids, pred_bboxes_rescaled, pred_classes, scores)
        #     if not self._coco_metrics.gt_annotations_path:
        #         self._coco_metrics.record_groundtruth(
        #             image_ids,
        #             gt_bboxes_rescaled,
        #             gt_classes,
        #             areas=area,
        #             is_crowds=is_crowd)

        # Image summary.
        if eval_step <= 10 or ret_results:
            image_ids_ = image_ids.numpy().flatten().astype(str)
            image_ids__ = list(image_ids_)
            gt_tuple = (gt_bboxes, gt_classes, scores * 0. + 1., 'gt')  # pylint: disable=unused-variable
            pred_tuple = (pred_bboxes, pred_bboxes_rescaled, pred_classes, scores, 'pred')
            vis_list = [pred_tuple]  # exclude gt for simplicity.
            ret_images = []
            for bboxes_, bboxes_rescaled_, classes_, scores_, tag_ in vis_list:
                tag = summary_tag + '/' + task_utils.join_if_not_none(
                    [tag_, 'bbox', eval_step], '_')
                bboxes_, bboxes_rescaled_, classes_, scores_ = (
                    bboxes_.numpy(), bboxes_rescaled_.numpy(), classes_.numpy(), scores_.numpy())
                images_ = np.copy(tf.image.convert_image_dtype(images, tf.uint8))
                ret_images += add_image_summary_with_bbox(
                    images_, bboxes_, bboxes_rescaled_, classes_, scores_, self._category_names,
                    image_ids__, train_step, tag,
                    out_vis_dir=out_vis_dir,
                    csv_data=csv_data,
                    max_images_shown=(-1 if ret_results else 3))

        logging.info('Done post-process')
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


def build_response_seq_from_video_bboxes(
        bboxes,
        label,
        quantization_bins,
        noise_bbox_weight,
        coord_vocab_shift,
        vid_len,
        class_label_corruption='rand_cls'):
    """"Build target seq from bounding bboxes for video detection.

    Objects are serialized using the format of yxyx...c.

    Args:
      bboxes: `float` bounding box of shape (bsz, n, 4).
      label: `int` label of shape (bsz, n).
      quantization_bins: `int`.
      noise_bbox_weight: `float` on the token weights for noise bboxes.
      coord_vocab_shift: `int`, shifting coordinates by a specified integer.
      class_label_corruption: `string` specifying how labels are corrupted for the
        input_seq.

    Returns:
      discrete sequences with shape (bsz, seqlen).
    """
    # bboxes = tf.where(
    #     bboxes == -1,
    #     vocab.NO_BOX_TOKEN,
    #     bboxes)
    # from utils import add_name, np_dict

    assert bboxes.shape[-1] % 4 == 0, f"invalid bboxes shape: {bboxes.shape}"

    n_bboxes_per_vid = int(bboxes.shape[-1] / 4)
    assert vid_len == n_bboxes_per_vid, f"Mismatch between vid_len: {vid_len} and n_bboxes_per_vid: {n_bboxes_per_vid}"

    is_no_box = tf.math.is_nan(bboxes)

    quantized_bboxes = utils.quantize(bboxes, quantization_bins)
    quantized_bboxes = quantized_bboxes + coord_vocab_shift

    # np_dict = utils.to_numpy(locals())

    # quantized_bboxes_np = quantized_bboxes.numpy()
    # add_name(vars())

    quantized_bboxes = tf.where(
        is_no_box,
        vocab.NO_BOX_TOKEN,
        quantized_bboxes)

    """set 0-labeled (padding) bboxes to zero"""
    is_padding = tf.expand_dims(tf.equal(label, 0), -1)
    quantized_bboxes = tf.where(
        is_padding,
        tf.zeros_like(quantized_bboxes),
        quantized_bboxes)
    new_label = tf.expand_dims(label + vocab.BASE_VOCAB_SHIFT, -1)
    new_label = tf.where(is_padding, tf.zeros_like(new_label), new_label)
    lb_shape = tf.shape(new_label)

    # Bbox and label serialization.
    response_seq = tf.concat([quantized_bboxes, new_label], axis=-1)

    """Merge last few dims to have rank-2 shape [bsz, n_tokens] where
    n_tokens = n_bboxes*5    
    """
    response_seq = utils.flatten_non_batch_dims(response_seq, 2)

    """
    different combinations of random, fake and real class labels apparently 
    created in case something other than the real labels is required 
    according to the class_label_corruption Parameter
    
    class_label_corruption=rand_n_fake_cls by default
    
    
    """
    rand_cls = vocab.BASE_VOCAB_SHIFT + tf.random.uniform(
        lb_shape,
        0,
        coord_vocab_shift - vocab.BASE_VOCAB_SHIFT,
        dtype=new_label.dtype)
    fake_cls = vocab.FAKE_CLASS_TOKEN + tf.zeros_like(new_label)
    rand_n_fake_cls = tf.where(
        tf.random.uniform(lb_shape) > 0.5, rand_cls, fake_cls)
    real_n_fake_cls = tf.where(
        tf.random.uniform(lb_shape) > 0.5, new_label, fake_cls)
    real_n_rand_n_fake_cls = tf.where(
        tf.random.uniform(lb_shape) > 0.5, new_label, rand_n_fake_cls)
    label_mapping = {'none': new_label,
                     'rand_cls': rand_cls,
                     'real_n_fake_cls': real_n_fake_cls,
                     'rand_n_fake_cls': rand_n_fake_cls,
                     'real_n_rand_n_fake_cls': real_n_rand_n_fake_cls}
    new_label_m = label_mapping[class_label_corruption]
    new_label_m = tf.where(is_padding, tf.zeros_like(new_label_m), new_label_m)

    """response_seq_class_m is same as response_seq if no corruptions are needed,
    i.e. if class_label_corruption=none
    otherwise, some or all the real labels are randomly replaced by noise label to 
    generate corrupted labels that are used as input sequence
    
    The rationale for corrupting the input sequences might be that we want the network to produce 
    the right class labels in the subsequent outputs even if the previous one was incorrect 
    So we do not want to condition the Generation of class output In the next tokens Lee to is strongly on 
    the class label being correct in the previously generated tokens
    """
    response_seq_class_m = tf.concat([quantized_bboxes, new_label_m], axis=-1)
    response_seq_class_m = utils.flatten_non_batch_dims(response_seq_class_m, 2)

    # Get token weights.
    is_real = tf.cast(tf.not_equal(new_label, vocab.FAKE_CLASS_TOKEN), tf.float32)

    """    
    noise and real bbox coord tokens have weights 1 and 0 respectively
    
    real bbox class tokens have weight 1
    noise bbox class tokens have weight noise_bbox_weight 
    
    noise_bbox_weight = 1.0 when training with fake objects
    
    We don't care about the coordinates of fake boxes but we do care about their class   
    """
    bbox_weight = tf.tile(is_real, [1, 1, int(bboxes.shape[-1])])
    label_weight = is_real + (1. - is_real) * noise_bbox_weight

    token_weights = tf.concat([bbox_weight, label_weight], -1)
    token_weights = utils.flatten_non_batch_dims(token_weights, 2)

    return response_seq, response_seq_class_m, token_weights


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