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
"""The image encoder and autoregressive decoder model."""
import os.path

import ml_collections

import utils
import vocab
from tasks import task_utils
from tasks.visualization import vis_utils

from architectures.transformers import add_vis_pos_emb
from architectures.transformers import AutoregressiveDecoder
from architectures.transformers import FIT
from architectures.transformers import MLP
from architectures.transformers import VisionTransformer

from architectures.video_transformers import VideoResNetTransformer

from models import model as model_lib
from models import model_utils

import numpy as np
import tensorflow as tf


@model_lib.ModelRegistry.register('video_encoder_ar_decoder')
class Model(tf.keras.models.Model):
    """Inputs images and returns activations."""

    def __init__(self, config: ml_collections.ConfigDict, **kwargs):
        # vocab_size and max_seq_len don't include start token, which is only used
        # inside this class.
        super().__init__(**kwargs)
        self.config_all = config
        self.config = config.model
        self.vid_len = config.dataset.length
        self.pos_encoding = self.config.pos_encoding

        mlp_ratio = self.config.dim_mlp // self.config.dim_att
        if self.config.resnet_variant == 'c1':
            self.encoder = VisionTransformer(
                self.config.image_size[0], self.config.image_size[1], self.config.patch_size,
                self.config.num_encoder_layers, self.config.dim_att, mlp_ratio,
                self.config.num_heads, self.config.drop_path, self.config.drop_units,
                self.config.drop_att, self.config.pos_encoding, self.config.use_cls_token,
                name='vit')
        else:
            self.encoder = VideoResNetTransformer(
                image_height=self.config.image_size[0],
                image_width=self.config.image_size[1],
                vid_len=self.vid_len,
                resnet_variant=self.config.resnet_variant,
                resnet_depth=self.config.resnet_depth,
                resnet_width_multiplier=self.config.resnet_width_multiplier,
                resnet_sk_ratio=self.config.resnet_sk_ratio,
                num_layers=self.config.num_encoder_layers,
                dim=self.config.dim_att,
                mlp_ratio=mlp_ratio,
                num_heads=self.config.num_heads,
                drop_path=self.config.drop_path,
                drop_units=self.config.drop_units,
                drop_att=self.config.drop_att,
                pos_encoding=self.config.pos_encoding,
                use_cls_token=self.config.use_cls_token,
                name='rest')

        mlp_ratio_dec = self.config.dim_mlp_dec // self.config.dim_att_dec
        self.proj = tf.keras.layers.Dense(
            self.config.dim_att_dec, name='proj/linear')
        self.proj_ln = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name='proj/ln')
        if self.config.dec_proj_mode in ['linear_p', 'mlp']:
            """
            add visual positional embedding
            """
            self.vis_pos_emb = add_vis_pos_emb(
                self,
                pos_encoding=self.config.pos_encoding,
                n_rows=self.encoder.n_rows,
                n_cols=self.encoder.n_cols,
                dim=self.config.dim_att_dec,
                name_prefix='proj',
                return_only=True,
                # n_images=self.vid_len,
            )
            if self.config.dec_proj_mode == 'mlp':
                self.proj_mlp = MLP(1, self.config.dim_att_dec, mlp_ratio, self.config.drop_path,
                                    self.config.drop_units, name='proj/mlp')

        self.decoder = AutoregressiveDecoder(
            self.config.vocab_size, self.config.max_seq_len, self.config.num_decoder_layers,
            self.config.dim_att_dec, mlp_ratio_dec, self.config.num_heads_dec,
            self.config.drop_path, self.config.drop_units, self.config.drop_att,
            self.config.pos_encoding_dec, self.config.shared_decoder_embedding,
            self.config.decoder_output_bias, name='ar_decoder')

    def _encode_videos(self, videos, training):
        config = self.config
        encoded = self.encoder(videos, training)
        # encoded = utils.flatten_vid(encoded)

        encoded = self.proj_ln(self.proj(encoded))
        # Add (optional) positional embedding to encoded visual units.
        if config.dec_proj_mode != 'linear':
            vis_pos_emb = tf.expand_dims(self.vis_pos_emb, 0)
            # vis_pos_emb = tf.expand_dims(vis_pos_emb, 0)
            # if self.pos_encoding == 'learned_3d':
            #     encoded = utils.unflatten_vid(encoded, self.vid_len)
            #     if config.use_cls_token:
            #         raise AssertionError('use_cls_token is not supported with 3-D positional encoding')
            #     else:
            #         encoded = encoded + vis_pos_emb
            #     encoded = utils.flatten_vid(encoded)
            # else:
            if config.use_cls_token:
                encoded = encoded + tf.concat(
                    [tf.zeros_like(vis_pos_emb[:, :1]), vis_pos_emb], 1)
            else:
                encoded = encoded + vis_pos_emb
            if config.dec_proj_mode == 'mlp':
                encoded = self.proj_mlp(encoded, training)
            else:
                assert config.dec_proj_mode == 'linear_p'
        # encoded = utils.unflatten_vid(encoded, self.vid_len)
        return encoded

    def call(self, images, seq,
             training=True):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
        """Model function call for *training*.

        Args:
          images: `float` tensor of (bsz, h, w, c).
          seq: `int` sequence visible to the model of shape (bsz, seqlen),
            or (bsz, instances, seqlen) if there are multiple sequences per image.
          training: `bool` indicator.

        Returns:
          logits for each predicted tokens of (bsz * instances, seqlen, vocab_size).
        """
        with tf.name_scope(''):  # for other functions to have the same name scope.
            encoded = self._encode_videos(images, training)

            """_tile_vis_output is only needed if seq is 3D or above"""
            # encoded, seq = self._tile_vis_output(encoded, seq)

            logits = self.decoder(seq, encoded, training)
            return logits

    def infer(self, videos, prompt_seq, encoded=None, max_seq_len=None,
              temperature=1, top_k=1, top_p=1., num_samples=1,
              sampling_callback=None):
        """Model function call for inference.

        Args:
          videos: `float` tensor of (bsz, v, h, w, c).
          prompt_seq: `int` sequence visible to the model of shape (bsz, seqlen),
            or (bsz, instances, seqlen) if there are multiple sequences per image.
          encoded: cache for encoded images for decoder. Skip image encoding if this
            is given.
          max_seq_len: `int` of max generated sequence length (including prompt).
          temperature: `float` scalar for scaling the logits before sampling.
          top_k: `int` scalar for truncating top-k tokens according to logits before
            token sampling.
          top_p: `float` scalar specifying the threshold of cumulative probablity
            for truncating tokens before token sampling.
          num_samples: `int` number of samples to be generated for each instance.
          sampling_callback: a callbak `function` that take `next_logits`, and
            return `next_token`. This is used when users need a specific logic
            for sampling. Default to `None` with standard free-form sampling.

        Returns:
          pred_seq: `int` prediction sequence of shape
              (bsz * instances * num_samples, seqlen)
          logits: `float` of shape
              (bsz * instances * num_samples, seqlen, vocab_size)
          encoded: `float` tensor of encoded images.
        """
        if encoded is None:
            encoded = self._encode_videos(videos, training=False)

        """only needed if prompt_seq is 3D or above"""
        # encoded, prompt_seq = self._tile_vis_output(encoded, prompt_seq)

        """only needed if num_samples > 1"""
        # encoded = utils.tile_along_batch(encoded, num_samples)
        # prompt_seq = utils.tile_along_batch(prompt_seq, num_samples)

        pred_seq, logits = self.decoder.infer(
            prompt_seq, encoded, max_seq_len,
            temperature, top_k, top_p, sampling_callback)

        return pred_seq, logits, encoded


@model_lib.TrainerRegistry.register('video_encoder_ar_decoder')
class ARTrainer(model_lib.Trainer):
    """A trainer for Video AR model."""

    def __init__(self, config: ml_collections.ConfigDict, **kwargs):
        """Init and setup basic training elements under strategy scope.

        Note: the trainer needs to be created under `strategy.scope()`.

        Args:
          config: object for holding hyperparameters and other configurations.
          **kwargs: other neccesary configurations to pass for training setup.
        """
        super().__init__(config, **kwargs)
        self.vid_len = config.dataset.length
        self._category_names = task_utils.get_category_names(
            config.dataset.get('category_names_path'))

        self._metrics.update({
            'loss_notpad': tf.keras.metrics.Mean('loss_notpad'),
            'accuracy_notpad': tf.keras.metrics.SparseCategoricalAccuracy(
                'accuracy_notpad'),
        })

    def debug_loss(self, examples, y_true, y_pred_logits, y_mask):
        vocab_size = self.config.model.vocab_size

        y_pred = tf.argmax(y_pred_logits, axis=2)
        y_true_logits = tf.one_hot(y_true, depth=vocab_size)

        y_total = tf.cast(tf.size(y_true), tf.int64)
        y_correct = tf.math.equal(y_true, y_pred)
        y_correct_count = tf.reduce_sum(tf.cast(y_correct, tf.int64))
        y_correct_pc = (y_correct_count / y_total) * 100

        m = tf.keras.metrics.SparseCategoricalAccuracy()
        m.update_state(y_true, y_pred_logits)
        accuracy_notpad = m.result().numpy()

        y_true_m = tf.boolean_mask(y_true, y_mask)
        y_pred_logits_m = tf.boolean_mask(y_pred_logits, y_mask)

        # y_mask_int = tf.cast(y_mask, tf.int64)
        # y_mask_count = tf.reduce_sum(y_mask_int, axis=1)

        # y_true_logits_masked = tf.one_hot(y_true_m, depth=vocab_size)

        """Don't care about output tokens corresponding to GT tokens marked as padding"""
        y_total_m = tf.cast(tf.size(y_true_m), tf.int64)
        y_pred_m = tf.argmax(y_pred_logits_m, axis=1)
        y_correct_m = tf.math.equal(y_true_m, y_pred_m)
        y_correct_count_m = tf.reduce_sum(tf.cast(y_correct_m, tf.int64))
        y_correct_pc_m = (y_correct_count_m / y_total_m) * 100

        # y_cmb = tf.stack((y_true_m, y_pred_m, y_correct_int), axis=0)

        m = tf.keras.metrics.SparseCategoricalAccuracy()
        m.update_state(y_true_m, y_pred_logits_m)
        accuracy_notpad_m = m.result().numpy()

        vis_utils.visualize_video(self.config, examples, y_true_logits, y_true, 'GT raw',
                                  self._category_names)
        vis_utils.visualize_video(self.config, examples, y_pred_logits, y_pred, 'Pred raw',
                                  self._category_names)

        vis_utils.visualize_video(self.config, examples, y_true_logits, y_true, 'GT masked',
                                  self._category_names, y_mask)
        vis_utils.visualize_video(self.config, examples, y_pred_logits, y_pred, 'Pred masked',
                                  self._category_names, y_mask)

        print()

    def compute_loss(self, preprocess_outputs):
        batched_examples, input_seq, target_seq, token_weights = preprocess_outputs

        videos = batched_examples['video']

        target_seq = utils.flatten_batch_dims(target_seq, out_rank=2)
        token_weights = utils.flatten_batch_dims(token_weights, out_rank=2)
        token_weights = utils.tf_float32(token_weights)
        is_padding = tf.equal(target_seq, vocab.PADDING_TOKEN)  # padding tokens.
        token_weights_notpad = tf.where(
            is_padding, tf.zeros_like(token_weights), token_weights)

        logits = self.model(videos, input_seq)
        losses = model_utils.get_loss(
            logits, target_seq, self.config.train.loss_type)
        loss = tf.reduce_sum(losses * token_weights) / (
                tf.reduce_sum(token_weights) + 1e-9)
        loss_notpad = tf.reduce_sum(losses * token_weights_notpad) / (
                tf.reduce_sum(token_weights_notpad) + 1e-9)

        y_mask = tf.greater(token_weights_notpad, 0)

        y_true = tf.boolean_mask(target_seq, y_mask)
        y_pred_logits = tf.boolean_mask(logits, y_mask)

        # update metrics
        self._metrics['loss_notpad'].update_state(loss_notpad)
        self._metrics['accuracy_notpad'].update_state(y_true, y_pred_logits)

        # self.debug_loss(batched_examples, target_seq, logits, y_mask)

        return loss
