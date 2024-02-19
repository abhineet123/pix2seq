import math
import re
import einops

import utils
from architectures import resnet
import tensorflow as tf

from architectures.transformers import (MLP, DropPath, get_shape, add_cls_token_emb,
                                        add_vis_pos_emb, suffix_id, TransformerEncoder)


class VideoTransformerEncoderLayer(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

    def __init__(self,
                 dim,
                 mlp_ratio,
                 num_heads,
                 vid_len,
                 drop_path=0.1,
                 drop_units=0.1,
                 drop_att=0.,
                 dim_x_att=None,
                 self_attention=True,
                 cross_attention=True,
                 use_mlp=True,
                 use_ffn_ln=False,
                 ln_scale_shift=True,
                 **kwargs):
        super(VideoTransformerEncoderLayer, self).__init__(**kwargs)

        self.vid_len = vid_len
        self.dim = dim

        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.use_mlp = use_mlp
        if self_attention:
            self.mha_ln = tf.keras.layers.LayerNormalization(
                epsilon=1e-6,
                center=ln_scale_shift,
                scale=ln_scale_shift,
                name='mha/ln')
            self.mha = tf.keras.layers.MultiHeadAttention(
                num_heads, dim // num_heads, dropout=drop_att, name='mha')
            if use_mlp:
                self.mlp = MLP(1, dim, mlp_ratio, drop_path, drop_units,
                                    use_ffn_ln=use_ffn_ln, ln_scale_shift=ln_scale_shift,
                                    name='mlp')
        if cross_attention:
            self.cross_ln = tf.keras.layers.LayerNormalization(
                epsilon=1e-6,
                center=ln_scale_shift,
                scale=ln_scale_shift,
                name='cross_mha/ln')
            dim_x_att = dim if dim_x_att is None else dim_x_att
            self.cross_mha = tf.keras.layers.MultiHeadAttention(
                num_heads, dim_x_att // num_heads,
                dropout=drop_att, name='cross_mha')
            if use_mlp:
                self.cross_mlp = MLP(
                    1, dim, mlp_ratio, drop_path, drop_units,
                    use_ffn_ln=use_ffn_ln, ln_scale_shift=ln_scale_shift,
                    name='cross_mlp')

        self.dropp = DropPath(drop_path)

    def call(self, x, mask, training):
        # x shape (bsz, vid_len, seq_len, dim_att), mask shape (bsz, seq_len, seq_len).
        bsz, vid_len, seq_len, dim_att = get_shape(x)

        assert vid_len == self.vid_len, "vid_len mismatch"

        if self.self_attention:
            x = utils.flatten_vid(x)
            x_ln = self.mha_ln(x)
            x_residual = self.mha(x_ln, x_ln, x_ln, attention_mask=mask, training=training)
            x = x + self.dropp(x_residual, training)
            if self.use_mlp:
                x = self.mlp(x, training)
            x = utils.unflatten_vid(x, self.vid_len)

        if self.cross_attention:
            x1 = x[:, 0, ...]
            for _id in range(1, self.vid_len):
                x2 = x[:, _id, ...]
                x1_ln = self.cross_ln(x1)
                x2_ln = self.cross_ln(x2)

                x_res = self.cross_mha(x1_ln, x2_ln, x2_ln, attention_mask=None, training=training)
                x1 = x1 + self.dropp(x_res, training)
            x = x1
            if self.use_mlp:
                x = self.cross_mlp(x, training)

        return x


class VideoTransformerEncoder(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

    def __init__(self,
                 num_layers,
                 dim,
                 mlp_ratio,
                 num_heads,
                 vid_len,
                 drop_path=0.1,
                 drop_units=0.1,
                 drop_att=0.,
                 self_attention=True,
                 use_ffn_ln=False,
                 ln_scale_shift=True,
                 **kwargs):
        super(VideoTransformerEncoder, self).__init__(**kwargs)

        assert vid_len >= 2, "vid_len must be >= 2"

        self.vid_len = vid_len
        self.num_layers = num_layers
        self.enc_layers = [
            VideoTransformerEncoderLayer(
                dim=dim,
                mlp_ratio=mlp_ratio,
                num_heads=num_heads,
                vid_len=vid_len,
                drop_path=drop_path,
                drop_units=drop_units,
                drop_att=drop_att,
                cross_attention=True if i == num_layers - 1 else False,
                self_attention=self_attention,
                use_ffn_ln=use_ffn_ln,
                ln_scale_shift=ln_scale_shift,
                name='transformer_encoder' + suffix_id(i))
            for i in range(num_layers)
        ]

    def call(self, x, mask, training, ret_list=False):
        if ret_list:
            x_list = [x]
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask, training)
            if ret_list:
                x_list.append(x)
        return (x, x_list) if ret_list else x


class VideoResNetTransformer(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

    def __init__(self,
                 image_height,
                 image_width,
                 vid_len,
                 resnet_variant,
                 resnet_depth,
                 resnet_width_multiplier,
                 resnet_sk_ratio,
                 num_layers,
                 dim,
                 mlp_ratio,
                 num_heads,
                 drop_path=0.1,
                 drop_units=0.1,
                 drop_att=0.,
                 pos_encoding='learned',
                 use_cls_token=True,
                 **kwargs):
        super(VideoResNetTransformer, self).__init__(**kwargs)
        self.vid_len = vid_len
        self.use_cls_token = use_cls_token
        self.resnet = resnet.resnet(
            resnet_depth=resnet_depth,
            width_multiplier=resnet_width_multiplier,
            sk_ratio=resnet_sk_ratio,
            variant=resnet_variant)
        self.dropout = tf.keras.layers.Dropout(drop_units)
        self.stem_projection = tf.keras.layers.Dense(dim, name='stem_projection')
        self.stem_ln = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name='stem_ln')
        if self.use_cls_token:
            add_cls_token_emb(self, dim)
        if resnet_variant in ['c3']:
            factor = 8.
        elif resnet_variant in ['c4', 'dc5']:
            factor = 16.
        else:
            factor = 32.
        self.n_rows = math.ceil(image_height / factor)
        self.n_cols = math.ceil(image_width / factor)
        self.vis_pos_emb = add_vis_pos_emb(
            self,
            pos_encoding,
            self.n_rows,
            self.n_cols,
            dim,
            return_only=True,
        )
        self.transformer_encoder = VideoTransformerEncoder(
            num_layers=num_layers,
            dim=dim,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            vid_len=vid_len,
            drop_path=drop_path,
            drop_units=drop_units,
            drop_att=drop_att,
            name='transformer_encoder')
        self.output_ln = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name='ouput_ln')

    def call_image(self, images, training):

        hidden_stack, _ = self.resnet(images, training)
        """last feature layer"""
        tokens = hidden_stack[-1]

        bsz, h, w, num_channels = get_shape(tokens)
        tokens = tf.reshape(tokens, [bsz, h * w, num_channels])
        tokens = self.stem_ln(self.stem_projection(self.dropout(tokens, training)))

        tokens_vis_pos_emb = tf.expand_dims(self.vis_pos_emb, 0)
        tokens = tokens + tokens_vis_pos_emb

        if self.use_cls_token:
            cls_token = tf.tile(tf.expand_dims(self.cls_token_emb, 0), [bsz, 1, 1])
            tokens = tf.concat([cls_token, tokens], 1)

        return tokens

    def call(self, images, training):
        images = utils.flatten_vid(images)

        tokens = self.call_image(images, training)

        tokens = utils.unflatten_vid(tokens, self.vid_len)

        tokens = self.transformer_encoder(
            tokens, None, training=training, ret_list=False)

        # tokens = utils.flatten_vid(tokens)
        x = self.output_ln(tokens)
        # x = utils.flatten_vid(x)

        return x

class VideoSwinTransformer(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

    def __init__(self,
                 image_height,
                 image_width,
                 vid_len,
                 num_layers,
                 dim,
                 mlp_ratio,
                 num_heads,
                 drop_path=0.1,
                 drop_units=0.1,
                 drop_att=0.,
                 pos_encoding='learned',
                 use_cls_token=True,
                 **kwargs):
        super(VideoSwinTransformer, self).__init__(**kwargs)
        self.vid_len = vid_len
        self.use_cls_token = use_cls_token
        self.backbone = tf.keras.models.load_model(
            'TFVideoSwinB_K400_IN1K_P244_W877_32x224', compile=False
        )
        self.dropout = tf.keras.layers.Dropout(drop_units)
        self.stem_projection = tf.keras.layers.Dense(dim, name='stem_projection')
        self.stem_ln = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name='stem_ln')
        if self.use_cls_token:
            add_cls_token_emb(self, dim)
        # if resnet_variant in ['c3']:
        #     factor = 8.
        # elif resnet_variant in ['c4', 'dc5']:
        #     factor = 16.
        # else:
        #     factor = 32.
        factor = 16.
        self.n_rows = math.ceil(image_height / factor)
        self.n_cols = math.ceil(image_width / factor)
        self.vis_pos_emb = add_vis_pos_emb(
            self,
            pos_encoding,
            self.n_rows,
            self.n_cols,
            dim,
            return_only=True,
        )
        self.transformer_encoder = TransformerEncoder(
            num_layers=num_layers,
            dim=dim,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            vid_len=vid_len,
            drop_path=drop_path,
            drop_units=drop_units,
            drop_att=drop_att,
            name='transformer_encoder')
        self.output_ln = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name='ouput_ln')

    def call(self, images, training):
        tokens = self.backbone(images)

        bsz, h, w, num_channels = get_shape(tokens)
        tokens = tf.reshape(tokens, [bsz, h * w, num_channels])
        tokens = self.stem_ln(self.stem_projection(self.dropout(tokens, training)))

        tokens_vis_pos_emb = tf.expand_dims(self.vis_pos_emb, 0)
        tokens = tokens + tokens_vis_pos_emb

        if self.use_cls_token:
            cls_token = tf.tile(tf.expand_dims(self.cls_token_emb, 0), [bsz, 1, 1])
            tokens = tf.concat([cls_token, tokens], 1)

        tokens = self.transformer_encoder(
            tokens, None, training=training, ret_list=False)

        # tokens = utils.flatten_vid(tokens)
        x = self.output_ln(tokens)
        # x = utils.flatten_vid(x)

        return x
