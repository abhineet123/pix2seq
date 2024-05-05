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
"""Train and eval script."""

import json
import os
import sys

sys.path.append("/home/abhineet/ipsc/ipsc_data_processing")

# env = dict(os.environ)
# print(env)

# exit()

from absl import app
from absl import logging
from absl import flags
from ml_collections.config_flags import config_flags

config_flags.DEFINE_config_file('cfg', '', 'The config file.', lock_config=False)
flags.DEFINE_list('j5', [], 'list of config json5 files to override settings from default and pretrained configs')
flags.DEFINE_string('cluster', 'cluster.json', 'cluster_cfg')
flags.DEFINE_string('j5_root', 'configs/j5', 'relative path of the folder containing the optional json files')
flags.DEFINE_integer('worker_id', 0, 'worker id for multi-machine training')
FLAGS = flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from utils import to_numpy

from tensorflow.python.framework.ops import EagerTensor

EagerTensor.to_numpy = to_numpy


# temp1 = tf.random.uniform((5, 5))
# temp2 = tf.random.uniform((5, 5))
# temp3 = tf.random.uniform((5, 5), name='temp3')


def get_worker_id(tf_config):
    worker_ip_addresses = [node.split(':')[0] for node in tf_config['cluster']['worker']]

    assert len(worker_ip_addresses) == len(set(worker_ip_addresses)), \
        "worker ID resolution cannot work with duplicate IP addresses"
    import netifaces as ni

    interfaces = ni.interfaces()
    self_ip_addresses = ''
    for interface in interfaces:
        ifaddresses = ni.ifaddresses(interface)
        try:
            ip = ifaddresses[ni.AF_INET][0]['addr']
        except KeyError:
            keys = list(ifaddresses.keys())
            ip = ifaddresses[keys[0]][0]['addr']
        try:
            worker_idx = worker_ip_addresses.index(ip)

        except ValueError:
            self_ip_addresses += f'{interface}: {ip}\n'
            continue
        else:
            print(f'found worker_idx: {worker_idx}')
            tf_config['task']['index'] = worker_idx
            # exit()
            break
    else:
        raise AssertionError(f'No matching ip address found\n'
                             f'worker_ip_addresses:\n{worker_ip_addresses}\n'
                             f'self_ip_addresses:\n{self_ip_addresses}')


def main(unused_argv):
    # filenames = [
    #     'datasets/ipsc/well3/all_frames_roi/all_frames_roi_12094_17082_16427_20915/image146.jpg'
    #     'datasets/ipsc/well3/all_frames_roi/all_frames_roi_12094_17082_16427_20915/image147.jpg'
    # ]

    # frames = tf.map_fn(
    #     lambda x: tf.io.decode_image(tf.io.read_file(x), channels=3),
    #     # read_video_frames,
    #     filenames,
    #     fn_output_signature=tf.uint8
    # )
    #
    # # img = tf.io.decode_image(tf.io.read_file(x), channels=3)
    # img_shape_1 = frames.shape
    # img_shape_2 = tf.shape(frames)

    # params = Params()
    # paramparse.process(params)
    assert FLAGS.cfg, "cfg must be provided"

    import config

    cfg = config.load(FLAGS)

    # if cfg.gpu:
    #     print(f'setting CUDA_VISIBLE_DEVICES to {cfg.gpu}')
    #     os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

    # is_debugging = int(os.environ.get('P2S_DEBUGGING_MODE', 0))
    # if is_debugging:
    #     cfg.debug = 1
    #     cfg.dist = 0

    if not cfg.debug:
        import sys
        gettrace = getattr(sys, 'gettrace', None)
        if gettrace is None:
            print('No sys.gettrace')
        elif gettrace():
            print('running in pycharm debugger')
            cfg.debug = 1
            cfg.eager = 1
            cfg.dyn_ram = 1
            cfg.eval.profile = 1
            cfg.dist = 0

    if cfg.dist == 2:
        tf_config = cfg.tf_config.to_dict()
        if tf_config['task']['index'] < 0:
            get_worker_id(tf_config)

        os.environ.pop('TF_CONFIG', None)
        os.environ['TF_CONFIG'] = json.dumps(tf_config)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        if cfg.dyn_ram:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                raise e
        # if cfg.dist != 2:
        #     """
        #     some weird and annoying conflicts between MultiWorkerMirroredStrategy init and gpu setup
        #     resulting in catch-22 type situation where strategy must be inited before gpu setup and
        #     gpu setup cannot be done after strategy init
        #     """
        #     logical_gpus = tf.config.list_logical_devices('GPU')
        #     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    tf.config.set_soft_device_placement(True)

    import utils

    strategy = utils.build_strategy(cfg.dist, cfg.use_tpu, cfg.master)

    # tf.logging.set_verbosity(tf.logging.ERROR)

    if cfg.debug:
        tf.data.experimental.enable_debug_mode()
        # tf.debugging.set_log_device_placement(True)

    if cfg.eager:
        tf.config.run_functions_eagerly(True)
    else:
        tf.config.run_functions_eagerly(False)

    # tf.config.set_visible_devices(gpus[1:], 'GPU')
    # logical_devices = tf.config.list_logical_devices('GPU')
    # assert len(logical_devices) == len(gpus) - 1

    """
    all these unused imports needed to register the various modules
    """
    from data import datasets  # pylint: disable=unused-import
    from data import transforms, video_transforms  # pylint: disable=unused-import
    from metrics import coco_metrics  # pylint: disable=unused-import
    from models import ar_model  # pylint: disable=unused-import
    from models import video_ar_model  # py6lint: disable=unused-import
    from models import image_ar_model  # pylint: disable=unused-import
    from models import image_diffusion_model  # pylint: disable=unused-import
    # from models import latent_diffusion_model  # pylint: disable=unused-import
    from models import video_diffusion_model  # pylint: disable=unused-import
    from models import image_discrete_diffusion_model  # pylint: disable=unused-import
    from models import model as model_lib
    from models import panoptic_diffusion  # pylint: disable=unused-import
    # pylint: disable=unused-import
    from tasks import captioning
    # from tasks import image_generation
    from tasks import instance_segmentation
    # from tasks import keypoint_detection
    from tasks import object_detection
    from tasks import video_detection
    from tasks import semantic_segmentation
    # pylint: enable=unused-import
    from tasks import task as task_lib

    with strategy.scope():
        # Allow cfg override: for eval, only take one task and one dataset.
        if 'tasks' not in cfg or len(cfg.tasks) == 1 or not cfg.training:
            cfg.tasks = [cfg.task]
        if 'datasets' not in cfg or len(cfg.datasets) == 1 or not cfg.training:
            cfg.datasets = [cfg.dataset]

        """dataset is simply the last entry in datasets"""
        tasks, train_datasets, train_dataset = config.build_tasks_and_datasets(
            cfg, training=cfg.training, validation=False, task_lib=task_lib)
        # Calculate steps stuff using last task info (assuming all tasks the same.)
        train_steps = utils.get_train_steps(
            train_dataset, cfg.train.steps, cfg.train.epochs,
            cfg.train.batch_size)

    if cfg.training:
        checkpoint_steps = utils.get_checkpoint_steps(
            train_dataset, cfg.train.checkpoint_steps,
            cfg.train.checkpoint_epochs, cfg.train.batch_size)
        checkpoint_steps = min(checkpoint_steps, train_steps)

        print()
        print(f'train.steps: {cfg.train.steps}')
        print(f'train.epochs: {cfg.train.epochs}')
        print(f'train.batch_size: {cfg.train.batch_size}')
        print(f'train_steps: {train_steps}')
        print(f'checkpoint_steps: {checkpoint_steps}')

        val_datasets = val_steps = None
        if cfg.train.val_epochs:
            _, val_datasets, val_dataset = config.build_tasks_and_datasets(
                cfg, training=False, validation=True, task_lib=task_lib)

            val_steps = utils.get_val_steps(
                val_dataset, cfg.eval.steps, cfg.eval.batch_size)

            print()
            print(f'val.steps: {cfg.eval.steps}')
            print(f'val.batch_size: {cfg.eval.batch_size}')
            print(f'val_steps: {val_steps}')
            print()

        import train
        train.run(cfg, train_datasets, val_datasets, tasks, train_steps, val_steps,
                  checkpoint_steps, train_dataset.num_train_examples, strategy, model_lib, tf)
    else:
        eval_steps = utils.get_eval_steps(
            train_dataset, cfg.eval.steps, cfg.eval.batch_size)

        print()
        print(f'num_eval_examples: {train_dataset.num_eval_examples}')
        print(f'cfg.eval.steps: {cfg.eval.steps}')
        print(f'cfg.eval.batch_size: {cfg.eval.batch_size}')
        print(f'eval_steps: {eval_steps}')

        print()
        # For eval, only one task and one dataset is passed in.
        assert len(train_datasets) == 1, 'Only one dataset is accepted in eval.'
        assert len(tasks) == 1, 'Only one task is accepted in eval.'

        checkpoint_dir = cfg.eval.get('checkpoint_dir', None)

        assert cfg.model_dir, "cfg.model_dir must be provided"

        if not checkpoint_dir:
            if cfg.eval.pt:
                assert cfg.pretrained, "pretrained dir must be provided for pretrained model eval"

                checkpoint_dir = cfg.pretrained
            else:
                assert cfg.model_dir, "model dir must be provided for trained model eval"

                if cfg.eval_type:
                    cfg.model_dir = os.path.join(cfg.model_dir, cfg.eval_type)

                checkpoint_dir = cfg.model_dir
        import eval
        for ckpt in tf.train.checkpoints_iterator(
                checkpoint_dir, min_interval_secs=1, timeout=5):
            csv_dir_name = eval.run(cfg, train_datasets[0], tasks[0], eval_steps, ckpt, strategy,
                              model_lib, tf)
            # if cfg.eval.pipeline:
            #     continue

            logging.info('Eval complete. Exiting...')
            break
        else:
            raise AssertionError(f'no checkpoints found in {checkpoint_dir}')


if __name__ == '__main__':
    app.run(main)
