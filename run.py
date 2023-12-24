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

import collections
import copy
import json
import os
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time

env = dict(os.environ)
# print(env)

# exit()

from absl import app
from absl import flags
from absl import logging
import ml_collections
from ml_collections.config_flags import config_flags


# import paramparse
def build_tasks_and_datasets(
        cfg: ml_collections.ConfigDict,
        training: bool,
        task_lib):
    from data import dataset as dataset_lib

    """Build tasks and datasets.

    Args:
      cfg: Config.
      training: bool.

    Returns:
      tasks: a list of task objects.
      mixed_datasets: a list of tf.data.Dataset corresponding to tasks.
      last_dataset: the last dataset_lib.Dataset instance.
    """
    mixed_datasets = []
    tasks = []

    # There are N tasks and N datasets. The same task may appear multiple times
    # but corresponds to different datasets, e.g. [task1, task1, task2] and
    # [ds1, ds2, ds3]. In this case, we create one td.data.Dataset for task1,
    # sampling from ds1 and ds2 according to weights.
    # First we keep track of datasets and weights for each task:
    t_name_to_t_config_map = {}
    t_name_to_ds_config_map = collections.defaultdict(list)
    t_name_to_weights_map = collections.defaultdict(list)
    for t_config, ds_config in zip(cfg.tasks, cfg.datasets):
        if t_config.name not in t_name_to_t_config_map:
            t_name_to_t_config_map[t_config.name] = t_config
        else:
            # Accumulate weight for task.
            t_name_to_t_config_map[t_config.name].weight += t_config.weight
        t_name_to_weights_map[t_config.name].append(t_config.weight)
        t_name_to_ds_config_map[t_config.name].append(ds_config)

    # For each task, create the Task instance and the dataset instance.
    for t_name, t_config in t_name_to_t_config_map.items():
        task_config = copy.deepcopy(cfg)
        task_config.task = t_config
        task = task_lib.TaskRegistry.lookup(t_name)(cfg)
        tasks.append(task)

        ds_configs = t_name_to_ds_config_map[t_name]
        ds_weights = t_name_to_weights_map[t_name]
        ds_weights = [w / sum(ds_weights) for w in ds_weights]

        # Build dataset for this task.
        input_fns = []
        for ds_config in ds_configs:
            task_ds_config = copy.deepcopy(task_config)
            task_ds_config.dataset = ds_config
            ds_fn = dataset_lib.DatasetRegistry.lookup(ds_config.name)
            ds = ds_fn(task_ds_config)
            input_fn = ds.pipeline(
                process_single_example=task.preprocess_single,
                global_batch_size=(
                    cfg.train.batch_size if training else cfg.eval.batch_size
                ),
                training=training,
            )
            input_fns.append(input_fn)
        mixed_ds = dataset_lib.mix_datasets(input_fns, ds_weights)
        mixed_datasets.append(mixed_ds)

    return tasks, mixed_datasets, ds


def perform_evaluation(cfg, dataset, task, eval_steps, ckpt, strategy,
                       model_lib, tf):
    """Perform evaluation."""
    eval_tag = cfg.eval.tag
    summary_writer = None
    # summary_writer = tf.summary.create_file_writer(FLAGS.model_dir)

    with strategy.scope():
        # Restore model checkpoint.
        model = model_lib.ModelRegistry.lookup(cfg.model.name)(cfg)
        logging.info('Restoring from %s', ckpt)
        checkpoint = tf.train.Checkpoint(
            model=model, global_step=tf.Variable(0, dtype=tf.int64))
        checkpoint.restore(ckpt).expect_partial()  # Not restore optimizer.
        global_step = checkpoint.global_step
        logging.info('Performing eval at step %d', global_step.numpy())

    def single_step(examples):
        preprocessed_outputs = task.preprocess_batched(examples, training=False)
        infer_outputs = task.infer(model, preprocessed_outputs)
        postprocessed_outputs = task.postprocess_tpu(*infer_outputs)
        return postprocessed_outputs

    # from datetime import datetime
    # timestamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")

    # out_csv_dir_name = "csv"
    # val_json = config.val_filename_for_metrics
    # ckpt_dir = os.path.dirname(ckpt)
    ckpt_name = os.path.splitext(os.path.basename(ckpt))[0]
    json_name = os.path.splitext(os.path.basename(cfg.dataset.val_filename_for_metrics))[0]
    out_dir = os.path.join(cfg.model_dir, f'{ckpt_name}-{json_name}')

    out_csv_dir = out_vis_dir = None
    if cfg.eval.save_csv:
        csv_dir_name = f'csv'
        if cfg.eval.suffix:
            csv_dir_name = f'{csv_dir_name}-{cfg.eval.suffix}'
        out_csv_dir = os.path.join(out_dir, csv_dir_name)
        print(f'\nwriting csv files to: {out_csv_dir}\n')
        os.makedirs(out_csv_dir, exist_ok=True)

    if cfg.eval.save_vis:
        vis_dir_name = f'vis'
        if cfg.eval.suffix:
            vis_dir_name = f'{vis_dir_name}-{cfg.eval.suffix}'
        out_vis_dir = os.path.join(out_dir, vis_dir_name)
        os.makedirs(out_vis_dir, exist_ok=True)

        print(f'\nwriting vis images to: {out_vis_dir}\n')

    with strategy.scope():
        @tf.function
        def run_single_step(dataset_iter):
            examples = next(dataset_iter)
            # outputs = single_step(examples)
            outputs = strategy.run(single_step, (examples,))
            if outputs is not None:
                outputs = [strategy.gather(t, axis=0) for t in outputs]
            return outputs

        iterator = iter(dataset)
        start_time = timestamp = time.time()
        cur_step = 0
        img_id = 0
        seq_to_csv_rows = collections.defaultdict(list)

        while True:
            if eval_steps and cur_step >= eval_steps:
                break
            # try:
            # with summary_writer.as_default():

            per_step_outputs = run_single_step(iterator)
            vis_images = task.postprocess_cpu(
                per_step_outputs,
                train_step=global_step.numpy(),
                out_vis_dir=out_vis_dir,
                csv_data=seq_to_csv_rows,
                eval_step=cur_step,
                summary_tag=eval_tag,
                ret_results=True)

            cur_step += 1
            if eval_steps:
                steps_per_sec = 1. / (time.time() - timestamp)
                timestamp = time.time()
                progress = cur_step / float(eval_steps) * 100
                eta = (eval_steps - cur_step) / steps_per_sec / 60.
                logging.info('Completed: {} / {} steps ({:.2f}%), ETA {:.2f} mins'
                             ''.format(cur_step, eval_steps, progress, eta))
            else:
                logging.info('Completed: %d steps', cur_step)
            # except tf.errors.OutOfRangeError:
            #     logging.info('Break due to OutOfRangeError exception')
            #     break
        logging.info('Finished eval in %.2f mins', (time.time() - start_time) / 60.)

    if cfg.eval.save_csv:
        csv_columns = [
            "ImageID", "LabelName",
            "XMin", "XMax", "YMin", "YMax", "Confidence",
        ]
        # if params.enable_mask:
        #     csv_columns += ['mask_w', 'mask_h', 'mask_counts']
        for csv_seq_name, csv_rows in seq_to_csv_rows.items():
            if not csv_rows:
                print(f'{csv_seq_name}: no csv data found')
                continue
            out_csv_name = f"{csv_seq_name}.csv"
            out_csv_path = os.path.join(out_csv_dir, out_csv_name)
            # print(f'{csv_seq_name} :: saving csv to {out_csv_path}')
            df = pd.DataFrame(csv_rows, columns=csv_columns)
            df.to_csv(out_csv_path, index=False)
        result = {
            'global_step': cur_step
        }
    else:
        # Write summaries and record results as JSON.
        cur_step = global_step.numpy()
        result = task.evaluate(summary_writer, cur_step, eval_tag)

        result.update({'global_step': cur_step})
        # logging.info(result)

        result_json_path = os.path.join(cfg.model_dir, eval_tag + '_result.json')
        with tf.io.gfile.GFile(result_json_path, 'w') as f:
            json.dump({k: float(v) for k, v in result.items()}, f)
        result_json_path = os.path.join(
            cfg.model_dir, eval_tag + 'result_%d.json' % result['global_step'])
        with tf.io.gfile.GFile(result_json_path, 'w') as f:
            json.dump({k: float(v) for k, v in result.items()}, f)

    return result


def perform_training(cfg, datasets, tasks, train_steps, steps_per_epoch, num_train_examples,
                     strategy, model_lib, tf):
    if cfg.pretrained:
        cfg.model.pretrained_ckpt = cfg.pretrained
    """Main training logic."""
    with strategy.scope():
        # Setup training elements.
        trainer = model_lib.TrainerRegistry.lookup(cfg.model.name)(
            cfg, model_dir=cfg.model_dir,
            num_train_examples=num_train_examples, train_steps=train_steps)
        data_iterators = [iter(dataset) for dataset in datasets]
        summary_writer = tf.summary.create_file_writer(cfg.model_dir)

        @tf.function
        def train_multiple_steps(data_iterators, tasks):
            """
            ts=tasks is just the default value for arg ts
            strategy is needed to get the num_replicas_in_sync to divide the gradient and compute its mean
            """
            train_step = lambda xs, ts=tasks: trainer.train_step(xs, ts, strategy)

            """
            train_steps = num_samples * num_epochs / batch_size
            """
            # temp=tf.TensorArray(size=1, dynamic_size=True, dtype=tf.int32)
            progbar = None
            if cfg.eager:
                progbar = tf.keras.utils.Progbar(steps_per_epoch)
            # step_id = tf.constant(0)
            for _ in tf.range(steps_per_epoch):  # using tf.range prevents unroll.
                with tf.name_scope(''):  # prevent `while_` prefix for variable names.
                    strategy.run(train_step, ([next(it) for it in data_iterators],))
                if cfg.eager:
                    progbar.add(1)
                # else:
                #     temp = temp.write(i, 1)
                #     tf.print(f'done step {int(temp.size())}')
                # if not cfg.eager:
                #     step_id += 1
                #     with tf.Session():
                #         step_id_val = step_id.eval()
                #     tf.print(f'done step {int(step_id_val)}/{int(steps_per_epoch)}')

        global_step = trainer.optimizer.iterations
        cur_step = global_step.numpy()
        timestamp = time.time()
        cur_epoch = 0
        # if not cfg.eager:
        #     print('compiling graph...')
        while cur_step < train_steps:
            cur_epoch += 1
            tf.print(f'Training epoch {cur_epoch} with {steps_per_epoch} steps...')
            with summary_writer.as_default():
                train_multiple_steps(data_iterators, tasks)
                trainer.check_checkpoint_restored()
                cur_step = global_step.numpy()
                trainer.checkpoint_manager.save(cur_step)
                steps_per_sec = steps_per_epoch / (time.time() - timestamp)
                timestamp = time.time()
                with tf.name_scope('train'):
                    for metric_name, metric_val in trainer.metrics.items():
                        tf.summary.scalar(metric_name, metric_val.result().numpy(), global_step)
                    lr = trainer.learning_rate(tf.cast(global_step, dtype=tf.float32))
                    tf.summary.scalar('lr', lr, global_step)
                    tf.summary.scalar('steps_per_sec', steps_per_sec, global_step)
                summary_writer.flush()
            progress = cur_step / float(train_steps) * 100
            eta = (train_steps - cur_step) / steps_per_sec / 60.
            print(f'Completed steps {cur_step} / {train_steps} ({progress:.2f}%), ETA {eta:.2f} mins')
            trainer.reset()
        logging.info('###########################################')
        logging.info('Training complete...')
        logging.info('###########################################')


def load_cfg_from_model(cfg, model_dir, cmd_cfg):
    pt_cfg_filepath = os.path.join(model_dir, 'config.json')

    assert os.path.isfile(pt_cfg_filepath), f"non-existent pretrained cfg json: {pt_cfg_filepath}"

    print(f'loading model cfg from {pt_cfg_filepath}')
    with open(pt_cfg_filepath, 'r') as f:
        cfg_pt = json.loads(f.read())

    """
    hack to deal with type mismatches between variables in the config py files and those in the 
    config json files accompanying the pre-trained models
    ConfigDict does not allow type override so type changes must be done in ordinary dict
    """
    image_size = cfg_pt['model']['image_size']
    if isinstance(image_size, int):
        cfg_pt['model']['image_size'] = (image_size, image_size)

    image_size = cfg_pt['task']['image_size']
    if isinstance(image_size, int):
        cfg_pt['task']['image_size'] = (image_size, image_size)

    cfg_pt = ml_collections.ConfigDict(cfg_pt)

    cfg.model.update(cfg_pt.model)
    cfg.task.update(cfg_pt.task)
    cfg.train.update(cfg_pt.train)
    cfg.optimization.update(cfg_pt.optimization)

    """
    hack to deal with independently defined target_size setting in tasks.eval_transforms even though it should match 
    image_size
    """
    image_size = cfg.task.image_size

    if cfg.task.name == 'object_detection':
        from configs import transform_configs
        train_transforms_fn = transform_configs.get_object_detection_train_transforms
        eval_transforms_fn = transform_configs.get_object_detection_eval_transforms
    else:
        raise AssertionError('unsupported task: {cfg.task.name}')

    for task in cfg.tasks:
        try:
            eval_transforms = task.eval_transforms
        except AttributeError:
            pass
        else:
            task.eval_transforms = eval_transforms_fn(image_size, task.max_instances_per_image_test)
        try:
            train_transforms = task.train_transforms
        except AttributeError:
            pass
        else:
            task.train_transforms = train_transforms_fn(image_size, task.max_instances_per_image)

    if cmd_cfg:
        cfg.update(cmd_cfg)


MAX_JSON_VARS = 10


def load_cfg_from_json5(json_list, json_root):
    all_cfg = ml_collections.ConfigDict()
    for json_data in json_list:
        json_vars = json_data.split('-')
        json_path = json_vars[0] + '.json5'
        if json_root:
            json_path = os.path.join(json_root, json_path)

        assert os.path.isfile(json_path), f"non-existent cfg json: {json_path}"
        # print(f'loading json cfg from {json_path}')
        with open(json_path, 'r') as f:
            json_str = f.read()
        for var_id, json_var in enumerate(json_vars[1:]):
            var_id_ = var_id
            # if ':' in json_var:
            #     var_id_, json_var = json_var.split(':')
            """optional vars"""
            json_str = json_str.replace(f'$${var_id_}$$', json_var)
            """compulsory vars"""
            json_str = json_str.replace(f'${var_id_}$', json_var)

        """
        remove lines with under specified optional vars
        json5 is needed to deal with trailing commas
        """
        json_lines = json_str.splitlines()
        valid_line_ids = []
        for line_id, json_line in enumerate(json_lines):
            if any(f'$${var_id}$$' in json_line for var_id in range(MAX_JSON_VARS)):
                continue
            valid_line_ids.append(line_id)
        json_str = '\n'.join(json_lines[i] for i in valid_line_ids)

        import json5

        json_dict = json5.loads(json_str)
        cfg_json = ml_collections.ConfigDict(json_dict)
        all_cfg.update(cfg_json)
    return all_cfg


TRAIN = 'train'
EVAL = 'eval'
config_flags.DEFINE_config_file('cfg', 'path/to/config/file.py', 'The config file.', lock_config=False)
flags.DEFINE_list('j5', [], 'list of config json5 files to override settings from default and pretrained configs')
flags.DEFINE_string('cluster', 'cluster.json', 'cluster_cfg')
flags.DEFINE_string('j5_root', 'configs/json', 'relative path of the folder containing the optional json files')
flags.DEFINE_integer('worker_id', 0, 'worker id for multi-machine training')


def load_cfg(cfg, FLAGS):
    cmd_cfg = load_cfg_from_json5(FLAGS.j5, FLAGS.j5_root)

    cfg.update(cmd_cfg)
    cfg.training = cfg.mode == TRAIN

    if cfg.pretrained:
        load_cfg_from_model(cfg, cfg.pretrained, cmd_cfg)

    if not cfg.model_dir:
        if not cfg.training and cfg.eval.pt:
            assert cfg.pretrained, "cfg.pretrained must be provided for pretrained model eval"

            cfg.model_dir = cfg.pretrained.replace('pretrained', 'log')
        else:
            model_dir_name = f'{cfg.dataset.train_name}_batch_{cfg.train.batch_size}'
            if cfg.pretrained:
                pretrained_name = os.path.basename(cfg.pretrained)
                model_dir_name = f'{pretrained_name}_{model_dir_name}'

            if cfg.train.suffix:
                model_dir_name = f'{model_dir_name}-{cfg.train.suffix}'

            if cfg.dist == 2 and cfg.dist2.task.index > 0:
                model_dir_name = f'{model_dir_name}-worker-{cfg.dist2.task.index}'

            cfg.model_dir = os.path.join('log', model_dir_name)

    if cfg.training:
        print(f'saving trained model to: {cfg.model_dir}')
    else:
        print(f'loading trained model from: {cfg.model_dir}')
        if not cfg.eval.pt:
            load_cfg_from_model(cfg, cfg.model_dir, cmd_cfg)

    # config_cmd_args = [k for k in dir(FLAGS) if k.startswith('cfg.')]
    # config_cmd_dict = {
    #     k: getattr(FLAGS, k) for k in dir(FLAGS) if k.startswith('cfg.')
    # }
    if cfg.dataset.name.startswith('ipsc'):
        from configs.dataset_configs import ipsc_post_process
        ipsc_post_process(cfg.dataset)

    import utils
    utils.log_cfg(cfg)


def main(unused_argv):
    # params = Params()
    # paramparse.process(params)
    FLAGS = flags.FLAGS
    cfg = FLAGS.cfg

    load_cfg(cfg, FLAGS)

    if cfg.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

    if cfg.dist == 2:
        tf_config = cfg.dist2.to_dict()

        worker_ip_addresses = [node.split(':')[0] for node in tf_config['cluster']['worker']]

        # import socket
        # hostname = socket.gethostname()
        # IPAddr = socket.gethostbyname(hostname)
        # print("Your Computer Name is:" + hostname)
        # print("Your Computer IP Address is:" + IPAddr)

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

        os.environ.pop('TF_CONFIG', None)
        os.environ['TF_CONFIG'] = json.dumps(tf_config)

    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        if cfg.dyn_ram:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                raise e
        if cfg.dist != 2:
            """
            some weird and annoying conflicts between MultiWorkerMirroredStrategy init and gpu setup
            resulting in catch-22 type situation where strategy must be inited before gpu setup and 
            gpu setup cannot be done after strategy init
            """
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    tf.config.set_soft_device_placement(True)

    import utils

    strategy = utils.build_strategy(cfg.dist, cfg.use_tpu, cfg.master)

    # tf.logging.set_verbosity(tf.logging.ERROR)

    if cfg.debug:
        tf.data.experimental.enable_debug_mode()

    if cfg.eager:
        tf.config.run_functions_eagerly(True)
    else:
        tf.config.run_functions_eagerly(False)

    """
    all these unused imports needed to register the various modules
    """
    from data import datasets  # pylint: disable=unused-import
    from data import transforms  # pylint: disable=unused-import
    from metrics import coco_metrics  # pylint: disable=unused-import
    from models import ar_model  # pylint: disable=unused-import
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
    # pylint: enable=unused-import
    from tasks import task as task_lib

    with strategy.scope():
        # Allow cfg override: for eval, only take one task and one dataset.
        if 'tasks' not in cfg or len(cfg.tasks) == 1 or not cfg.training:
            cfg.tasks = [cfg.task]
        if 'datasets' not in cfg or len(cfg.datasets) == 1 or not cfg.training:
            cfg.datasets = [cfg.dataset]

        """dataset is simply the last dataset"""
        tasks, dses, dataset = build_tasks_and_datasets(cfg, cfg.training, task_lib)

        # Calculate steps stuff using last task info (assuming all tasks the same.)
        train_steps = utils.get_train_steps(
            dataset, cfg.train.steps, cfg.train.epochs,
            cfg.train.batch_size)
        eval_steps = utils.get_eval_steps(
            dataset, cfg.eval.steps, cfg.eval.batch_size)
        checkpoint_steps = utils.get_checkpoint_steps(
            dataset, cfg.train.checkpoint_steps,
            cfg.train.checkpoint_epochs, cfg.train.batch_size)
        checkpoint_steps = min(checkpoint_steps, train_steps)

    if cfg.training:
        perform_training(cfg, dses, tasks, train_steps, checkpoint_steps,
                         dataset.num_train_examples, strategy, model_lib, tf)
    else:
        # For eval, only one task and one dataset is passed in.
        assert len(dses) == 1, 'Only one dataset is accepted in eval.'
        assert len(tasks) == 1, 'Only one task is accepted in eval.'

        checkpoint_dir = cfg.eval.get('checkpoint_dir', None)

        assert cfg.model_dir, "cfg.model_dir must be provided"

        if not checkpoint_dir:
            if cfg.eval.pt:
                checkpoint_dir = cfg.pretrained
            else:
                checkpoint_dir = cfg.model_dir

        for ckpt in tf.train.checkpoints_iterator(
                checkpoint_dir, min_interval_secs=1, timeout=1):
            result = perform_evaluation(cfg, dses[0], tasks[0], eval_steps, ckpt, strategy,
                                        model_lib, tf)
            if result['global_step'] >= train_steps:
                logging.info('Eval complete. Exiting...')
                break


if __name__ == '__main__':
    app.run(main)
