import os.path

from absl import logging

import time

import numpy as np
import pandas as pd

import utils


def check_ckpt_vars(cfg, trainer):
    cur_step_ = trainer.optimizer.iterations.numpy()
    ckpt_vars_pt = trainer.ckpt_vars_p

    if cur_step_ != 1 or ckpt_vars_pt is None:
        return

    name_to_shape_pt = trainer.name_to_shape_p
    trainer.checkpoint_manager.save(cur_step_)
    ckpt_vars, name_to_shape = utils.save_ckpt_vars(cfg.model_dir)

    ckpt_names_pt = set(k for k in ckpt_vars_pt['name'] if 'optimizer' not in k)
    ckpt_names = set(k for k in ckpt_vars['name'] if 'optimizer' not in k)

    ckpt_names_file = os.path.join(cfg.model_dir, 'ckpt_names.txt')
    with open(ckpt_names_file, 'w') as fid:
        fid.write('\n'.join(ckpt_names))

    ckpt_names_pt_file = os.path.join(cfg.model_dir, 'ckpt_names_pt.txt')
    with open(ckpt_names_pt_file, 'w') as fid:
        fid.write('\n'.join(ckpt_names_pt))

    unmatched_names_model = ckpt_names - ckpt_names_pt
    unmatched_names_pt = ckpt_names_pt - ckpt_names

    matched_names = ckpt_names.intersection(ckpt_names_pt)

    if unmatched_names_model:
        unmatched_names_model_file = os.path.join(cfg.model_dir, 'unmatched_names_model.txt')
        with open(unmatched_names_model_file, 'w') as fid:
            fid.write('\n'.join(unmatched_names_model))

    if unmatched_names_pt:
        unmatched_names_pt_file = os.path.join(cfg.model_dir, 'unmatched_names_pt.txt')
        with open(unmatched_names_pt_file, 'w') as fid:
            fid.write('\n'.join(unmatched_names_pt))

    unmatched_shapes = {
        name: (name_to_shape_pt[name], name_to_shape[name])
        for name in matched_names
        if name_to_shape_pt[name] != name_to_shape[name]
    }

    if unmatched_shapes:
        names = list(unmatched_shapes.keys())
        unmatched_shapes_dict = dict(
            name=names,
            shapes=[unmatched_shapes[name] for name in names]
        )

        import pandas as pd
        unmatched_shapes_df = pd.DataFrame.from_dict(unmatched_shapes_dict)

        unmatched_shapes_csv = os.path.join(cfg.model_dir, 'unmatched_shapes.csv')
        print(f'saving unmatched_shapes to {unmatched_shapes_csv}')
        unmatched_shapes_df.to_csv(
            unmatched_shapes_csv,
            index=False,
        )


def run(cfg, datasets, tasks, train_steps, steps_per_epoch, num_train_examples,
        strategy, model_lib, tf):
    if cfg.train.pt:
        assert cfg.pretrained, "cfg.pretrained must be provided to load pt and continue training from pretrained model"
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
            ts=tasks is just specifying the default value for optional arg ts
            strategy is needed to get the num_replicas_in_sync to divide the gradient and compute its mean
            """
            train_step = lambda xs, ts=tasks: trainer.train_step(xs, ts, strategy)

            """
            train_steps = num_samples * num_epochs / batch_size
            """
            progbar = None
            if cfg.eager:
                progbar = tf.keras.utils.Progbar(steps_per_epoch)

            # step_id = tf.constant(0)
            for _ in tf.range(steps_per_epoch):  # using tf.range prevents unroll.
                with tf.name_scope(''):  # prevent `while_` prefix for variable names.
                    strategy.run(train_step, ([next(it) for it in data_iterators],))

                for metric_name, metric_val in trainer.metrics.items():
                    metric_val_np = metric_val.result().numpy()
                    if np.isnan(metric_val_np):
                        logging.error(f'NaN value found for {metric_name} found so terminating training')
                        break

                if not cfg.eager:
                    continue

                metrics_np_dict = {
                    metric_name: metric_val.result().numpy() for metric_name, metric_val in trainer.metrics.items()
                }
                metric_val_df = pd.DataFrame.from_dict(metrics_np_dict)
                progbar.add(1)

                # check_ckpt_vars(cfg, trainer)
                # print()

        global_step = trainer.optimizer.iterations
        cur_step = global_step.numpy()
        timestamp = time.time()
        cur_epoch = 0
        # if not cfg.eager:
        #     print('compiling graph...')
        # trainer.checkpoint_manager.save(cur_step)
        # ckpt_vars_0 = utils.save_ckpt_vars(cfg.model_dir)

        while cur_step < train_steps:
            cur_epoch += 1
            tf.print(f'Training epoch {cur_epoch} with {steps_per_epoch} steps...')
            with summary_writer.as_default():
                train_multiple_steps(data_iterators, tasks)
                trainer.check_checkpoint_restored()

                cur_step = global_step.numpy()
                # if cfg.dist != 2 or cfg.worker_idx == 0:
                trainer.checkpoint_manager.save(cur_step)
                steps_per_sec = steps_per_epoch / (time.time() - timestamp)
                timestamp = time.time()
                with tf.name_scope('train'):
                    for metric_name, metric_val in trainer.metrics.items():
                        metric_val_np = metric_val.result().numpy()
                        if np.isnan(metric_val_np):
                            logging.error(f'NaN {metric_name} found so terminating training')
                            break
                        tf.summary.scalar(metric_name, metric_val_np, global_step)
                    lr = trainer.learning_rate(tf.cast(global_step, dtype=tf.float32))
                    tf.summary.scalar('lr', lr, global_step)
                    tf.summary.scalar('steps_per_sec', steps_per_sec, global_step)
                summary_writer.flush()
            progress = cur_step / float(train_steps) * 100
            eta = (train_steps - cur_step) / steps_per_sec / 60.
            logging.info(f'Completed steps {cur_step} / {train_steps} ({progress:.2f}%), ETA {eta:.2f} mins')
            trainer.reset()
        logging.info('###########################################')
        logging.info('Training complete...')
        logging.info('###########################################')
