import os.path

from absl import logging
import time

import utils


def run(cfg, datasets, tasks, train_steps, steps_per_epoch, num_train_examples,
        strategy, model_lib, tf):
    if cfg.train.pt:
        assert cfg.pretrained, "cfg.pretrained must be provided to load pt and continue training from pretrained model"
        cfg.model.pretrained_ckpt = cfg.pretrained
    """Main training logic."""
    with (strategy.scope()):
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
        trainer.checkpoint_manager.save(cur_step)
        ckpt_vars_0 = utils.save_ckpt_vars(cfg.model_dir)
        ckpt_vars_pt = trainer.ckpt_vars_p
        name_to_shape_pt = trainer.name_to_shape_p

        ckpt_names_pt = set(ckpt_vars_pt['name'])

        while cur_step < train_steps:
            cur_epoch += 1
            tf.print(f'Training epoch {cur_epoch} with {steps_per_epoch} steps...')
            with summary_writer.as_default():
                trainer.check_checkpoint_restored()
                train_multiple_steps(data_iterators, tasks)

                cur_step = global_step.numpy()
                # if cfg.dist != 2 or cfg.worker_idx == 0:
                trainer.checkpoint_manager.save(cur_step)
                ckpt_vars, name_to_shape = utils.save_ckpt_vars(cfg.model_dir)

                if ckpt_vars_pt is not None:
                    ckpt_names = set(ckpt_vars['name'])

                    unmatched_names_model = ckpt_names - ckpt_names_pt
                    unmatched_names_pt = ckpt_names_pt - ckpt_names

                    matched_names = ckpt_names.intersection(ckpt_names_pt)

                    unmatched_shapes = {
                        name: (name_to_shape_pt[name], name_to_shape[name])
                        for name in matched_names
                        if name_to_shape_pt[name] != name_to_shape[name]
                    }

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
