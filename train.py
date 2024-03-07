import os.path

from absl import logging

import time


import numpy as np

import eval

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

                # tf.print(f'\rstep {step_id}')

                # nan_metric = 0
                # for metric_name, metric_val in trainer.metrics.items():
                #     metric_val = metric_val.result()
                #     if tf.math.is_nan(metric_val):
                #         step = trainer.optimizer.iterations
                #         logging.error(f'NaN value found for {metric_name} in step {step} so terminating training')
                #         nan_metric = 1
                #
                # if nan_metric:
                #     break

                if not cfg.eager:
                    continue

                # rows = tuple(trainer.metrics.keys())
                # cols = ('val',)
                #
                # metric_val_df = pd.DataFrame(np.zeros((len(rows), len(cols)), dtype=object), index=rows, columns=cols)
                # for metric_name, metric_val in trainer.metrics.items():
                #     metric_val_np = metric_val.result().numpy()
                #     metric_val_df.loc[metric_name, 'val'] = metric_val_np
                # metric_val_df = pd.DataFrame({
                #     metric_name: metric_val.result().numpy().item() for metric_name, metric_val in
                #     trainer.metrics.items()
                # }, index=[0])

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

                """
                this check happens after the first forward pass because of deferred restoration 
                in tf.train.Checkpoint
                which only restores many of the variables after they are created in the first call 
                when the input shape becomes available
                """
                trainer.check_checkpoint_restored()

                cur_step = global_step.numpy()

                trainer.checkpoint_manager.save(cur_step)
                with tf.name_scope('val'):
                    ckpt = trainer.checkpoint_manager.latest_checkpoint
                    cfg.eval.save_csv = cfg.eval.save_vis = False
                    result = eval.run(cfg, datasets[0], tasks[0], cfg.eval.steps, ckpt, strategy, model_lib, tf)

                steps_per_sec = steps_per_epoch / (time.time() - timestamp)
                timestamp = time.time()
                with tf.name_scope('train'):
                    nan_metric = 0
                    for metric_name, metric_val in trainer.metrics.items():
                        metric_val_np = metric_val.result().numpy()
                        if np.isnan(metric_val_np):
                            logging.error(f'NaN value found for {metric_name} so terminating training')
                            nan_metric = 1
                            break
                        tf.summary.scalar(metric_name, metric_val_np, global_step)
                    if nan_metric:
                        break
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
