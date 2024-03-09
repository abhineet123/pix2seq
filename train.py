import os.path

from absl import logging

import time

import numpy as np


def run(cfg, train_datasets, val_datasets, tasks, train_steps, val_steps, steps_per_epoch, num_train_examples,
        strategy, model_lib, tf):
    if cfg.train.pt:
        assert cfg.pretrained, "cfg.pretrained must be provided to load pt and continue training from pretrained model"
        cfg.model.pretrained_ckpt = cfg.pretrained

    def single_val_step(examples, model):
        preprocessed_outputs = [
            t.preprocess_batched(e, training=False) for e, t in zip(examples, tasks)]
        val_outputs = [t.infer(model, o) for o, t in zip(preprocessed_outputs, tasks)]
        return val_outputs

    with strategy.scope():
        trainer = model_lib.TrainerRegistry.lookup(cfg.model.name)(
            cfg, model_dir=cfg.model_dir,
            num_train_examples=num_train_examples, train_steps=train_steps)
        train_data_iters = [iter(dataset) for dataset in train_datasets]
        val_data_iters = [iter(dataset) for dataset in val_datasets]
        summary_writer = tf.summary.create_file_writer(cfg.model_dir)

        is_greater = lambda x, y: x > y
        is_smaller = lambda x, y: x < y
        is_better = dict(
            loss=is_smaller,
            loss_notpad=is_smaller,
            correct_pc=is_greater,
            accuracy_notpad=is_greater,
        )
        best_val_metrics = dict(
            loss=np.inf,
            loss_notpad=np.inf,
            correct_pc=0,
            accuracy_notpad=0,
        )

        val_ckpt_managers = {
            metric_name: tf.train.CheckpointManager(
                tf.train.Checkpoint(model=trainer.model), os.path.join(cfg.model_dir, f'best-val-{metric_name}'), 1)
            for metric_name in best_val_metrics.keys()
        }

        @tf.function
        def validate(data_iterators):
            progbar = None
            if cfg.eager:
                progbar = tf.keras.utils.Progbar(val_steps)

            loss = tf.TensorArray(tf.float32, size=val_steps)
            loss_notpad = tf.TensorArray(tf.float32, size=val_steps)
            correct_pc = tf.TensorArray(tf.float32, size=val_steps)
            accuracy_notpad = tf.TensorArray(tf.float32, size=val_steps)

            for step_id in tf.range(val_steps):
                with tf.name_scope(''):
                    val_outputs = strategy.run(single_val_step, ([next(it) for it in data_iterators], trainer.model))
                    if val_outputs is not None:
                        val_outputs = [strategy.gather(o, axis=0) for o in val_outputs]
                    loss_, loss_notpad_, correct_pc_, accuracy_notpad_ = val_outputs[0]
                    loss = loss.write(step_id, loss_)
                    loss_notpad = loss.write(step_id, loss_notpad_)
                    correct_pc = loss.write(step_id, correct_pc_)
                    accuracy_notpad = loss.write(step_id, accuracy_notpad_)

                    if cfg.eager:
                        progbar.add(1)

            loss = loss.stack()
            loss_notpad = loss_notpad.stack()
            correct_pc = correct_pc.stack()
            accuracy_notpad = accuracy_notpad.stack()

            return loss, loss_notpad, correct_pc, accuracy_notpad

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

        best_val_metrics_json = os.path.join(cfg.model_dir, "best_val_metrics.json")

        if os.path.exists(best_val_metrics_json):
            print(f'loading best_val_metrics from {best_val_metrics_json}')
            import json

            with open(best_val_metrics_json, 'r') as f:
                best_val_metrics = json.loads(f.read())
            print(f'best_val_metrics:\n{best_val_metrics}')

        while cur_step < train_steps:
            cur_epoch += 1
            tf.print(f'Training epoch {cur_epoch} with {steps_per_epoch} steps...')
            with summary_writer.as_default():
                train_multiple_steps(train_data_iters, tasks)

                """
                this check happens after the first forward pass because of deferred restoration 
                in tf.train.Checkpoint
                which only restores many of the variables after they are created in the first call 
                when the input shape becomes available
                """
                trainer.check_checkpoint_restored()

                cur_step = global_step.numpy()

                trainer.checkpoint_manager.save(cur_step)

                if cfg.train.val_epochs and cur_epoch % cfg.train.val_epochs == 0:
                    tf.print(f'validating epoch {cur_epoch} with {val_steps} steps...')

                    for t in tasks:
                        t.config.validation = True
                    val_metrics = validate(val_data_iters)
                    loss, loss_notpad, correct_pc, accuracy_notpad = val_metrics

                    val_metrics_dict = dict(
                        loss=tf.reduce_mean(loss),
                        loss_notpad=tf.reduce_mean(loss_notpad),
                        correct_pc=tf.reduce_mean(correct_pc),
                        accuracy_notpad=tf.reduce_mean(accuracy_notpad),
                    )

                    for t in tasks:
                        t.config.validation = False

                    with tf.name_scope('val'):
                        for metric_name, metric_val in val_metrics_dict.items():
                            metric_val_np = metric_val.numpy()
                            tf.summary.scalar(metric_name, metric_val_np, global_step)

                            if is_better[metric_name](metric_val, best_val_metrics[metric_name]):
                                import json
                                print(f'found better val {metric_name}: {metric_val_np}')
                                with open(best_val_metrics_json, 'w') as f:
                                    f.write(json.dumps(best_val_metrics, indent=4))
                                best_val_metrics[metric_name] = metric_val
                                val_ckpt_managers[metric_name].save(cur_step)

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
