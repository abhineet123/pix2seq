from absl import logging
import os
import collections
import time
import json

import utils

from eval_utils import profile


def run(cfg, dataset, task, eval_steps, ckpt, strategy, model_lib, tf):
    """Perform evaluation."""
    eval_tag = cfg.eval.tag
    summary_writer = None
    # summary_writer = tf.summary.create_file_writer(FLAGS.model_dir)

    is_video = 'video' in cfg.task.name

    with strategy.scope():
        # Restore model checkpoint.
        model = model_lib.ModelRegistry.lookup(cfg.model.name)(cfg)

        logging.info('Restoring from %s', ckpt)
        checkpoint = tf.train.Checkpoint(
            model=model, global_step=tf.Variable(0, dtype=tf.int64))
        status = checkpoint.restore(ckpt).expect_partial()  # Not restore optimizer.
        verify_restored = status.assert_consumed
        verify_existing = status.assert_existing_objects_matched
        global_step = checkpoint.global_step
        logging.info('Performing eval at step %d', global_step.numpy())

    ckpt_name = os.path.splitext(os.path.basename(ckpt))[0]
    json_name = cfg.dataset.eval_filename_for_metrics

    assert json_name, "eval_filename_for_metrics must be provided for evaluation"
    json_name = os.path.basename(json_name).split(os.extsep)[0]

    out_dir = os.path.join(cfg.model_dir, f'{ckpt_name}-{json_name}')

    out_csv_dir = out_vis_dir = None
    save_suffix = ''
    if cfg.eval.save_suffix:
        save_suffix = '-'.join(cfg.eval.save_suffix)

    if cfg.eval.save_csv:
        csv_dir_name = f'csv'
        if save_suffix:
            csv_dir_name = f'{csv_dir_name:s}-{save_suffix:s}'
        out_csv_dir = os.path.join(out_dir, csv_dir_name)
        print(f'\nwriting csv files to: {out_csv_dir}\n')
        os.makedirs(out_csv_dir, exist_ok=True)

    if cfg.eval.save_vis:
        vis_dir_name = f'vis'
        if save_suffix:
            vis_dir_name = f'{vis_dir_name:s}-{save_suffix:s}'
        out_vis_dir = os.path.join(out_dir, vis_dir_name)
        os.makedirs(out_vis_dir, exist_ok=True)

        print(f'\nwriting vis images to: {out_vis_dir}\n')

    seq_to_csv_rows = collections.defaultdict(list)
    seq_to_vid_cap = collections.defaultdict(lambda: None)

    def single_step(examples):
        preprocessed_outputs = task.preprocess_batched(examples, training=False)
        infer_outputs = task.infer(model, preprocessed_outputs)
        postprocessed_outputs = task.postprocess_tpu(*infer_outputs)
        return postprocessed_outputs

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

        # print(f'min_score_thresh: {cfg.eval.min_score_thresh}')

        while True:
            if eval_steps and cur_step >= eval_steps:
                break
            # try:
            # with summary_writer.as_default():

            if cfg.eager:
                enable_profiling = cfg.eval.profile
                _times = collections.OrderedDict()
                _rel_times = collections.OrderedDict()
                with profile('iterator', _times, _rel_times, enable_profiling, show=True):
                    examples = next(iterator)
                with profile('preprocess_batched', _times, _rel_times, enable_profiling, show=True):
                    preprocessed_outputs = task.preprocess_batched(examples, training=False)
                with profile('infer', _times, _rel_times, enable_profiling, show=True):
                    infer_outputs = task.infer(model, preprocessed_outputs)
                with profile('postprocess_tpu', _times, _rel_times, enable_profiling, show=True):
                    per_step_outputs = task.postprocess_tpu(*infer_outputs)

                if enable_profiling:
                    print(f'times: {_times}')
                    print(f'rel_times: {_rel_times}')
            else:
                per_step_outputs = run_single_step(iterator)

            if cur_step == 0:
                utils.check_checkpoint_restored(
                    strict_verifiers=(),
                    loose_verifiers=[verify_restored, verify_existing],
                )

            task.postprocess_cpu(
                outputs=per_step_outputs,
                train_step=global_step.numpy(),
                out_vis_dir=out_vis_dir,
                vid_cap=seq_to_vid_cap,
                csv_data=seq_to_csv_rows,
                eval_step=cur_step,
                summary_tag=eval_tag,
                min_score_thresh=cfg.eval.min_score_thresh,
                ret_results=False)

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
        import pandas as pd
        csv_columns = [
            "ImageID", "LabelName",
            "XMin", "XMax", "YMin", "YMax", "Confidence",
        ]
        if is_video:
            csv_columns.insert(1, 'VideoID')
        # if params.enable_mask:
        #     csv_columns += ['mask_w', 'mask_h', 'mask_counts']
        for seq_name, vid_cap_seq in seq_to_vid_cap.items():
            if vid_cap_seq is not None:
                vid_cap_seq.release()

        for csv_seq_name, csv_rows in seq_to_csv_rows.items():
            if not csv_rows:
                print(f'{csv_seq_name}: no csv data found')
                # continue
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
