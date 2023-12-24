import os
import collections
import ml_collections
import json
import copy

TRAIN = 'train'
EVAL = 'eval'
MAX_JSON_VARS = 10
NAMED_JSON_VARS = ['mode']


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


def load_from_model(cfg, model_dir, cmd_cfg, pt=False):
    pt_cfg_filepath = os.path.join(model_dir, 'config.json')

    assert os.path.isfile(pt_cfg_filepath), f"non-existent model cfg json: {pt_cfg_filepath}"

    print(f'loading model cfg from {pt_cfg_filepath}')
    with open(pt_cfg_filepath, 'r') as f:
        cfg_model = json.loads(f.read())

    """
    hack to deal with type mismatches between variables in the config py files and those in the 
    config json files accompanying the pre-trained models
    ConfigDict does not allow type override so type changes must be done in ordinary dict
    """
    image_size = cfg_model['model']['image_size']
    if isinstance(image_size, int):
        cfg_model['model']['image_size'] = (image_size, image_size)

    image_size = cfg_model['task']['image_size']
    if isinstance(image_size, int):
        cfg_model['task']['image_size'] = (image_size, image_size)

    cfg_model = ml_collections.ConfigDict(cfg_model)

    if pt:
        cfg.model.update(cfg_model.model)
        cfg.task.update(cfg_model.task)
        cfg.train.update(cfg_model.train)
        cfg.optimization.update(cfg_model.optimization)
    else:
        cfg.update(cfg_model)
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


def load_from_json5(json_list, json_root):
    import collections

    """ml_collections.ConfigDict supports recursive updating for direct but not for list so this 
    function is needed for distributed list specification to work"""

    def update(orig_dict, new_dict):
        for key, val in new_dict.items():
            if isinstance(val, collections.abc.Mapping):
                tmp = update(orig_dict.get(key, {}), val)
                orig_dict[key] = tmp
            elif isinstance(val, list):
                orig_dict[key] = (orig_dict.get(key, []) + val)
            else:
                orig_dict[key] = new_dict[key]
        return orig_dict

    all_json_dict = {}
    # all_json_dict2 = {}
    # all_json_cfg = ml_collections.ConfigDict()
    for json_data in json_list:
        json_vars = json_data.split('-')
        json_name, json_vars = json_vars[0], json_vars[1:]
        json_path = json_name + '.json5'
        if json_root:
            json_path = os.path.join(json_root, json_path)

        assert os.path.isfile(json_path), f"non-existent cfg json: {json_path}"
        # print(f'loading json cfg from {json_path}')
        with open(json_path, 'r') as f:
            json_str = f.read()

        """named vars"""
        for named_json_var in NAMED_JSON_VARS:
            try:
                json_var_val = all_json_dict[named_json_var]
            except KeyError:
                pass
            else:
                json_str = json_str.replace(f'${named_json_var}$', json_var_val)

        # if not json_vars:
        #     continue

        """combined vars for cases where bvar may have a separator as part of it"""
        cmb_json_vars = '-'.join(json_vars)
        json_str = json_str.replace('$*$', cmb_json_vars)

        for var_id, json_var in enumerate(json_vars):
            """optional vars"""
            json_str = json_str.replace(f'$${var_id}$$', json_var)
            """compulsory vars"""
            json_str = json_str.replace(f'${var_id}$', json_var)

        """
        remove lines with under specified optional vars
        json5 is needed to deal with trailing commas
        """
        json_lines = json_str.splitlines()
        valid_line_ids = []
        for line_id, json_line in enumerate(json_lines):
            if any(f'$${var_id}$$' in json_line for var_id in range(MAX_JSON_VARS)):
                continue
            if any(f'${var_id}$' in json_line for var_id in range(MAX_JSON_VARS)):
                raise AssertionError(f'{json_name}: unsubstituted position variable found in {json_line}')
            if any(f'${var}$' in json_line for var in NAMED_JSON_VARS):
                raise AssertionError(f'{json_name}: unsubstituted named variable found in {json_line}')

            valid_line_ids.append(line_id)
        json_str = '\n'.join(json_lines[i] for i in valid_line_ids)

        import json5

        json_dict = json5.loads(json_str)
        all_json_dict = update(all_json_dict, json_dict)
        # all_json_dict2.update(json_dict)
        # all_json_cfg.update(ml_collections.ConfigDict(json_dict))
        # print()

    all_cfg = ml_collections.ConfigDict(all_json_dict)

    return all_cfg


def load(FLAGS):
    cfg = FLAGS.cfg

    cmd_cfg = load_from_json5(FLAGS.j5, FLAGS.j5_root)

    cfg.update(cmd_cfg)

    cfg.training = cfg.mode == TRAIN

    if cfg.model_dir:
        load_from_model(cfg, cfg.model_dir, cmd_cfg, pt=False)
    else:
        if cfg.pretrained:
            load_from_model(cfg, cfg.pretrained, cmd_cfg, pt=True)

        if not cfg.training and cfg.eval.pt:
            assert cfg.pretrained, "cfg.pretrained must be provided for pretrained model eval"

            cfg.model_dir = cfg.pretrained.replace('pretrained', 'log')
        else:
            model_dir_name = f'{cfg.dataset.train_name}_batch_{cfg.train.batch_size}'
            if cfg.pretrained:
                pretrained_name = os.path.basename(cfg.pretrained)
                model_dir_name = f'{pretrained_name}_{model_dir_name}'

            if cfg.train.suffix:
                suffix = '-'.join(cfg.train.suffix)
                model_dir_name = f'{model_dir_name}-{suffix}'

            if cfg.dist == 2 and cfg.dist2.task.index > 0:
                model_dir_name = f'{model_dir_name}-worker-{cfg.dist2.task.index}'

            cfg.model_dir = os.path.join('log', model_dir_name)

    if cfg.training:
        print(f'saving trained model to: {cfg.model_dir}')
    else:
        print(f'loading trained model from: {cfg.model_dir}')
        if not cfg.eval.pt:
            load_from_model(cfg, cfg.model_dir, cmd_cfg)

    # config_cmd_args = [k for k in dir(FLAGS) if k.startswith('cfg.')]
    # config_cmd_dict = {
    #     k: getattr(FLAGS, k) for k in dir(FLAGS) if k.startswith('cfg.')
    # }
    if cfg.dataset.name.startswith('ipsc'):
        from configs.dataset_configs import ipsc_post_process
        ipsc_post_process(cfg.dataset)

    import utils
    utils.log_cfg(cfg)

    return cfg
