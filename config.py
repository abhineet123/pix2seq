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


def from_dict(in_dict):
    for key, val in in_dict.items():
        if isinstance(val, list):
            in_dict[key] = from_list(val)
        elif isinstance(val, dict):
            in_dict[key] = from_dict(val)
    in_dict = ml_collections.ConfigDict(in_dict)
    return in_dict


def from_list(in_list):
    for idx, val in enumerate(in_list):
        if isinstance(val, list):
            in_list[idx] = from_list(val)
        elif isinstance(val, dict):
            in_list[idx] = from_dict(val)
    return in_list


def load_from_model(cfg, model_dir, cmd_cfg, pt=False):
    pt_cfg_filepath = os.path.join(model_dir, 'config.json')

    assert os.path.isfile(pt_cfg_filepath), f"non-existent model cfg json: {pt_cfg_filepath}"

    print(f'loading model cfg from {pt_cfg_filepath}')
    with open(pt_cfg_filepath, 'r') as f:
        cfg_dict = json.loads(f.read())

    if not pt:
        cfg.update(from_dict(cfg_dict))
        cfg.update(cmd_cfg)
        return

    """
    hack to deal with type mismatches between variables in the config py files and those in the 
    config json files accompanying the pretrained models
    ConfigDict does not allow type override so type changes must be done in ordinary dict
    """
    image_size = cfg_dict['model']['image_size']
    if isinstance(image_size, int):
        cfg_dict['model']['image_size'] = (image_size, image_size)

    image_size = cfg_dict['task']['image_size']
    if isinstance(image_size, int):
        cfg_dict['task']['image_size'] = (image_size, image_size)

    """buggy ml_collections.ConfigDict does not convert list of dicts into list of ConfigDicts or vice versa"""

    model_cfg = from_dict(cfg_dict)
    model_cfg = ml_collections.ConfigDict(model_cfg)

    cfg.model.update(model_cfg.model)
    # cfg.task.update(model_cfg.task)
    # cfg.train.update(model_cfg.train)
    # cfg.optimization.update(model_cfg.optimization)

    try:
        cfg.model.update(cmd_cfg.model)
    except AttributeError:
        """no model cfg params supplied at command line"""
        pass

    # task_config_hack_old(cfg)


def expand_list(val):
    val = val.strip()
    if val.startswith('range('):
        """standard (exclusive) range"""
        end_id = val.find(')')
        assert end_id > 6, f"invalid tuple str: {val}"
        substr = val[:end_id + 1]
        val_list = val[6:].replace(')', '').split(',')
        val_list = [int(x) for x in val_list]
        val_list = list(range(*val_list))
        out_str = val.replace(substr, f'{val_list}')
        return val_list
    elif val.startswith('irange('):
        end_id = val.find(')')
        assert end_id > 7, f"invalid tuple str: {val}"
        substr = val[:end_id + 1]
        """inclusive range"""
        val_list = val.replace('irange(', '').replace(')', '').split(',')
        val_list = [int(x) for x in val_list if x]
        if len(val_list) == 1:
            val_list[0] += 1
        elif len(val_list) >= 2:
            val_list[1] += 1
        val_list = list(range(*val_list))
        out_str = val.replace(substr, f'{val_list}')
        return out_str
    return val


def load_from_json5(json_list, json_root):
    """ml_collections.ConfigDict supports recursive updating for dict but not for list so this
    function is needed for distributed list specification to work"""

    def update(orig_dict, new_dict):
        for key, val in new_dict.items():
            if isinstance(val, dict):
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
            rep_str = f'%{named_json_var}%'
            if rep_str not in json_str:
                continue
            json_var_val = all_json_dict[named_json_var]
            json_str = json_str.replace(rep_str, json_var_val)

        # if not json_vars:
        #     continue

        """combined vars for cases where var may have a separator as part of it"""
        cmb_json_vars = '-'.join(json_vars)
        json_str = json_str.replace('%*%', cmb_json_vars)

        for var_id, json_var in enumerate(json_vars):
            """optional vars"""
            json_str = json_str.replace(f'%%{var_id}%%', json_var)
            """compulsory vars"""
            json_str = json_str.replace(f'%{var_id}%', json_var)

        """
        remove lines with unspecified optional vars
        """
        json_lines = json_str.splitlines()
        valid_line_ids = []
        for line_id, json_line in enumerate(json_lines):
            is_valid = True
            if any(f'%%{var_id}%%' in json_line for var_id in range(MAX_JSON_VARS)):
                """remove %% for subsequent comparison to work"""
                json_line = json_line.replace('%%', '$$')
                is_valid = False
            if any(f'%{var_id}%' in json_line for var_id in range(MAX_JSON_VARS)):
                raise AssertionError(f'{json_name}: unsubstituted position variable found in {json_line}')
            if any(f'%{var}%' in json_line for var in NAMED_JSON_VARS):
                raise AssertionError(f'{json_name}: unsubstituted named variable found in {json_line}')
            if is_valid:
                valid_line_ids.append(line_id)
            else:
                continue
            try:
                arg_name, arg_val = json_line.split(':')
            except ValueError:
                pass
            else:
                arg_tuple = expand_list(arg_val)
                json_line_ = f'{arg_name}: {arg_tuple}'
                json_lines[line_id] = json_line_

        json_str = '\n'.join(json_lines[i] for i in valid_line_ids)

        """
        json5 is needed to deal with trailing commas
        """
        import json5

        json_dict = json5.loads(json_str)
        all_json_dict = update(all_json_dict, json_dict)

        # all_json_dict2.update(json_dict)
        # all_json_cfg.update(ml_collections.ConfigDict(json_dict))
        # print()

    """annoyingly buggy ml_collections.ConfigDict doesn't handle recursive dict to ConfigDict conversion"""
    all_json_dict = from_dict(all_json_dict)
    all_cfg = ml_collections.ConfigDict(all_json_dict)

    return all_cfg


def load(FLAGS):
    cfg = FLAGS.cfg

    cmd_cfg = load_from_json5(FLAGS.j5, FLAGS.j5_root)

    cfg.update(cmd_cfg)

    is_video = 'video' in cfg.task.name

    model_root_dir = 'log'
    if is_video:
        model_root_dir = os.path.join(model_root_dir, 'video')

    if cfg.model_dir:
        assert not cfg.eval.pt, "pre-trained evaluation must be disabled if custom model directory is specified"
        """load config from manually specified model_dirs"""
        cmd_cfg.model_dir = cfg.model_dir = os.path.join(model_root_dir, cfg.model_dir)
        load_from_model(cfg, cfg.model_dir, cmd_cfg, pt=False)
    else:
        if cfg.pretrained:
            load_from_model(cfg, cfg.pretrained, cmd_cfg, pt=True)

    if cfg.dataset.name.startswith('ipsc'):
        from configs.dataset_configs import ipsc_post_process
        ipsc_post_process(cfg.dataset)

    cmd_cfg.training = cfg.training = cfg.mode == TRAIN

    if not cfg.model_dir:
        """construct model_dir name from params"""

        if not cfg.training and cfg.eval.pt:
            """evaluate on pretrained model but save results in the log folder"""
            assert cfg.pretrained, "cfg.pretrained must be provided for pretrained model eval"

            cfg.model_dir = cfg.pretrained.replace('pretrained', model_root_dir)
        else:
            model_dir_name = f'{cfg.dataset.train_name}'
            if cfg.pretrained:
                pretrained_name = os.path.basename(cfg.pretrained)
                if cfg.resnet_replace:
                    pretrained_name=pretrained_name.replace('resnet', cfg.resnet_replace)
                model_dir_name = f'{pretrained_name}_{model_dir_name}'

            if cfg.train.save_suffix:
                print(f'train.save_suffix: {cfg.train.save_suffix}')
                save_suffix = '-'.join(cfg.train.save_suffix)
                model_dir_name = f'{model_dir_name}-{save_suffix}'

            # if cfg.dist == 2 and cfg.tf_config.task.index > 0:
            #     model_dir_name = f'{model_dir_name}-worker-{cfg.tf_config.task.index}'

            cfg.model_dir = os.path.join(model_root_dir, model_dir_name)
        if not cfg.training and not cfg.eval.pt:
            """load config from model_dir constructed from params"""
            load_from_model(cfg, cfg.model_dir, cmd_cfg)

    if cfg.training:
        if cfg.train.pt:
            print(f'loading pretrained model from: {cfg.pretrained}')
        print(f'saving trained model to: {cfg.model_dir}')
    else:
        if cfg.eval.pt:
            print(f'evaluating on pretrained model from: {cfg.pretrained}')
        else:
            print(f'evaluating on trained model from: {cfg.model_dir}')

    # config_cmd_args = [k for k in dir(FLAGS) if k.startswith('cfg.')]
    # config_cmd_dict = {
    #     k: getattr(FLAGS, k) for k in dir(FLAGS) if k.startswith('cfg.')
    # }

    if cfg.task.name == 'object_detection':
        from configs.config_det_ipsc import update_task_config
        update_task_config(cfg)
    elif cfg.task.name == 'video_detection':
        from configs.config_video_det import update_task_config
        update_task_config(cfg)
    else:
        raise AssertionError(f'unsupported task: {cfg.task.name}')

    import utils
    utils.log_cfg(cfg)

    """buggy ml_collections.ConfigDict does not convert list of dicts into list of ConfigDicts or vice versa"""

    def to_dict(in_dict):
        for key, val in in_dict.items():
            if isinstance(val, list):
                in_dict[key] = to_list(val)
            elif isinstance(val, ml_collections.ConfigDict):
                in_dict[key] = to_dict(val)
        in_dict = in_dict.to_dict()
        return in_dict

    def to_list(in_list):
        for idx, val in enumerate(in_list):
            if isinstance(val, list):
                in_list[idx] = to_list(val)
            elif isinstance(val, ml_collections.ConfigDict):
                in_list[idx] = to_dict(val)
        return in_list

    # import paramparse
    # cfg_dict = to_dict(cfg)
    # from config_params import ConfigParams
    # cfg_class = ConfigParams()
    # paramparse.from_dict(cfg_dict, class_name='ConfigParams2', add_help=False, add_init=False)
    # exit()

    return cfg
