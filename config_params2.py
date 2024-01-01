class ConfigParams2:
    cfg = ()
    dataset = ConfigParams2.Dataset()
    datasets = [ConfigParams2.Datasets0(), ]
    debug = 1
    dist = 0
    dyn_ram = 1
    eager = 1
    eval = ConfigParams2.Eval()
    gpu = ''
    master = None
    mode = 'eval'
    model = ConfigParams2.Model()
    model_dir = 'log/resnet_640'
    optimization = ConfigParams2.Optimization()
    pretrained = 'pretrained/resnet_640'
    task = ConfigParams2.Task()
    tasks = [ConfigParams2.Tasks0(), ]
    train = ConfigParams2.Train()
    training = False
    use_tpu = 0


class Dataset:
    batch_duplicates = 1
    cache_dataset = True
    category_names_path = './datasets/ipsc/well3/all_frames_roi/ext_reorg_roi_g2_54_126.json'
    coco_annotations_dir_for_metrics = './datasets/ipsc/well3/all_frames_roi'
    eval_file_pattern = './datasets/ipsc/well3/all_frames_roi/tfrecord/ext_reorg_roi_g2_54_126*'
    eval_filename_for_metrics = 'ext_reorg_roi_g2_54_126.json'
    eval_name = 'ext_reorg_roi_g2_54_126'
    eval_num_examples = 2263
    eval_split = 'validation'
    label_shift = 0
    name = 'ipsc_object_detection'
    root_dir = './datasets/ipsc/well3/all_frames_roi'
    train_file_pattern = './datasets/ipsc/well3/all_frames_roi/tfrecord/ext_reorg_roi_g2_0_53*'
    train_filename_for_metrics = 'ext_reorg_roi_g2_0_53.json'
    train_name = 'ext_reorg_roi_g2_0_53'
    train_num_examples = 1674
    train_split = 'train'


class Datasets0:
    batch_duplicates = 1
    cache_dataset = True
    category_names_path = './datasets/ipsc/well3/all_frames_roi/ext_reorg_roi_g2_54_126.json'
    coco_annotations_dir_for_metrics = './datasets/ipsc/well3/all_frames_roi'
    eval_file_pattern = './datasets/ipsc/well3/all_frames_roi/tfrecord/ext_reorg_roi_g2_54_126*'
    eval_filename_for_metrics = 'ext_reorg_roi_g2_54_126.json'
    eval_name = 'ext_reorg_roi_g2_54_126'
    eval_num_examples = 2263
    eval_split = 'validation'
    label_shift = 0
    name = 'ipsc_object_detection'
    root_dir = './datasets/ipsc/well3/all_frames_roi'
    train_file_pattern = './datasets/ipsc/well3/all_frames_roi/tfrecord/ext_reorg_roi_g2_0_53*'
    train_filename_for_metrics = 'ext_reorg_roi_g2_0_53.json'
    train_name = 'ext_reorg_roi_g2_0_53'
    train_num_examples = 1674
    train_split = 'train'


class Eval:
    batch_size = 32
    checkpoint_dir = ''
    pt = 1
    save_csv = 1
    save_vis = 0
    steps = 0
    suffix = ['batch_32', ]
    tag = 'eval'


class Model:
    coord_vocab_shift = 1000
    dec_proj_mode = 'mlp'
    decoder_output_bias = True
    dim_att = 256
    dim_att_dec = 256
    dim_mlp = 1024
    dim_mlp_dec = 1024
    drop_att = 0.0
    drop_path = 0.1
    drop_units = 0.1
    image_size = (640, 640)
    max_seq_len = 512
    name = 'encoder_ar_decoder'
    num_decoder_layers = 6
    num_encoder_layers = 6
    num_heads = 8
    num_heads_dec = 8
    patch_size = 16
    pos_encoding = 'sin_cos'
    pos_encoding_dec = 'learned'
    pretrained_ckpt = ''
    resnet_depth = 50
    resnet_sk_ratio = 0.0
    resnet_variant = 'standard'
    resnet_width_multiplier = 1
    shared_decoder_embedding = True
    text_vocab_shift = 3000
    use_cls_token = False
    vocab_size = 3000


class Optimization:
    beta1 = 0.9
    beta2 = 0.95
    end_lr_factor = 0.01
    eps = 1e-08
    global_clipnorm = -1
    learning_rate = 0.0001
    learning_rate_scaling = None
    learning_rate_schedule = 'linear'
    optimizer = 'adamw'
    warmup_epochs = 10
    warmup_steps = None
    weight_decay = 0.05


class Task:
    class_label_corruption = 'rand_n_fake_cls'
    color_jitter_strength = 0.0
    custom_sampling = False
    eos_token_weight = 0.1
    eval_transforms = [ConfigParams2.Task.EvalTransforms0(), ConfigParams2.Task.EvalTransforms1(),
                       ConfigParams2.Task.EvalTransforms2(), ConfigParams2.Task.EvalTransforms3(), ]
    image_size = (640, 640)
    jitter_scale_max = 2.0
    jitter_scale_min = 0.3
    max_instances_per_image = 100
    max_instances_per_image_test = 100
    metric = ConfigParams2.Task.Metric()
    name = 'object_detection'
    noise_bbox_weight = 1.0
    object_order = 'random'
    quantization_bins = 1000
    temperature = 1.0
    top_k = 0
    top_p = 0.4
    train_transforms = [ConfigParams2.Task.TrainTransforms0(), ConfigParams2.Task.TrainTransforms1(),
                        ConfigParams2.Task.TrainTransforms2(), ConfigParams2.Task.TrainTransforms3(),
                        ConfigParams2.Task.TrainTransforms4(), ConfigParams2.Task.TrainTransforms5(),
                        ConfigParams2.Task.TrainTransforms6(), ConfigParams2.Task.TrainTransforms7(),
                        ConfigParams2.Task.TrainTransforms8(), ]
    vocab_id = 10
    weight = 1.0


class EvalTransforms0:
    name = 'record_original_image_size'


class EvalTransforms1:
    antialias = [True, ]
    inputs = ['image', ]
    name = 'resize_image'
    target_size = (640, 640)


class EvalTransforms2:
    inputs = ['image', ]
    name = 'pad_image_to_max_size'
    object_coordinate_keys = ['bbox', ]
    target_size = (640, 640)


class EvalTransforms3:
    inputs = ['bbox', 'label', 'area', 'is_crowd', ]
    max_instances = 100
    name = 'truncate_or_pad_to_max_instances'


class Metric:
    name = 'coco_object_detection'


class TrainTransforms0:
    name = 'record_original_image_size'


class TrainTransforms1:
    inputs = ['image', ]
    max_scale = 2.0
    min_scale = 0.3
    name = 'scale_jitter'
    target_size = (640, 640)


class TrainTransforms2:
    inputs = ['image', ]
    name = 'fixed_size_crop'
    object_coordinate_keys = ['bbox', ]
    target_size = (640, 640)


class TrainTransforms3:
    bbox_keys = ['bbox', ]
    inputs = ['image', ]
    name = 'random_horizontal_flip'


class TrainTransforms4:
    filter_keys = ['is_crowd', ]
    inputs = ['bbox', 'label', 'area', 'is_crowd', ]
    name = 'filter_invalid_objects'


class TrainTransforms5:
    inputs = ['bbox', 'label', 'area', 'is_crowd', ]
    name = 'reorder_object_instances'
    order = 'random'


class TrainTransforms6:
    max_instances_per_image = 100
    name = 'inject_noise_bbox'


class TrainTransforms7:
    inputs = ['image', ]
    name = 'pad_image_to_max_size'
    object_coordinate_keys = ['bbox', ]
    target_size = (640, 640)


class TrainTransforms8:
    inputs = ['bbox', 'label', 'area', 'is_crowd', ]
    max_instances = 100
    name = 'truncate_or_pad_to_max_instances'


class Tasks0:
    class_label_corruption = 'rand_n_fake_cls'
    color_jitter_strength = 0.0
    custom_sampling = False
    eos_token_weight = 0.1
    eval_transforms = [ConfigParams2.Tasks0.EvalTransforms0(), ConfigParams2.Tasks0.EvalTransforms1(),
                       ConfigParams2.Tasks0.EvalTransforms2(), ConfigParams2.Tasks0.EvalTransforms3(), ]
    image_size = (640, 640)
    jitter_scale_max = 2.0
    jitter_scale_min = 0.3
    max_instances_per_image = 100
    max_instances_per_image_test = 100
    metric = ConfigParams2.Tasks0.Metric()
    name = 'object_detection'
    noise_bbox_weight = 1.0
    object_order = 'random'
    quantization_bins = 1000
    temperature = 1.0
    top_k = 0
    top_p = 0.4
    train_transforms = [ConfigParams2.Tasks0.TrainTransforms0(), ConfigParams2.Tasks0.TrainTransforms1(),
                        ConfigParams2.Tasks0.TrainTransforms2(), ConfigParams2.Tasks0.TrainTransforms3(),
                        ConfigParams2.Tasks0.TrainTransforms4(), ConfigParams2.Tasks0.TrainTransforms5(),
                        ConfigParams2.Tasks0.TrainTransforms6(), ConfigParams2.Tasks0.TrainTransforms7(),
                        ConfigParams2.Tasks0.TrainTransforms8(), ]
    vocab_id = 10
    weight = 1.0


class EvalTransforms0:
    name = 'record_original_image_size'


class EvalTransforms1:
    antialias = [True, ]
    inputs = ['image', ]
    name = 'resize_image'
    target_size = (640, 640)


class EvalTransforms2:
    inputs = ['image', ]
    name = 'pad_image_to_max_size'
    object_coordinate_keys = ['bbox', ]
    target_size = (640, 640)


class EvalTransforms3:
    inputs = ['bbox', 'label', 'area', 'is_crowd', ]
    max_instances = 100
    name = 'truncate_or_pad_to_max_instances'


class Metric:
    name = 'coco_object_detection'


class TrainTransforms0:
    name = 'record_original_image_size'


class TrainTransforms1:
    inputs = ['image', ]
    max_scale = 2.0
    min_scale = 0.3
    name = 'scale_jitter'
    target_size = (640, 640)


class TrainTransforms2:
    inputs = ['image', ]
    name = 'fixed_size_crop'
    object_coordinate_keys = ['bbox', ]
    target_size = (640, 640)


class TrainTransforms3:
    bbox_keys = ['bbox', ]
    inputs = ['image', ]
    name = 'random_horizontal_flip'


class TrainTransforms4:
    filter_keys = ['is_crowd', ]
    inputs = ['bbox', 'label', 'area', 'is_crowd', ]
    name = 'filter_invalid_objects'


class TrainTransforms5:
    inputs = ['bbox', 'label', 'area', 'is_crowd', ]
    name = 'reorder_object_instances'
    order = 'random'


class TrainTransforms6:
    max_instances_per_image = 100
    name = 'inject_noise_bbox'


class TrainTransforms7:
    inputs = ['image', ]
    name = 'pad_image_to_max_size'
    object_coordinate_keys = ['bbox', ]
    target_size = (640, 640)


class TrainTransforms8:
    inputs = ['bbox', 'label', 'area', 'is_crowd', ]
    max_instances = 100
    name = 'truncate_or_pad_to_max_instances'


class Train:
    batch_size = 128
    checkpoint_epochs = 1
    checkpoint_steps = None
    epochs = 80
    keep_checkpoint_max = 5
    loss_type = 'xent'
    steps = None
    suffix = []
    warmup_epochs = 2
