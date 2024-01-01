class ConfigParams:
    def __init__(self):
        self.cfg = ()
        self.dataset = ConfigParams.Dataset()
        self.datasets = [ConfigParams.Datasets0(), ]
        self.debug = 1
        self.dist = 0
        self.dyn_ram = 1
        self.eager = 1
        self.eval = ConfigParams.Eval()
        self.gpu = ''
        self.master = None
        self.mode = 'eval'
        self.model = ConfigParams.Model()
        self.model_dir = 'log/resnet_640'
        self.optimization = ConfigParams.Optimization()
        self.pretrained = 'pretrained/resnet_640'
        self.task = ConfigParams.Task()
        self.tasks = [ConfigParams.Tasks0(), ]
        self.train = ConfigParams.Train()
        self.training = False
        self.use_tpu = 0

    class Dataset:
        def __init__(self):
            self.batch_duplicates = 1
            self.cache_dataset = True
            self.category_names_path = './datasets/ipsc/well3/all_frames_roi/ext_reorg_roi_g2_54_126.json'
            self.coco_annotations_dir_for_metrics = './datasets/ipsc/well3/all_frames_roi'
            self.eval_file_pattern = './datasets/ipsc/well3/all_frames_roi/tfrecord/ext_reorg_roi_g2_54_126*'
            self.eval_filename_for_metrics = 'ext_reorg_roi_g2_54_126.json'
            self.eval_name = 'ext_reorg_roi_g2_54_126'
            self.eval_num_examples = 2263
            self.eval_split = 'validation'
            self.label_shift = 0
            self.name = 'ipsc_object_detection'
            self.root_dir = './datasets/ipsc/well3/all_frames_roi'
            self.train_file_pattern = './datasets/ipsc/well3/all_frames_roi/tfrecord/ext_reorg_roi_g2_0_53*'
            self.train_filename_for_metrics = 'ext_reorg_roi_g2_0_53.json'
            self.train_name = 'ext_reorg_roi_g2_0_53'
            self.train_num_examples = 1674
            self.train_split = 'train'

    class Datasets0:
        def __init__(self):
            self.batch_duplicates = 1
            self.cache_dataset = True
            self.category_names_path = './datasets/ipsc/well3/all_frames_roi/ext_reorg_roi_g2_54_126.json'
            self.coco_annotations_dir_for_metrics = './datasets/ipsc/well3/all_frames_roi'
            self.eval_file_pattern = './datasets/ipsc/well3/all_frames_roi/tfrecord/ext_reorg_roi_g2_54_126*'
            self.eval_filename_for_metrics = 'ext_reorg_roi_g2_54_126.json'
            self.eval_name = 'ext_reorg_roi_g2_54_126'
            self.eval_num_examples = 2263
            self.eval_split = 'validation'
            self.label_shift = 0
            self.name = 'ipsc_object_detection'
            self.root_dir = './datasets/ipsc/well3/all_frames_roi'
            self.train_file_pattern = './datasets/ipsc/well3/all_frames_roi/tfrecord/ext_reorg_roi_g2_0_53*'
            self.train_filename_for_metrics = 'ext_reorg_roi_g2_0_53.json'
            self.train_name = 'ext_reorg_roi_g2_0_53'
            self.train_num_examples = 1674
            self.train_split = 'train'

    class Eval:
        def __init__(self):
            self.batch_size = 32
            self.checkpoint_dir = ''
            self.pt = 1
            self.save_csv = 1
            self.save_vis = 0
            self.steps = 0
            self.suffix = ['batch_32', ]
            self.tag = 'eval'

    class Model:
        def __init__(self):
            self.coord_vocab_shift = 1000
            self.dec_proj_mode = 'mlp'
            self.decoder_output_bias = True
            self.dim_att = 256
            self.dim_att_dec = 256
            self.dim_mlp = 1024
            self.dim_mlp_dec = 1024
            self.drop_att = 0.0
            self.drop_path = 0.1
            self.drop_units = 0.1
            self.image_size = (640, 640)
            self.max_seq_len = 512
            self.name = 'encoder_ar_decoder'
            self.num_decoder_layers = 6
            self.num_encoder_layers = 6
            self.num_heads = 8
            self.num_heads_dec = 8
            self.patch_size = 16
            self.pos_encoding = 'sin_cos'
            self.pos_encoding_dec = 'learned'
            self.pretrained_ckpt = ''
            self.resnet_depth = 50
            self.resnet_sk_ratio = 0.0
            self.resnet_variant = 'standard'
            self.resnet_width_multiplier = 1
            self.shared_decoder_embedding = True
            self.text_vocab_shift = 3000
            self.use_cls_token = False
            self.vocab_size = 3000

    class Optimization:
        def __init__(self):
            self.beta1 = 0.9
            self.beta2 = 0.95
            self.end_lr_factor = 0.01
            self.eps = 1e-08
            self.global_clipnorm = -1
            self.learning_rate = 0.0001
            self.learning_rate_scaling = None
            self.learning_rate_schedule = 'linear'
            self.optimizer = 'adamw'
            self.warmup_epochs = 10
            self.warmup_steps = None
            self.weight_decay = 0.05

    class Task:
        def __init__(self):
            self.class_label_corruption = 'rand_n_fake_cls'
            self.color_jitter_strength = 0.0
            self.custom_sampling = False
            self.eos_token_weight = 0.1
            self.eval_transforms = [ConfigParams.Task.EvalTransforms0(), ConfigParams.Task.EvalTransforms1(),
                                    ConfigParams.Task.EvalTransforms2(), ConfigParams.Task.EvalTransforms3(), ]
            self.image_size = (640, 640)
            self.jitter_scale_max = 2.0
            self.jitter_scale_min = 0.3
            self.max_instances_per_image = 100
            self.max_instances_per_image_test = 100
            self.metric = ConfigParams.Task.Metric()
            self.name = 'object_detection'
            self.noise_bbox_weight = 1.0
            self.object_order = 'random'
            self.quantization_bins = 1000
            self.temperature = 1.0
            self.top_k = 0
            self.top_p = 0.4
            self.train_transforms = [ConfigParams.Task.TrainTransforms0(), ConfigParams.Task.TrainTransforms1(),
                                     ConfigParams.Task.TrainTransforms2(), ConfigParams.Task.TrainTransforms3(),
                                     ConfigParams.Task.TrainTransforms4(), ConfigParams.Task.TrainTransforms5(),
                                     ConfigParams.Task.TrainTransforms6(), ConfigParams.Task.TrainTransforms7(),
                                     ConfigParams.Task.TrainTransforms8(), ]
            self.vocab_id = 10
            self.weight = 1.0

        class EvalTransforms0:
            def __init__(self):
                self.name = 'record_original_image_size'

        class EvalTransforms1:
            def __init__(self):
                self.antialias = [True, ]
                self.inputs = ['image', ]
                self.name = 'resize_image'
                self.target_size = (640, 640)

        class EvalTransforms2:
            def __init__(self):
                self.inputs = ['image', ]
                self.name = 'pad_image_to_max_size'
                self.object_coordinate_keys = ['bbox', ]
                self.target_size = (640, 640)

        class EvalTransforms3:
            def __init__(self):
                self.inputs = ['bbox', 'label', 'area', 'is_crowd', ]
                self.max_instances = 100
                self.name = 'truncate_or_pad_to_max_instances'

        class Metric:
            def __init__(self):
                self.name = 'coco_object_detection'

        class TrainTransforms0:
            def __init__(self):
                self.name = 'record_original_image_size'

        class TrainTransforms1:
            def __init__(self):
                self.inputs = ['image', ]
                self.max_scale = 2.0
                self.min_scale = 0.3
                self.name = 'scale_jitter'
                self.target_size = (640, 640)

        class TrainTransforms2:
            def __init__(self):
                self.inputs = ['image', ]
                self.name = 'fixed_size_crop'
                self.object_coordinate_keys = ['bbox', ]
                self.target_size = (640, 640)

        class TrainTransforms3:
            def __init__(self):
                self.bbox_keys = ['bbox', ]
                self.inputs = ['image', ]
                self.name = 'random_horizontal_flip'

        class TrainTransforms4:
            def __init__(self):
                self.filter_keys = ['is_crowd', ]
                self.inputs = ['bbox', 'label', 'area', 'is_crowd', ]
                self.name = 'filter_invalid_objects'

        class TrainTransforms5:
            def __init__(self):
                self.inputs = ['bbox', 'label', 'area', 'is_crowd', ]
                self.name = 'reorder_object_instances'
                self.order = 'random'

        class TrainTransforms6:
            def __init__(self):
                self.max_instances_per_image = 100
                self.name = 'inject_noise_bbox'

        class TrainTransforms7:
            def __init__(self):
                self.inputs = ['image', ]
                self.name = 'pad_image_to_max_size'
                self.object_coordinate_keys = ['bbox', ]
                self.target_size = (640, 640)

        class TrainTransforms8:
            def __init__(self):
                self.inputs = ['bbox', 'label', 'area', 'is_crowd', ]
                self.max_instances = 100
                self.name = 'truncate_or_pad_to_max_instances'

    class Tasks0:
        def __init__(self):
            self.class_label_corruption = 'rand_n_fake_cls'
            self.color_jitter_strength = 0.0
            self.custom_sampling = False
            self.eos_token_weight = 0.1
            self.eval_transforms = [ConfigParams.Tasks0.EvalTransforms0(), ConfigParams.Tasks0.EvalTransforms1(),
                                    ConfigParams.Tasks0.EvalTransforms2(), ConfigParams.Tasks0.EvalTransforms3(), ]
            self.image_size = (640, 640)
            self.jitter_scale_max = 2.0
            self.jitter_scale_min = 0.3
            self.max_instances_per_image = 100
            self.max_instances_per_image_test = 100
            self.metric = ConfigParams.Tasks0.Metric()
            self.name = 'object_detection'
            self.noise_bbox_weight = 1.0
            self.object_order = 'random'
            self.quantization_bins = 1000
            self.temperature = 1.0
            self.top_k = 0
            self.top_p = 0.4
            self.train_transforms = [ConfigParams.Tasks0.TrainTransforms0(), ConfigParams.Tasks0.TrainTransforms1(),
                                     ConfigParams.Tasks0.TrainTransforms2(), ConfigParams.Tasks0.TrainTransforms3(),
                                     ConfigParams.Tasks0.TrainTransforms4(), ConfigParams.Tasks0.TrainTransforms5(),
                                     ConfigParams.Tasks0.TrainTransforms6(), ConfigParams.Tasks0.TrainTransforms7(),
                                     ConfigParams.Tasks0.TrainTransforms8(), ]
            self.vocab_id = 10
            self.weight = 1.0

        class EvalTransforms0:
            def __init__(self):
                self.name = 'record_original_image_size'

        class EvalTransforms1:
            def __init__(self):
                self.antialias = [True, ]
                self.inputs = ['image', ]
                self.name = 'resize_image'
                self.target_size = (640, 640)

        class EvalTransforms2:
            def __init__(self):
                self.inputs = ['image', ]
                self.name = 'pad_image_to_max_size'
                self.object_coordinate_keys = ['bbox', ]
                self.target_size = (640, 640)

        class EvalTransforms3:
            def __init__(self):
                self.inputs = ['bbox', 'label', 'area', 'is_crowd', ]
                self.max_instances = 100
                self.name = 'truncate_or_pad_to_max_instances'

        class Metric:
            def __init__(self):
                self.name = 'coco_object_detection'

        class TrainTransforms0:
            def __init__(self):
                self.name = 'record_original_image_size'

        class TrainTransforms1:
            def __init__(self):
                self.inputs = ['image', ]
                self.max_scale = 2.0
                self.min_scale = 0.3
                self.name = 'scale_jitter'
                self.target_size = (640, 640)

        class TrainTransforms2:
            def __init__(self):
                self.inputs = ['image', ]
                self.name = 'fixed_size_crop'
                self.object_coordinate_keys = ['bbox', ]
                self.target_size = (640, 640)

        class TrainTransforms3:
            def __init__(self):
                self.bbox_keys = ['bbox', ]
                self.inputs = ['image', ]
                self.name = 'random_horizontal_flip'

        class TrainTransforms4:
            def __init__(self):
                self.filter_keys = ['is_crowd', ]
                self.inputs = ['bbox', 'label', 'area', 'is_crowd', ]
                self.name = 'filter_invalid_objects'

        class TrainTransforms5:
            def __init__(self):
                self.inputs = ['bbox', 'label', 'area', 'is_crowd', ]
                self.name = 'reorder_object_instances'
                self.order = 'random'

        class TrainTransforms6:
            def __init__(self):
                self.max_instances_per_image = 100
                self.name = 'inject_noise_bbox'

        class TrainTransforms7:
            def __init__(self):
                self.inputs = ['image', ]
                self.name = 'pad_image_to_max_size'
                self.object_coordinate_keys = ['bbox', ]
                self.target_size = (640, 640)

        class TrainTransforms8:
            def __init__(self):
                self.inputs = ['bbox', 'label', 'area', 'is_crowd', ]
                self.max_instances = 100
                self.name = 'truncate_or_pad_to_max_instances'

    class Train:
        def __init__(self):
            self.batch_size = 128
            self.checkpoint_epochs = 1
            self.checkpoint_steps = None
            self.epochs = 80
            self.keep_checkpoint_max = 5
            self.loss_type = 'xent'
            self.steps = None
            self.suffix = []
            self.warmup_epochs = 2
