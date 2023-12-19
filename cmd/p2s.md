<!-- MarkdownTOC -->

- [tfrecord](#tfrecor_d_)
    - [ipsc       @ tfrecord](#ipsc___tfrecord_)
        - [ext_reorg_roi_g2_16_53       @ ipsc/tfrecord](#ext_reorg_roi_g2_16_53___ipsc_tfrecor_d_)
        - [ext_reorg_roi_g2_0_1       @ ipsc/tfrecord](#ext_reorg_roi_g2_0_1___ipsc_tfrecor_d_)
- [train](#train_)
    - [resnet-640       @ train](#resnet_640___trai_n_)
        - [g2_16_53       @ resnet-640/train](#g2_16_53___resnet_640_train_)
- [eval](#eva_l_)
    - [resnet-640       @ eval](#resnet_640___eval_)
        - [mninstmot       @ resnet-640/eval](#mninstmot___resnet_640_eva_l_)
        - [g2_0_1       @ resnet-640/eval](#g2_0_1___resnet_640_eva_l_)
        - [g2_16_53       @ resnet-640/eval](#g2_16_53___resnet_640_eva_l_)
            - [dist       @ g2_16_53/resnet-640/eval](#dist___g2_16_53_resnet_640_eval_)
    - [resnet-1333       @ eval](#resnet_1333___eval_)
        - [g2_0_1       @ resnet-1333/eval](#g2_0_1___resnet_1333_eval_)
        - [g2_16_53       @ resnet-1333/eval](#g2_16_53___resnet_1333_eval_)
    - [resnet_c4-640       @ eval](#resnet_c4_640___eval_)
        - [g2_16_53       @ resnet_c4-640/eval](#g2_16_53___resnet_c4_640_eval_)
    - [resnet_c4_1333       @ eval](#resnet_c4_1333___eval_)
        - [g2_0_1       @ resnet_c4_1333/eval](#g2_0_1___resnet_c4_1333_eva_l_)
        - [g2_16_53       @ resnet_c4_1333/eval](#g2_16_53___resnet_c4_1333_eva_l_)

<!-- /MarkdownTOC -->

<a id="tfrecor_d_"></a>
# tfrecord
<a id="ipsc___tfrecord_"></a>
## ipsc       @ tfrecord-->p2s
python3 data/scripts/create_ipsc_tfrecord.py
<a id="ext_reorg_roi_g2_16_53___ipsc_tfrecor_d_"></a>
### ext_reorg_roi_g2_16_53       @ ipsc/tfrecord-->p2s
python3 data/scripts/create_ipsc_tfrecord.py --ann_file=ext_reorg_roi_g2_16_53.json --n_proc=0
<a id="ext_reorg_roi_g2_0_1___ipsc_tfrecor_d_"></a>
### ext_reorg_roi_g2_0_1       @ ipsc/tfrecord-->p2s
python data/scripts/create_ipsc_tfrecord.py --ann_file=ext_reorg_roi_g2_0_1.json --n_proc=0
<a id="hello___ext_reorg_roi_g2_0_1_ipsc_nazi_o_"></a>

<a id="train_"></a>
# train
<a id="resnet_640___trai_n_"></a>
## resnet-640       @ train-->p2s
<a id="g2_16_53___resnet_640_train_"></a>
### g2_16_53       @ resnet-640/train-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --json=train,resnet-640,ipsc-g2_16_53-train,ipsc-g2_0_15-val,batch-2,dist-0,dbg-1,dyn-1,suffix-train-dist2

<a id="eva_l_"></a>
# eval
<a id="resnet_640___eval_"></a>
## resnet-640       @ eval-->p2s
<a id="mninstmot___resnet_640_eva_l_"></a>
### mninstmot       @ resnet-640/eval-->p2s
python3 run.py --mode=eval --model_dir=pretrained/resnet_640 --cfg=configs/config_det_mninstmot.py

<a id="g2_0_1___resnet_640_eva_l_"></a>
### g2_0_1       @ resnet-640/eval-->p2s
__batch-48__
python3 run.py --cfg=configs/config_det_ipsc.py  --json=eval,resnet-640,ipsc-g2_0_1,batch-48,save-vis-1,save-csv-1,dist-0
__batch-2__
python3 run.py --cfg=configs/config_det_ipsc.py  --json=eval,resnet-640,ipsc-g2_0_1,batch-2,save-vis-1,save-csv-1,dist-0
<a id="g2_16_53___resnet_640_eva_l_"></a>
### g2_16_53       @ resnet-640/eval-->p2s
__-batch-64-__
python3 run.py --cfg=configs/config_det_ipsc.py  --json=eval,resnet-640,ipsc-g2_16_53,batch-64,save-vis-1,save-csv-1,dist-0
__batch-32__
python3 run.py --cfg=configs/config_det_ipsc.py  --json=eval,resnet-640,ipsc-g2_16_53,batch-32,save-vis-1,save-csv-1,dist-0

<a id="dist___g2_16_53_resnet_640_eval_"></a>
#### dist       @ g2_16_53/resnet-640/eval-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --json=eval,resnet-640,ipsc-g2_0_1,batch-2,save-vis-1,save-csv-1,dist-1,eager-0

<a id="resnet_1333___eval_"></a>
## resnet-1333       @ eval-->p2s
<a id="g2_0_1___resnet_1333_eval_"></a>
### g2_0_1       @ resnet-1333/eval-->p2s
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --json=eval,resnet-1333,ipsc-g2_0_1,batch-48,save-vis-1,save-csv-1,dist-0
<a id="g2_16_53___resnet_1333_eval_"></a>
### g2_16_53       @ resnet-1333/eval-->p2s
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --json=eval,resnet-1333,ipsc-g2_16_53,batch-24,save-vis-1,save-csv-1,dist-0

<a id="resnet_c4_640___eval_"></a>
## resnet_c4-640       @ eval-->p2s
<a id="g2_16_53___resnet_c4_640_eval_"></a>
### g2_16_53       @ resnet_c4-640/eval-->p2s
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --json=eval,resnet_c4-640,ipsc-g2_16_53,batch-16,save-vis-1,save-csv-1,dist-0

<a id="resnet_c4_1333___eval_"></a>
## resnet_c4_1333       @ eval-->p2s
<a id="g2_0_1___resnet_c4_1333_eva_l_"></a>
### g2_0_1       @ resnet_c4_1333/eval-->p2s
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --json=eval,resnet_c4-1333,ipsc-g2_0_1,batch-1,save-vis-1,save-csv-1,dist-0
<a id="g2_16_53___resnet_c4_1333_eva_l_"></a>
### g2_16_53       @ resnet_c4_1333/eval-->p2s
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --json=eval,resnet_c4-1333,ipsc-g2_16_53,batch-1,save-vis-1,save-csv-1,dist-0


