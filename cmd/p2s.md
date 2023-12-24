<!-- MarkdownTOC -->

- [tfrecord](#tfrecor_d_)
    - [ipsc       @ tfrecord](#ipsc___tfrecord_)
        - [ext_reorg_roi_g2_16_53       @ ipsc/tfrecord](#ext_reorg_roi_g2_16_53___ipsc_tfrecor_d_)
        - [ext_reorg_roi_g2_0_1       @ ipsc/tfrecord](#ext_reorg_roi_g2_0_1___ipsc_tfrecor_d_)
        - [ext_reorg_roi_g2_0_15       @ ipsc/tfrecord](#ext_reorg_roi_g2_0_15___ipsc_tfrecor_d_)
- [resnet-640](#resnet_64_0_)
    - [pt       @ resnet-640](#pt___resnet_640_)
        - [mninstmot       @ pt/resnet-640](#mninstmot___pt_resnet_64_0_)
        - [g2_0_1       @ pt/resnet-640](#g2_0_1___pt_resnet_64_0_)
        - [g2_16_53       @ pt/resnet-640](#g2_16_53___pt_resnet_64_0_)
            - [dist       @ g2_16_53/pt/resnet-640](#dist___g2_16_53_pt_resnet_640_)
    - [g2_16_53       @ resnet-640](#g2_16_53___resnet_640_)
        - [dist-1       @ g2_16_53/resnet-640](#dist_1___g2_16_53_resnet_64_0_)
        - [dist-2       @ g2_16_53/resnet-640](#dist_2___g2_16_53_resnet_64_0_)
            - [local       @ dist-2/g2_16_53/resnet-640](#local___dist_2_g2_16_53_resnet_640_)
            - [xe       @ dist-2/g2_16_53/resnet-640](#xe___dist_2_g2_16_53_resnet_640_)
                - [eval       @ xe/dist-2/g2_16_53/resnet-640](#eval___xe_dist_2_g2_16_53_resnet_64_0_)
            - [gx       @ dist-2/g2_16_53/resnet-640](#gx___dist_2_g2_16_53_resnet_640_)
            - [gxe2       @ dist-2/g2_16_53/resnet-640](#gxe2___dist_2_g2_16_53_resnet_640_)
- [resnet-1333](#resnet_1333_)
    - [pt       @ resnet-1333](#pt___resnet_133_3_)
        - [g2_0_1       @ pt/resnet-1333](#g2_0_1___pt_resnet_1333_)
        - [g2_16_53       @ pt/resnet-1333](#g2_16_53___pt_resnet_1333_)
- [resnet_c4-640](#resnet_c4_640_)
    - [pt       @ resnet_c4-640](#pt___resnet_c4_64_0_)
        - [g2_16_53       @ pt/resnet_c4-640](#g2_16_53___pt_resnet_c4_640_)
- [resnet_c4_1333](#resnet_c4_133_3_)
    - [pt       @ resnet_c4_1333](#pt___resnet_c4_1333_)
        - [g2_0_1       @ pt/resnet_c4_1333](#g2_0_1___pt_resnet_c4_133_3_)
        - [g2_16_53       @ pt/resnet_c4_1333](#g2_16_53___pt_resnet_c4_133_3_)

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
<a id="ext_reorg_roi_g2_0_15___ipsc_tfrecor_d_"></a>
### ext_reorg_roi_g2_0_15       @ ipsc/tfrecord-->p2s
python data/scripts/create_ipsc_tfrecord.py --ann_file=ext_reorg_roi_g2_0_15.json --n_proc=0

<a id="resnet_64_0_"></a>
# resnet-640 
<a id="pt___resnet_640_"></a>
## pt       @ resnet-640-->p2s
<a id="mninstmot___pt_resnet_64_0_"></a>
### mninstmot       @ pt/resnet-640-->p2s
python3 run.py --mode=eval --model_dir=pretrained/resnet_640 --cfg=configs/config_det_mninstmot.py
<a id="g2_0_1___pt_resnet_64_0_"></a>
### g2_0_1       @ pt/resnet-640-->p2s
__batch-48__
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,resnet-640,ipsc-g2_0_1,batch-48,save-vis-1,save-csv-1,dist-0
__batch-2__
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,resnet-640,ipsc-g2_0_1,batch-2,save-vis-1,save-csv-1,dist-0
<a id="g2_16_53___pt_resnet_64_0_"></a>
### g2_16_53       @ pt/resnet-640-->p2s
__-batch-64-__
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,resnet-640,ipsc-g2_16_53,batch-64,save-vis-1,save-csv-1,dist-0
__batch-32__
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,resnet-640,ipsc-g2_16_53,batch-32,save-vis-1,save-csv-1,dist-0
<a id="dist___g2_16_53_pt_resnet_640_"></a>
#### dist       @ g2_16_53/pt/resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,resnet-640,ipsc-g2_0_1,batch-2,save-vis-1,save-csv-1,dist-1,eager-0

<a id="g2_16_53___resnet_640_"></a>
## g2_16_53       @ resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train-dist0,resnet-640,ipsc-train-g2_16_53,ipsc-val-g2_0_15,batch-6,dist-0,dbg-1,dyn-1
<a id="dist_1___g2_16_53_resnet_64_0_"></a>
### dist-1       @ g2_16_53/resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train-dist1,resnet-640,ipsc-train-g2_16_53,ipsc-val-g2_0_15,batch-12,dist-1,dbg-0,dyn-0
<a id="dist_2___g2_16_53_resnet_64_0_"></a>
### dist-2       @ g2_16_53/resnet-640-->p2s
<a id="local___dist_2_g2_16_53_resnet_640_"></a>
#### local       @ dist-2/g2_16_53/resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train-dist2,resnet-640,ipsc-train-g2_16_53,ipsc-val-g2_0_15,batch-4,dist-2,dbg-0,dyn-1,local-0,gpu-0

python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train-dist2,resnet-640,ipsc-train-g2_16_53,ipsc-val-g2_0_15,batch-4,dist-2,dbg-0,dyn-1,local-1,gpu-1

<a id="xe___dist_2_g2_16_53_resnet_640_"></a>
#### xe       @ dist-2/g2_16_53/resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train-dist2,resnet-640,ipsc-train-g2_16_53,ipsc-val-g2_0_15,batch-32,dist-2,dbg-0,dyn-1,xe-0

python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train-dist2,resnet-640,ipsc-train-g2_16_53,ipsc-val-g2_0_15,batch-32,dist-2,dbg-0,dyn-1,xe-1
<a id="eval___xe_dist_2_g2_16_53_resnet_64_0_"></a>
##### eval       @ xe/dist-2/g2_16_53/resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train-dist2,resnet-640,ipsc-train-g2_16_53,ipsc-val-g2_0_15,batch-32,dist-0,dbg-1,dyn-1,eval,batch_e-16,save-vis-1,save-csv-1

<a id="gx___dist_2_g2_16_53_resnet_640_"></a>
#### gx       @ dist-2/g2_16_53/resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train-dist2,resnet-640,ipsc-train-g2_16_53,ipsc-val-g2_0_15,batch-36,dist-2,dbg-0,dyn-1,gx-0

python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train-dist2,resnet-640,ipsc-train-g2_16_53,ipsc-val-g2_0_15,batch-36,dist-2,dbg-0,dyn-1,gx-1

<a id="gxe2___dist_2_g2_16_53_resnet_640_"></a>
#### gxe2       @ dist-2/g2_16_53/resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train-dist2,resnet-640,ipsc-train-g2_16_53,ipsc-val-g2_0_15,batch-48,dist-2,dbg-0,dyn-0,gxe2-0

python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train-dist2,resnet-640,ipsc-train-g2_16_53,ipsc-val-g2_0_15,batch-48,dist-2,dbg-0,dyn-0,gxe2-1

python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train-dist2,resnet-640,ipsc-train-g2_16_53,ipsc-val-g2_0_15,batch-48,dist-2,dbg-0,dyn-0,gxe2-2


<a id="resnet_1333_"></a>
# resnet-1333 
<a id="pt___resnet_133_3_"></a>
## pt       @ resnet-1333-->p2s
<a id="g2_0_1___pt_resnet_1333_"></a>
### g2_0_1       @ pt/resnet-1333-->p2s
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,resnet-1333,ipsc-g2_0_1,batch-48,save-vis-1,save-csv-1,dist-0
<a id="g2_16_53___pt_resnet_1333_"></a>
### g2_16_53       @ pt/resnet-1333-->p2s
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,resnet-1333,ipsc-g2_16_53,batch-24,save-vis-1,save-csv-1,dist-0

<a id="resnet_c4_640_"></a>
# resnet_c4-640 
<a id="pt___resnet_c4_64_0_"></a>
## pt       @ resnet_c4-640-->p2s
<a id="g2_16_53___pt_resnet_c4_640_"></a>
### g2_16_53       @ pt/resnet_c4-640-->p2s
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,resnet_c4-640,ipsc-g2_16_53,batch-16,save-vis-1,save-csv-1,dist-0

<a id="resnet_c4_133_3_"></a>
# resnet_c4_1333 
<a id="pt___resnet_c4_1333_"></a>
## pt       @ resnet_c4_1333-->p2s
<a id="g2_0_1___pt_resnet_c4_133_3_"></a>
### g2_0_1       @ pt/resnet_c4_1333-->p2s
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,resnet_c4-1333,ipsc-g2_0_1,batch-1,save-vis-1,save-csv-1,dist-0
<a id="g2_16_53___pt_resnet_c4_133_3_"></a>
### g2_16_53       @ pt/resnet_c4_1333-->p2s
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,resnet_c4-1333,ipsc-g2_16_53,batch-1,save-vis-1,save-csv-1,dist-0



