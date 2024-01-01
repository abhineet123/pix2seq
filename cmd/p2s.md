<!-- MarkdownTOC -->

- [tfrecord](#tfrecor_d_)
    - [ipsc       @ tfrecord](#ipsc___tfrecord_)
        - [ext_reorg_roi_g2_16_53       @ ipsc/tfrecord](#ext_reorg_roi_g2_16_53___ipsc_tfrecor_d_)
        - [ext_reorg_roi_g2_0_1       @ ipsc/tfrecord](#ext_reorg_roi_g2_0_1___ipsc_tfrecor_d_)
        - [ext_reorg_roi_g2_0_15       @ ipsc/tfrecord](#ext_reorg_roi_g2_0_15___ipsc_tfrecor_d_)
- [resnet-640](#resnet_64_0_)
    - [pt       @ resnet-640](#pt___resnet_640_)
        - [on-mninstmot       @ pt/resnet-640](#on_mninstmot___pt_resnet_64_0_)
        - [on-g2_0_1       @ pt/resnet-640](#on_g2_0_1___pt_resnet_64_0_)
        - [on-g2_16_53       @ pt/resnet-640](#on_g2_16_53___pt_resnet_64_0_)
            - [dist       @ on-g2_16_53/pt/resnet-640](#dist___on_g2_16_53_pt_resnet_64_0_)
        - [on-g2_54_126       @ pt/resnet-640](#on_g2_54_126___pt_resnet_64_0_)
        - [on-g2_0_15       @ pt/resnet-640](#on_g2_0_15___pt_resnet_64_0_)
    - [g2_16_53       @ resnet-640](#g2_16_53___resnet_640_)
        - [dist-1       @ g2_16_53/resnet-640](#dist_1___g2_16_53_resnet_64_0_)
        - [dist-2       @ g2_16_53/resnet-640](#dist_2___g2_16_53_resnet_64_0_)
            - [local       @ dist-2/g2_16_53/resnet-640](#local___dist_2_g2_16_53_resnet_640_)
            - [xe       @ dist-2/g2_16_53/resnet-640](#xe___dist_2_g2_16_53_resnet_640_)
                - [on-g2_0_15       @ xe/dist-2/g2_16_53/resnet-640](#on_g2_0_15___xe_dist_2_g2_16_53_resnet_64_0_)
                - [on-g2_0_15       @ xe/dist-2/g2_16_53/resnet-640](#on_g2_0_15___xe_dist_2_g2_16_53_resnet_64_0__1)
            - [gx       @ dist-2/g2_16_53/resnet-640](#gx___dist_2_g2_16_53_resnet_640_)
            - [gxe       @ dist-2/g2_16_53/resnet-640](#gxe___dist_2_g2_16_53_resnet_640_)
                - [on-g2_0_15       @ gxe/dist-2/g2_16_53/resnet-640](#on_g2_0_15___gxe_dist_2_g2_16_53_resnet_640_)
                - [on-g2_54_126       @ gxe/dist-2/g2_16_53/resnet-640](#on_g2_54_126___gxe_dist_2_g2_16_53_resnet_640_)
    - [g2_16_53       @ resnet-640](#g2_16_53___resnet_640__1)
- [resnet-1333](#resnet_1333_)
    - [pt       @ resnet-1333](#pt___resnet_133_3_)
        - [on-g2_0_1       @ pt/resnet-1333](#on_g2_0_1___pt_resnet_1333_)
        - [on-g2_16_53       @ pt/resnet-1333](#on_g2_16_53___pt_resnet_1333_)
    - [g2_16_53       @ resnet-1333](#g2_16_53___resnet_133_3_)
- [resnet_c4-640](#resnet_c4_640_)
    - [pt       @ resnet_c4-640](#pt___resnet_c4_64_0_)
        - [on-g2_16_53       @ pt/resnet_c4-640](#on_g2_16_53___pt_resnet_c4_640_)
- [resnet_c4_1333](#resnet_c4_133_3_)
    - [pt       @ resnet_c4_1333](#pt___resnet_c4_1333_)
        - [on-g2_0_1       @ pt/resnet_c4_1333](#on_g2_0_1___pt_resnet_c4_133_3_)
        - [on-g2_16_53       @ pt/resnet_c4_1333](#on_g2_16_53___pt_resnet_c4_133_3_)

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
<a id="on_mninstmot___pt_resnet_64_0_"></a>
### on-mninstmot       @ pt/resnet-640-->p2s
python3 run.py --mode=eval --model_dir=pretrained/resnet_640 --cfg=configs/config_det_mninstmot.py
<a id="on_g2_0_1___pt_resnet_64_0_"></a>
### on-g2_0_1       @ pt/resnet-640-->p2s
``batch-48``  
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,resnet-640,ipsc-g2_0_1,batch-48,save-vis-1,save-csv-1,dist-0
``batch-2``  
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,resnet-640,ipsc-g2_0_1,batch-2,save-vis-1,save-csv-1,dist-0
<a id="on_g2_16_53___pt_resnet_64_0_"></a>
### on-g2_16_53       @ pt/resnet-640-->p2s
``batch-64``  
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,resnet-640,ipsc-g2_16_53,batch-64,save-vis-1,save-csv-1,dist-0
``batch-32``  
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,resnet-640,ipsc-g2_16_53,batch-32,save-vis-1,save-csv-1,dist-0
<a id="dist___on_g2_16_53_pt_resnet_64_0_"></a>
#### dist       @ on-g2_16_53/pt/resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,resnet-640,ipsc-g2_0_1,batch-2,save-vis-1,save-csv-1,dist-1,eager-0

<a id="on_g2_54_126___pt_resnet_64_0_"></a>
### on-g2_54_126       @ pt/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,resnet-640,ipsc-g2_54_126,batch-32,save-vis-0,save-csv-1,dist-0
<a id="on_g2_0_15___pt_resnet_64_0_"></a>
### on-g2_0_15       @ pt/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,resnet-640,ipsc-g2_0_15,batch-32,save-vis-0,save-csv-1,dist-0

<a id="g2_16_53___resnet_640_"></a>
## g2_16_53       @ resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train,resnet-640,ipsc-g2_16_53,batch-6,dist-0,dbg-1,dyn-1
<a id="dist_1___g2_16_53_resnet_64_0_"></a>
### dist-1       @ g2_16_53/resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train,resnet-640,ipsc-g2_16_53,batch-12,dist-1,dbg-0,dyn-0
<a id="dist_2___g2_16_53_resnet_64_0_"></a>
### dist-2       @ g2_16_53/resnet-640-->p2s
<a id="local___dist_2_g2_16_53_resnet_640_"></a>
#### local       @ dist-2/g2_16_53/resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train,resnet-640,ipsc-g2_16_53,batch-4,dbg-0,dyn-1,local-0,gpu-0

python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train,resnet-640,ipsc-g2_16_53,batch-4,dbg-0,dyn-1,local-1,gpu-1

<a id="xe___dist_2_g2_16_53_resnet_640_"></a>
#### xe       @ dist-2/g2_16_53/resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train,resnet-640,ipsc-g2_16_53,batch-32,dbg-0,dyn-1,xe
<a id="on_g2_0_15___xe_dist_2_g2_16_53_resnet_64_0_"></a>
##### on-g2_0_15       @ xe/dist-2/g2_16_53/resnet-640-->p2s
<a id="on_g2_0_15___xe_dist_2_g2_16_53_resnet_64_0__1"></a>
##### on-g2_0_15       @ xe/dist-2/g2_16_53/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,m-resnet_640_ext_reorg_roi_g2_16_53-batch_32-xe,ipsc-g2_0_15,batch-32,save-vis-1,dist-1

python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train,resnet-640,ipsc-g2_16_53,batch-32,dist-0,dbg-1,dyn-1,eval,ipsc-g2_0_15,batch_e-16,save-vis-1,save-csv-1

<a id="gx___dist_2_g2_16_53_resnet_640_"></a>
#### gx       @ dist-2/g2_16_53/resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train,resnet-640,ipsc-g2_16_53,batch-36,dbg-0,dyn-1,gx

<a id="gxe___dist_2_g2_16_53_resnet_640_"></a>
#### gxe       @ dist-2/g2_16_53/resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train,resnet-640,ipsc-g2_16_53,epoch-500,batch-48,dbg-0,dyn-0,gxe
<a id="on_g2_0_15___gxe_dist_2_g2_16_53_resnet_640_"></a>
##### on-g2_0_15       @ gxe/dist-2/g2_16_53/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,m-resnet_640_ext_reorg_roi_g2_16_53-batch_48-gxe,ipsc-g2_0_15,batch-32,save-vis-1,dist-1
<a id="on_g2_54_126___gxe_dist_2_g2_16_53_resnet_640_"></a>
##### on-g2_54_126       @ gxe/dist-2/g2_16_53/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,m-resnet_640_ext_reorg_roi_g2_16_53-batch_48-gxe,ipsc-g2_54_126,batch-32,save-vis-1,dist-1

<a id="g2_16_53___resnet_640__1"></a>
## g2_16_53       @ resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train-dist0,resnet-640,ipsc-g2_16_53,batch-6,dist-0,dbg-1,dyn-1

<a id="resnet_1333_"></a>
# resnet-1333 
<a id="pt___resnet_133_3_"></a>
## pt       @ resnet-1333-->p2s
<a id="on_g2_0_1___pt_resnet_1333_"></a>
### on-g2_0_1       @ pt/resnet-1333-->p2s
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,resnet-1333,ipsc-g2_0_1,batch-48,save-vis-1,save-csv-1,dist-0
<a id="on_g2_16_53___pt_resnet_1333_"></a>
### on-g2_16_53       @ pt/resnet-1333-->p2s
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,resnet-1333,ipsc-g2_16_53,batch-24,save-vis-1,save-csv-1,dist-0

<a id="g2_16_53___resnet_133_3_"></a>
## g2_16_53       @ resnet-1333-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train,resnet-1333,ipsc-g2_16_53,batch-6,dist-0,dbg-1,dyn-1

<a id="resnet_c4_640_"></a>
# resnet_c4-640 
<a id="pt___resnet_c4_64_0_"></a>
## pt       @ resnet_c4-640-->p2s
<a id="on_g2_16_53___pt_resnet_c4_640_"></a>
### on-g2_16_53       @ pt/resnet_c4-640-->p2s
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,resnet_c4-640,ipsc-g2_16_53,batch-16,save-vis-1,save-csv-1,dist-0

<a id="resnet_c4_133_3_"></a>
# resnet_c4_1333 
<a id="pt___resnet_c4_1333_"></a>
## pt       @ resnet_c4_1333-->p2s
<a id="on_g2_0_1___pt_resnet_c4_133_3_"></a>
### on-g2_0_1       @ pt/resnet_c4_1333-->p2s
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,resnet_c4-1333,ipsc-g2_0_1,batch-1,save-vis-1,save-csv-1,dist-0
<a id="on_g2_16_53___pt_resnet_c4_133_3_"></a>
### on-g2_16_53       @ pt/resnet_c4_1333-->p2s
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,resnet_c4-1333,ipsc-g2_16_53,batch-1,save-vis-1,save-csv-1,dist-0



