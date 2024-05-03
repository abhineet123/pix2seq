<!-- MarkdownTOC -->

- [resnet-640](#resnet_64_0_)
    - [ipsc-16_53       @ resnet-640](#ipsc_16_53___resnet_640_)
        - [on-16_53       @ ipsc-16_53/resnet-640](#on_16_53___ipsc_16_53_resnet_64_0_)
        - [on-54_126       @ ipsc-16_53/resnet-640](#on_54_126___ipsc_16_53_resnet_64_0_)
    - [ipsc-0_37       @ resnet-640](#ipsc_0_37___resnet_640_)
        - [on-54_126       @ ipsc-0_37/resnet-640](#on_54_126___ipsc_0_37_resnet_640_)
    - [ipsc-16_53-jtr-res-1280       @ resnet-640](#ipsc_16_53_jtr_res_1280___resnet_640_)
        - [on-train       @ ipsc-16_53-jtr-res-1280/resnet-640](#on_train___ipsc_16_53_jtr_res_1280_resnet_640_)
            - [acc       @ on-train/ipsc-16_53-jtr-res-1280/resnet-640](#acc___on_train_ipsc_16_53_jtr_res_1280_resnet_64_0_)
        - [on-54_126       @ ipsc-16_53-jtr-res-1280/resnet-640](#on_54_126___ipsc_16_53_jtr_res_1280_resnet_640_)
            - [acc       @ on-54_126/ipsc-16_53-jtr-res-1280/resnet-640](#acc___on_54_126_ipsc_16_53_jtr_res_1280_resnet_640_)
    - [ipsc-0_37-jtr-res-1280       @ resnet-640](#ipsc_0_37_jtr_res_1280___resnet_640_)
        - [on-54_126       @ ipsc-0_37-jtr-res-1280/resnet-640](#on_54_126___ipsc_0_37_jtr_res_1280_resnet_64_0_)
            - [acc       @ on-54_126/ipsc-0_37-jtr-res-1280/resnet-640](#acc___on_54_126_ipsc_0_37_jtr_res_1280_resnet_64_0_)
    - [ipsc-16_53-buggy       @ resnet-640](#ipsc_16_53_buggy___resnet_640_)
        - [batch-4-scratch       @ ipsc-16_53-buggy/resnet-640](#batch_4_scratch___ipsc_16_53_buggy_resnet_64_0_)
            - [on-g2_0_15       @ batch-4-scratch/ipsc-16_53-buggy/resnet-640](#on_g2_0_15___batch_4_scratch_ipsc_16_53_buggy_resnet_64_0_)
            - [on-g2_54_126       @ batch-4-scratch/ipsc-16_53-buggy/resnet-640](#on_g2_54_126___batch_4_scratch_ipsc_16_53_buggy_resnet_64_0_)
        - [dist-1       @ ipsc-16_53-buggy/resnet-640](#dist_1___ipsc_16_53_buggy_resnet_64_0_)
        - [local       @ ipsc-16_53-buggy/resnet-640](#local___ipsc_16_53_buggy_resnet_64_0_)
        - [xe       @ ipsc-16_53-buggy/resnet-640](#xe___ipsc_16_53_buggy_resnet_64_0_)
            - [on-g2_0_15       @ xe/ipsc-16_53-buggy/resnet-640](#on_g2_0_15___xe_ipsc_16_53_buggy_resnet_640_)
            - [on-g2_54_126       @ xe/ipsc-16_53-buggy/resnet-640](#on_g2_54_126___xe_ipsc_16_53_buggy_resnet_640_)
        - [gx       @ ipsc-16_53-buggy/resnet-640](#gx___ipsc_16_53_buggy_resnet_64_0_)
        - [gxe       @ ipsc-16_53-buggy/resnet-640](#gxe___ipsc_16_53_buggy_resnet_64_0_)
            - [on-g2_0_15       @ gxe/ipsc-16_53-buggy/resnet-640](#on_g2_0_15___gxe_ipsc_16_53_buggy_resnet_64_0_)
            - [on-g2_54_126       @ gxe/ipsc-16_53-buggy/resnet-640](#on_g2_54_126___gxe_ipsc_16_53_buggy_resnet_64_0_)
    - [ipsc-0_37-buggy       @ resnet-640](#ipsc_0_37_buggy___resnet_640_)
        - [batch_6       @ ipsc-0_37-buggy/resnet-640](#batch_6___ipsc_0_37_buggy_resnet_640_)
        - [gxe       @ ipsc-0_37-buggy/resnet-640](#gxe___ipsc_0_37_buggy_resnet_640_)
            - [on-g2_38_53       @ gxe/ipsc-0_37-buggy/resnet-640](#on_g2_38_53___gxe_ipsc_0_37_buggy_resnet_640_)
    - [pt       @ resnet-640](#pt___resnet_640_)
        - [on-mninstmot       @ pt/resnet-640](#on_mninstmot___pt_resnet_64_0_)
        - [on-g2_0_1       @ pt/resnet-640](#on_g2_0_1___pt_resnet_64_0_)
        - [on-g2_16_53       @ pt/resnet-640](#on_g2_16_53___pt_resnet_64_0_)
            - [dist       @ on-g2_16_53/pt/resnet-640](#dist___on_g2_16_53_pt_resnet_64_0_)
        - [on-g2_54_126       @ pt/resnet-640](#on_g2_54_126___pt_resnet_64_0_)
        - [on-g2_0_15       @ pt/resnet-640](#on_g2_0_15___pt_resnet_64_0_)
- [resnet-1333](#resnet_1333_)
    - [pt       @ resnet-1333](#pt___resnet_133_3_)
        - [on-g2_0_1       @ pt/resnet-1333](#on_g2_0_1___pt_resnet_1333_)
        - [on-g2_16_53       @ pt/resnet-1333](#on_g2_16_53___pt_resnet_1333_)
    - [g2_16_53       @ resnet-1333](#g2_16_53___resnet_133_3_)
- [resnet_c4-640](#resnet_c4_640_)
    - [pt       @ resnet_c4-640](#pt___resnet_c4_64_0_)
        - [on-g2_16_53       @ pt/resnet_c4-640](#on_g2_16_53___pt_resnet_c4_640_)
- [resnet_c4-1024](#resnet_c4_102_4_)
    - [g2_0_37       @ resnet_c4-1024](#g2_0_37___resnet_c4_1024_)
- [resnet_c4_1333](#resnet_c4_133_3_)
    - [pt       @ resnet_c4_1333](#pt___resnet_c4_1333_)
        - [on-g2_0_1       @ pt/resnet_c4_1333](#on_g2_0_1___pt_resnet_c4_133_3_)
        - [on-g2_16_53       @ pt/resnet_c4_1333](#on_g2_16_53___pt_resnet_c4_133_3_)
    - [g2_0_37       @ resnet_c4_1333](#g2_0_37___resnet_c4_1333_)

<!-- /MarkdownTOC -->
<a id="resnet_64_0_"></a>
# resnet-640 
<a id="ipsc_16_53___resnet_640_"></a>
## ipsc-16_53       @ resnet-640-->p2s_seg
python3 run.py --cfg=configs/config_sem_seg.py  --j5=train,resnet-640,seg-16_53:size-80:smin-0:smax-0:rmin-15:rmax-345:rnum-0:flip-0:resize-640,batch-18,dbg-0,dyn-1,dist-0,ep-10000,gz,pt-1

<a id="on_16_53___ipsc_16_53_resnet_64_0_"></a>
### on-16_53       @ ipsc-16_53/resnet-640-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_sem_seg.py  --j5=m-resnet_640_ext_reorg_roi_g2-frame-16_53-batch_18,_eval_,ipsc-16_53,batch-36,save-vis-1,dbg-0,dyn-1
<a id="on_54_126___ipsc_16_53_resnet_64_0_"></a>
### on-54_126       @ ipsc-16_53/resnet-640-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_sem_seg.py  --j5=m-resnet_640_ext_reorg_roi_g2-frame-16_53-batch_18,_eval_,ipsc-54_126,batch-36,save-vis-1,dbg-0,dyn-1

<a id="ipsc_0_37___resnet_640_"></a>
## ipsc-0_37       @ resnet-640-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_sem_seg.py  --j5=val-1,ipsc-54_126_8,batch-24,train,resnet-640,ipsc-0_37,batch-18,dbg-0,dyn-1,dist-0,ep-10000,gz,pt-1
<a id="on_54_126___ipsc_0_37_resnet_640_"></a>
### on-54_126       @ ipsc-0_37/resnet-640-->p2s_seg
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_sem_seg.py  --j5=m-resnet_640_ext_reorg_roi_g2-0_37-batch_18,_eval_,ipsc-54_126,batch-36,save-vis-1,dbg-0,dyn-1,dist-0

<a id="ipsc_16_53_jtr_res_1280___resnet_640_"></a>
## ipsc-16_53-jtr-res-1280       @ resnet-640-->p2s_seg
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_sem_seg.py  --j5=val-1,ipsc-54_126_8,batch-24,train,resnet-640,ipsc-16_53,batch-18,dbg-0,dyn-1,dist-0,ep-10000,gz,pt-1,jtr,res-1280
<a id="on_train___ipsc_16_53_jtr_res_1280_resnet_640_"></a>
### on-train       @ ipsc-16_53-jtr-res-1280/resnet-640-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_sem_seg.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-batch_18-jtr-res_1280,_eval_,ipsc-16_53,batch-16,save-vis-1
<a id="acc___on_train_ipsc_16_53_jtr_res_1280_resnet_64_0_"></a>
#### acc       @ on-train/ipsc-16_53-jtr-res-1280/resnet-640-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_sem_seg.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-batch_18-jtr-res_1280,_eval_,ipsc-16_53,batch-16,save-vis-1,acc

<a id="on_54_126___ipsc_16_53_jtr_res_1280_resnet_640_"></a>
### on-54_126       @ ipsc-16_53-jtr-res-1280/resnet-640-->p2s_seg
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_sem_seg.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-batch_18-jtr-res_1280,_eval_,ipsc-54_126,batch-16,save-vis-1
<a id="acc___on_54_126_ipsc_16_53_jtr_res_1280_resnet_640_"></a>
#### acc       @ on-54_126/ipsc-16_53-jtr-res-1280/resnet-640-->p2s_seg
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_sem_seg.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-batch_18-jtr-res_1280,_eval_,ipsc-54_126,batch-16,save-vis-1,acc

<a id="ipsc_0_37_jtr_res_1280___resnet_640_"></a>
## ipsc-0_37-jtr-res-1280       @ resnet-640-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_sem_seg.py  --j5=val-1,ipsc-54_126_8,batch-24,train,resnet-640,ipsc-0_37,batch-18,dbg-0,dyn-1,dist-0,ep-10000,gz,pt-1,jtr,res-1280
<a id="on_54_126___ipsc_0_37_jtr_res_1280_resnet_64_0_"></a>
### on-54_126       @ ipsc-0_37-jtr-res-1280/resnet-640-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_sem_seg.py  --j5=m-resnet_640_ext_reorg_roi_g2-0_37-batch_18-jtr-res_1280,_eval_,ipsc-54_126,batch-16,save-vis-1
<a id="acc___on_54_126_ipsc_0_37_jtr_res_1280_resnet_64_0_"></a>
#### acc       @ on-54_126/ipsc-0_37-jtr-res-1280/resnet-640-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_sem_seg.py  --j5=m-resnet_640_ext_reorg_roi_g2-0_37-batch_18-jtr-res_1280,_eval_,ipsc-54_126,batch-16,save-vis-1,acc

<a id="ipsc_16_53_buggy___resnet_640_"></a>
## ipsc-16_53-buggy       @ resnet-640-->p2s_seg
<a id="batch_4_scratch___ipsc_16_53_buggy_resnet_64_0_"></a>
### batch-4-scratch       @ ipsc-16_53-buggy/resnet-640-->p2s_seg
python3 run.py --cfg=configs/config_sem_seg.py  --j5=train,resnet-640,ipsc-16_53,batch-4,scratch,dbg-1,dyn-1
<a id="on_g2_0_15___batch_4_scratch_ipsc_16_53_buggy_resnet_64_0_"></a>
#### on-g2_0_15       @ batch-4-scratch/ipsc-16_53-buggy/resnet-640-->p2s_seg
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_sem_seg.py  --j5=eval,m-resnet_640_ext_reorg_roi_g2_16_53-batch_4-scratch,ipsc-0_15,batch-32,save-vis-1
<a id="on_g2_54_126___batch_4_scratch_ipsc_16_53_buggy_resnet_64_0_"></a>
#### on-g2_54_126       @ batch-4-scratch/ipsc-16_53-buggy/resnet-640-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_sem_seg.py  --j5=eval,m-resnet_640_ext_reorg_roi_g2_16_53-batch_4-scratch,ipsc-54_126,batch-32,save-vis-1

<a id="dist_1___ipsc_16_53_buggy_resnet_64_0_"></a>
### dist-1       @ ipsc-16_53-buggy/resnet-640-->p2s_seg
python3 run.py --cfg=configs/config_sem_seg.py  --j5=train,resnet-640,ipsc-16_53,batch-12,dist-1,dbg-0,dyn-0

<a id="local___ipsc_16_53_buggy_resnet_64_0_"></a>
### local       @ ipsc-16_53-buggy/resnet-640-->p2s_seg
python3 run.py --cfg=configs/config_sem_seg.py  --j5=train,resnet-640,ipsc-16_53,batch-4,dbg-0,dyn-1,local-0,gpu-0

python3 run.py --cfg=configs/config_sem_seg.py  --j5=train,resnet-640,ipsc-16_53,batch-4,dbg-0,dyn-1,local-1,gpu-1
<a id="xe___ipsc_16_53_buggy_resnet_64_0_"></a>
### xe       @ ipsc-16_53-buggy/resnet-640-->p2s_seg
python3 run.py --cfg=configs/config_sem_seg.py  --j5=train,resnet-640,ipsc-16_53,batch-32,dbg-0,dyn-1,xe
<a id="on_g2_0_15___xe_ipsc_16_53_buggy_resnet_640_"></a>
#### on-g2_0_15       @ xe/ipsc-16_53-buggy/resnet-640-->p2s_seg
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_sem_seg.py  --j5=eval,m-resnet_640_ext_reorg_roi_g2_16_53-batch_32-xe,ipsc-0_15,batch-64,save-vis-1,dist-0
<a id="on_g2_54_126___xe_ipsc_16_53_buggy_resnet_640_"></a>
#### on-g2_54_126       @ xe/ipsc-16_53-buggy/resnet-640-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_sem_seg.py  --j5=eval,m-resnet_640_ext_reorg_roi_g2_16_53-batch_32-xe,ipsc-54_126,batch-64,save-vis-1,dist-0
<a id="gx___ipsc_16_53_buggy_resnet_64_0_"></a>
### gx       @ ipsc-16_53-buggy/resnet-640-->p2s_seg
python3 run.py --cfg=configs/config_sem_seg.py  --j5=train,resnet-640,ipsc-16_53,batch-36,dbg-0,dyn-1,gx
<a id="gxe___ipsc_16_53_buggy_resnet_64_0_"></a>
### gxe       @ ipsc-16_53-buggy/resnet-640-->p2s_seg
python3 run.py --cfg=configs/config_sem_seg.py  --j5=train,resnet-640,ipsc-16_53,ep-500,batch-48,dbg-0,dyn-0,gxe
<a id="on_g2_0_15___gxe_ipsc_16_53_buggy_resnet_64_0_"></a>
#### on-g2_0_15       @ gxe/ipsc-16_53-buggy/resnet-640-->p2s_seg
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_sem_seg.py  --j5=eval,m-resnet_640_ext_reorg_roi_g2_16_53-batch_48-gxe,ipsc-0_15,batch-32,save-vis-1,dist-1
<a id="on_g2_54_126___gxe_ipsc_16_53_buggy_resnet_64_0_"></a>
#### on-g2_54_126       @ gxe/ipsc-16_53-buggy/resnet-640-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_sem_seg.py  --j5=eval,m-resnet_640_ext_reorg_roi_g2_16_53-batch_48-gxe,ipsc-54_126,batch-32,save-vis-1,dist-1

<a id="ipsc_0_37_buggy___resnet_640_"></a>
## ipsc-0_37-buggy       @ resnet-640-->p2s_seg
\<a id="batch_6___ipsc_0_37_resnet_640_"></a>
<a id="batch_6___ipsc_0_37_buggy_resnet_640_"></a>
### batch_6       @ ipsc-0_37-buggy/resnet-640-->p2s_seg
python3 run.py --cfg=configs/config_sem_seg.py  --j5=train,resnet-640,ipsc-0_37,batch-8,dbg-1,dyn-1
<a id="gxe___ipsc_0_37_buggy_resnet_640_"></a>
### gxe       @ ipsc-0_37-buggy/resnet-640-->p2s_seg
python3 run.py --cfg=configs/config_sem_seg.py  --j5=train,resnet-640,ipsc-0_37,batch-48,dbg-0,dyn-0,gxe,ep-4000
<a id="on_g2_38_53___gxe_ipsc_0_37_buggy_resnet_640_"></a>
#### on-g2_38_53       @ gxe/ipsc-0_37-buggy/resnet-640-->p2s_seg
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_sem_seg.py  --j5=eval,m-resnet_640_ext_reorg_roi_g2_0_37-batch_48-gxe,ipsc-38_53,batch-16,save-vis-1,conf-0

<a id="pt___resnet_640_"></a>
## pt       @ resnet-640-->p2s_seg
<a id="on_mninstmot___pt_resnet_64_0_"></a>
### on-mninstmot       @ pt/resnet-640-->p2s_seg
python3 run.py --mode=eval --model_dir=pretrained/resnet_640 --cfg=configs/config_det_mninstmot.py
<a id="on_g2_0_1___pt_resnet_64_0_"></a>
### on-g2_0_1       @ pt/resnet-640-->p2s_seg
``batch-48``  
python3 run.py --cfg=configs/config_sem_seg.py  --j5=eval,pt,resnet-640,ipsc-0_1,batch-48,save-vis-1,save-csv-1
``batch-2``  
python3 run.py --cfg=configs/config_sem_seg.py  --j5=eval,pt,resnet-640,ipsc-0_1,batch-2,save-vis-1,save-csv-1
<a id="on_g2_16_53___pt_resnet_64_0_"></a>
### on-g2_16_53       @ pt/resnet-640-->p2s_seg
``batch-64``  
python3 run.py --cfg=configs/config_sem_seg.py  --j5=eval,pt,resnet-640,ipsc-16_53,batch-64,save-vis-1,save-csv-1
``batch-32``  
python3 run.py --cfg=configs/config_sem_seg.py  --j5=eval,pt,resnet-640,ipsc-16_53,batch-32,save-vis-1,save-csv-1
<a id="dist___on_g2_16_53_pt_resnet_64_0_"></a>
#### dist       @ on-g2_16_53/pt/resnet-640-->p2s_seg
python3 run.py --cfg=configs/config_sem_seg.py  --j5=eval,pt,resnet-640,ipsc-0_1,batch-2,save-vis-1,save-csv-1,dist-1,eager-0

<a id="on_g2_54_126___pt_resnet_64_0_"></a>
### on-g2_54_126       @ pt/resnet-640-->p2s_seg
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_sem_seg.py  --j5=eval,pt,resnet-640,ipsc-54_126,batch-32,save-vis-0,save-csv-1
<a id="on_g2_0_15___pt_resnet_64_0_"></a>
### on-g2_0_15       @ pt/resnet-640-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_sem_seg.py  --j5=eval,pt,resnet-640,ipsc-0_15,batch-32,save-vis-0,save-csv-1


<a id="resnet_1333_"></a>
# resnet-1333 
<a id="pt___resnet_133_3_"></a>
## pt       @ resnet-1333-->p2s_seg
<a id="on_g2_0_1___pt_resnet_1333_"></a>
### on-g2_0_1       @ pt/resnet-1333-->p2s_seg
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_sem_seg.py  --j5=eval,pt,resnet-1333,ipsc-0_1,batch-48,save-vis-1,save-csv-1
<a id="on_g2_16_53___pt_resnet_1333_"></a>
### on-g2_16_53       @ pt/resnet-1333-->p2s_seg
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_sem_seg.py  --j5=eval,pt,resnet-1333,ipsc-16_53,batch-24,save-vis-1,save-csv-1

<a id="g2_16_53___resnet_133_3_"></a>
## g2_16_53       @ resnet-1333-->p2s_seg
python3 run.py --cfg=configs/config_sem_seg.py  --j5=train,resnet-1333,ipsc-16_53,batch-2,dbg-1,dyn-1

<a id="resnet_c4_640_"></a>
# resnet_c4-640 
<a id="pt___resnet_c4_64_0_"></a>
## pt       @ resnet_c4-640-->p2s_seg
<a id="on_g2_16_53___pt_resnet_c4_640_"></a>
### on-g2_16_53       @ pt/resnet_c4-640-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_sem_seg.py  --j5=eval,pt,resnet_c4-640,ipsc-16_53,batch-16,save-vis-1,save-csv-1

<a id="resnet_c4_102_4_"></a>
# resnet_c4-1024 
<a id="g2_0_37___resnet_c4_1024_"></a>
## g2_0_37       @ resnet_c4-1024-->p2s_seg
python3 run.py --cfg=configs/config_sem_seg.py  --j5=train,resnet_c4-1024,ipsc-0_37,batch-6,dbg-0,dyn-0,gxe,ep-1000

<a id="resnet_c4_133_3_"></a>
# resnet_c4_1333 
<a id="pt___resnet_c4_1333_"></a>
## pt       @ resnet_c4_1333-->p2s_seg
<a id="on_g2_0_1___pt_resnet_c4_133_3_"></a>
### on-g2_0_1       @ pt/resnet_c4_1333-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_sem_seg.py  --j5=eval,pt,resnet_c4-1333,ipsc-0_1,batch-1,save-vis-1,save-csv-1
<a id="on_g2_16_53___pt_resnet_c4_133_3_"></a>
### on-g2_16_53       @ pt/resnet_c4_1333-->p2s_seg
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_sem_seg.py  --j5=eval,pt,resnet_c4-1333,ipsc-16_53,batch-1,save-vis-1,save-csv-1

<a id="g2_0_37___resnet_c4_1333_"></a>
## g2_0_37       @ resnet_c4_1333-->p2s_seg
python3 run.py --cfg=configs/config_sem_seg.py  --j5=train,resnet_c4-1333,ipsc-0_37,batch-6,dbg-0,dyn-0,gxe,ep-1000


