<!-- MarkdownTOC -->

- [resnet-640-16_53](#resnet_640_16_5_3_)
    - [r-2560       @ resnet-640-16_53](#r_2560___resnet_640_16_53_)
        - [p-640-sub-8       @ r-2560/resnet-640-16_53](#p_640_sub_8___r_2560_resnet_640_16_5_3_)
            - [on-16_53       @ p-640-sub-8/r-2560/resnet-640-16_53](#on_16_53___p_640_sub_8_r_2560_resnet_640_16_5_3_)
            - [on-54_126       @ p-640-sub-8/r-2560/resnet-640-16_53](#on_54_126___p_640_sub_8_r_2560_resnet_640_16_5_3_)
        - [p-640-sub-8-mc       @ r-2560/resnet-640-16_53](#p_640_sub_8_mc___r_2560_resnet_640_16_5_3_)
            - [on-16_53       @ p-640-sub-8-mc/r-2560/resnet-640-16_53](#on_16_53___p_640_sub_8_mc_r_2560_resnet_640_16_53_)
            - [on-54_126       @ p-640-sub-8-mc/r-2560/resnet-640-16_53](#on_54_126___p_640_sub_8_mc_r_2560_resnet_640_16_53_)
        - [p-640-sub-8-lac       @ r-2560/resnet-640-16_53](#p_640_sub_8_lac___r_2560_resnet_640_16_5_3_)
            - [on-16_53       @ p-640-sub-8-lac/r-2560/resnet-640-16_53](#on_16_53___p_640_sub_8_lac_r_2560_resnet_640_16_5_3_)
            - [on-54_126       @ p-640-sub-8-lac/r-2560/resnet-640-16_53](#on_54_126___p_640_sub_8_lac_r_2560_resnet_640_16_5_3_)
            - [on-54_126-strd-160       @ p-640-sub-8-lac/r-2560/resnet-640-16_53](#on_54_126_strd_160___p_640_sub_8_lac_r_2560_resnet_640_16_5_3_)
        - [p-640-aug-sub-8       @ r-2560/resnet-640-16_53](#p_640_aug_sub_8___r_2560_resnet_640_16_5_3_)
            - [on-train       @ p-640-aug-sub-8/r-2560/resnet-640-16_53](#on_train___p_640_aug_sub_8_r_2560_resnet_640_16_5_3_)
            - [on-54_126       @ p-640-aug-sub-8/r-2560/resnet-640-16_53](#on_54_126___p_640_aug_sub_8_r_2560_resnet_640_16_5_3_)
        - [p-640-aug-sub-8-lac       @ r-2560/resnet-640-16_53](#p_640_aug_sub_8_lac___r_2560_resnet_640_16_5_3_)
        - [p-640-aug-sub-4       @ r-2560/resnet-640-16_53](#p_640_aug_sub_4___r_2560_resnet_640_16_5_3_)
            - [on-train       @ p-640-aug-sub-4/r-2560/resnet-640-16_53](#on_train___p_640_aug_sub_4_r_2560_resnet_640_16_5_3_)
            - [on-54_126       @ p-640-aug-sub-4/r-2560/resnet-640-16_53](#on_54_126___p_640_aug_sub_4_r_2560_resnet_640_16_5_3_)
        - [p-640-sub-4-lac       @ r-2560/resnet-640-16_53](#p_640_sub_4_lac___r_2560_resnet_640_16_5_3_)
            - [on-16_53       @ p-640-sub-4-lac/r-2560/resnet-640-16_53](#on_16_53___p_640_sub_4_lac_r_2560_resnet_640_16_5_3_)
            - [on-54_126       @ p-640-sub-4-lac/r-2560/resnet-640-16_53](#on_54_126___p_640_sub_4_lac_r_2560_resnet_640_16_5_3_)
    - [r-640       @ resnet-640-16_53](#r_640___resnet_640_16_53_)
        - [p-640-aug       @ r-640/resnet-640-16_53](#p_640_aug___r_640_resnet_640_16_53_)
        - [p-640-aug-mc       @ r-640/resnet-640-16_53](#p_640_aug_mc___r_640_resnet_640_16_53_)
        - [p-640-aug-lac       @ r-640/resnet-640-16_53](#p_640_aug_lac___r_640_resnet_640_16_53_)
        - [p-640-aug-lac-sub-4       @ r-640/resnet-640-16_53](#p_640_aug_lac_sub_4___r_640_resnet_640_16_53_)
        - [p-640-aug-lac-sub-8       @ r-640/resnet-640-16_53](#p_640_aug_lac_sub_8___r_640_resnet_640_16_53_)
        - [p-640-aug-lac-2d       @ r-640/resnet-640-16_53](#p_640_aug_lac_2d___r_640_resnet_640_16_53_)
        - [p-80       @ r-640/resnet-640-16_53](#p_80___r_640_resnet_640_16_53_)
            - [on-train       @ p-80/r-640/resnet-640-16_53](#on_train___p_80_r_640_resnet_640_16_5_3_)
            - [on-54_126       @ p-80/r-640/resnet-640-16_53](#on_54_126___p_80_r_640_resnet_640_16_5_3_)
        - [p-80-seq-0       @ r-640/resnet-640-16_53](#p_80_seq_0___r_640_resnet_640_16_53_)
            - [on-train       @ p-80-seq-0/r-640/resnet-640-16_53](#on_train___p_80_seq_0_r_640_resnet_640_16_5_3_)
            - [on-seq-1       @ p-80-seq-0/r-640/resnet-640-16_53](#on_seq_1___p_80_seq_0_r_640_resnet_640_16_5_3_)
        - [p-80-mc       @ r-640/resnet-640-16_53](#p_80_mc___r_640_resnet_640_16_53_)
            - [on-train       @ p-80-mc/r-640/resnet-640-16_53](#on_train___p_80_mc_r_640_resnet_640_16_53_)
            - [on-54_126       @ p-80-mc/r-640/resnet-640-16_53](#on_54_126___p_80_mc_r_640_resnet_640_16_53_)
    - [r-320       @ resnet-640-16_53](#r_320___resnet_640_16_53_)
        - [p-80       @ r-320/resnet-640-16_53](#p_80___r_320_resnet_640_16_53_)
            - [on-train       @ p-80/r-320/resnet-640-16_53](#on_train___p_80_r_320_resnet_640_16_5_3_)
            - [on-54_126       @ p-80/r-320/resnet-640-16_53](#on_54_126___p_80_r_320_resnet_640_16_5_3_)
        - [p-80-aug       @ r-320/resnet-640-16_53](#p_80_aug___r_320_resnet_640_16_53_)
            - [on-train       @ p-80-aug/r-320/resnet-640-16_53](#on_train___p_80_aug_r_320_resnet_640_16_5_3_)
            - [on-54_126       @ p-80-aug/r-320/resnet-640-16_53](#on_54_126___p_80_aug_r_320_resnet_640_16_5_3_)
        - [p-160       @ r-320/resnet-640-16_53](#p_160___r_320_resnet_640_16_53_)
            - [on-train       @ p-160/r-320/resnet-640-16_53](#on_train___p_160_r_320_resnet_640_16_53_)
            - [on-54_126       @ p-160/r-320/resnet-640-16_53](#on_54_126___p_160_r_320_resnet_640_16_53_)

<!-- /MarkdownTOC -->
<a id="resnet_640_16_5_3_"></a>
# resnet-640-16_53
<a id="r_2560___resnet_640_16_53_"></a>
## r-2560       @ resnet-640-16_53-->p2s_seg
<a id="p_640_sub_8___r_2560_resnet_640_16_5_3_"></a>
### p-640-sub-8       @ r-2560/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-640:r-2560:sub-8,batch-24,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1
`seq-0`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-640:r-2560:sub-8:seq-0,batch-8,dbg-0,dyn-1,dist-0,ep-10000,gz,pt-1
<a id="on_16_53___p_640_sub_8_r_2560_resnet_640_16_5_3_"></a>
#### on-16_53       @ p-640-sub-8/r-2560/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-sub_8-batch_24,_eval_,batch-8,save-vis-1,dbg-0,dyn-1,seg-16_53:p-640:r-2560:sub-8
<a id="on_54_126___p_640_sub_8_r_2560_resnet_640_16_5_3_"></a>
#### on-54_126       @ p-640-sub-8/r-2560/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-sub_8-batch_24,_eval_,batch-32,save-vis-1,dbg-0,dyn-1,seg-54_126:p-640:r-2560:sub-8


<a id="p_640_sub_8_mc___r_2560_resnet_640_16_5_3_"></a>
### p-640-sub-8-mc       @ r-2560/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-640:r-2560:sub-8,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,mc
<a id="on_16_53___p_640_sub_8_mc_r_2560_resnet_640_16_53_"></a>
#### on-16_53       @ p-640-sub-8-mc/r-2560/resnet-640-16_53-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-sub_8-mc-batch_32,_eval_,batch-16,save-vis-1,dbg-0,dyn-1,seg-16_53:p-640:r-2560:sub-8,mc
<a id="on_54_126___p_640_sub_8_mc_r_2560_resnet_640_16_53_"></a>
#### on-54_126       @ p-640-sub-8-mc/r-2560/resnet-640-16_53-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-sub_8-mc-batch_32,_eval_,batch-16,save-vis-1,dbg-0,dyn-1,seg-54_126:p-640:r-2560:sub-8,mc

<a id="p_640_sub_8_lac___r_2560_resnet_640_16_5_3_"></a>
### p-640-sub-8-lac       @ r-2560/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-640:r-2560:sub-8,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,lac
<a id="on_16_53___p_640_sub_8_lac_r_2560_resnet_640_16_5_3_"></a>
#### on-16_53       @ p-640-sub-8-lac/r-2560/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-sub_8-lac-batch_32,_eval_,batch-16,save-vis-1,dbg-0,dyn-1,seg-16_53:p-640:r-2560:sub-8,lac
<a id="on_54_126___p_640_sub_8_lac_r_2560_resnet_640_16_5_3_"></a>
#### on-54_126       @ p-640-sub-8-lac/r-2560/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-sub_8-lac-batch_32,_eval_,batch-16,save-vis-1,dbg-0,dyn-1,seg-54_126:p-640:r-2560:sub-8,lac
<a id="on_54_126_strd_160___p_640_sub_8_lac_r_2560_resnet_640_16_5_3_"></a>
#### on-54_126-strd-160       @ p-640-sub-8-lac/r-2560/resnet-640-16_53-->p2s_seg
`dbg`
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-sub_8-lac-batch_32,_eval_,batch-16,save-vis-1,dbg-0,dyn-1,seg-54_126:p-640:r-2560:sub-8:strd-160:seq-0,lac
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-sub_8-lac-batch_32,_eval_,batch-16,save-vis-1,dbg-0,dyn-1,seg-54_55:p-640:r-2560:sub-8:strd-160:seq-0,lac

<a id="p_640_aug_sub_8___r_2560_resnet_640_16_5_3_"></a>
### p-640-aug-sub-8       @ r-2560/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-640:r-2560:rot-15_345_4:sub-8,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1
`dbg`
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-640:r-2560:rot-15_345_4:seq-0:sub-8,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1
<a id="on_train___p_640_aug_sub_8_r_2560_resnet_640_16_5_3_"></a>
#### on-train       @ p-640-aug-sub-8/r-2560/resnet-640-16_53-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-rot_15_345_4-sub_8-batch_32,_eval_,batch-16,save-vis-1,dbg-0,dyn-1,seg-16_53:p-640:r-2560:sub-8
<a id="on_54_126___p_640_aug_sub_8_r_2560_resnet_640_16_5_3_"></a>
#### on-54_126       @ p-640-aug-sub-8/r-2560/resnet-640-16_53-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-rot_15_345_4-sub_8-batch_32,_eval_,batch-16,save-vis-1,dbg-0,dyn-1,seg-54_126:p-640:r-2560:sub-8

<a id="p_640_aug_sub_8_lac___r_2560_resnet_640_16_5_3_"></a>
### p-640-aug-sub-8-lac       @ r-2560/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-640:r-2560:rot-15_345_4:sub-8,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,lac

<a id="p_640_aug_sub_4___r_2560_resnet_640_16_5_3_"></a>
### p-640-aug-sub-4       @ r-2560/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg_160-16_53:p-640:r-2560:rot-15_345_4:sub-4,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1
<a id="on_train___p_640_aug_sub_4_r_2560_resnet_640_16_5_3_"></a>
#### on-train       @ p-640-aug-sub-4/r-2560/resnet-640-16_53-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-rot_15_345_4-sub_4-batch_32,_eval_,batch-16,save-vis-1,dbg-0,dyn-1,seg_160-16_53:p-640:r-2560:sub-4
<a id="on_54_126___p_640_aug_sub_4_r_2560_resnet_640_16_5_3_"></a>
#### on-54_126       @ p-640-aug-sub-4/r-2560/resnet-640-16_53-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-rot_15_345_4-sub_4-batch_32,_eval_,batch-16,save-vis-1,dbg-0,dyn-1,seg_160-54_126:p-640:r-2560:sub-4

<a id="p_640_sub_4_lac___r_2560_resnet_640_16_5_3_"></a>
### p-640-sub-4-lac       @ r-2560/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg_160-16_53:p-640:r-2560:sub-4,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,lac
<a id="on_16_53___p_640_sub_4_lac_r_2560_resnet_640_16_5_3_"></a>
#### on-16_53       @ p-640-sub-4-lac/r-2560/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-sub_4-lac-batch_32,_eval_,batch-16,save-vis-1,dbg-0,dyn-1,seg_160-16_53:p-640:r-2560:sub-4,lac
<a id="on_54_126___p_640_sub_4_lac_r_2560_resnet_640_16_5_3_"></a>
#### on-54_126       @ p-640-sub-4-lac/r-2560/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-sub_4-lac-batch_32,_eval_,batch-16,save-vis-1,dbg-0,dyn-1,seg_160-54_126:p-640:r-2560:sub-4,lac

<a id="r_640___resnet_640_16_53_"></a>
## r-640       @ resnet-640-16_53-->p2s_seg
<a id="p_640_aug___r_640_resnet_640_16_53_"></a>
### p-640-aug       @ r-640/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-640:r-640:rot-15_345_4:flip-1,batch-24,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,seq3k
<a id="p_640_aug_mc___r_640_resnet_640_16_53_"></a>
### p-640-aug-mc       @ r-640/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-640:r-640:rot-15_345_4:flip-1,batch-24,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,mc,seq6k
<a id="p_640_aug_lac___r_640_resnet_640_16_53_"></a>
### p-640-aug-lac       @ r-640/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-640:r-640:rot-15_345_4:flip-1,batch-24,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,lac,seq6k
<a id="p_640_aug_lac_sub_4___r_640_resnet_640_16_53_"></a>
### p-640-aug-lac-sub-4       @ r-640/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-640:r-640:rot-15_345_4:flip-1:sub-4,batch-24,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,lac,seq1k,voc28
<a id="p_640_aug_lac_sub_8___r_640_resnet_640_16_53_"></a>
### p-640-aug-lac-sub-8       @ r-640/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-640:r-640:rot-15_345_4:flip-1:sub-8,batch-24,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,lac,voc28
<a id="p_640_aug_lac_2d___r_640_resnet_640_16_53_"></a>
### p-640-aug-lac-2d       @ r-640/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-640:r-640:rot-15_345_4:flip-1,batch-24,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,lac,2d,seq6k

<a id="p_80___r_640_resnet_640_16_53_"></a>
### p-80       @ r-640/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-80:r-640,batch-36,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1
<a id="on_train___p_80_r_640_resnet_640_16_5_3_"></a>
#### on-train       @ p-80/r-640/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_640-16_53-80_80-80_80-batch_36,_eval_,batch-18,save-vis-1,dbg-0,dyn-1,seg-16_53:p-80:r-640
<a id="on_54_126___p_80_r_640_resnet_640_16_5_3_"></a>
#### on-54_126       @ p-80/r-640/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_640-16_53-80_80-80_80-batch_36,_eval_,batch-18,save-vis-1,dbg-0,dyn-1,seg-54_126:p-80:r-640


<a id="p_80_seq_0___r_640_resnet_640_16_53_"></a>
### p-80-seq-0       @ r-640/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,batch-36,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,seg-16_53:p-80:r-640:seq-0
<a id="on_train___p_80_seq_0_r_640_resnet_640_16_5_3_"></a>
#### on-train       @ p-80-seq-0/r-640/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_640-16_53-80_80-80_80-seq_0_0-batch_36,_eval_,batch-18,save-vis-1,dbg-0,dyn-1,seg-16_53:p-80:r-640:seq-0
<a id="on_seq_1___p_80_seq_0_r_640_resnet_640_16_5_3_"></a>
#### on-seq-1       @ p-80-seq-0/r-640/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_640-16_53-80_80-80_80-seq_0_0-batch_36,_eval_,batch-18,save-vis-1,dbg-0,dyn-1,seg-16_53:p-80:r-640:seq-1


<a id="p_80_mc___r_640_resnet_640_16_53_"></a>
### p-80-mc       @ r-640/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-80:r-640,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,mc
<a id="on_train___p_80_mc_r_640_resnet_640_16_53_"></a>
#### on-train       @ p-80-mc/r-640/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_640-16_53-80_80-80_80-mc-batch_32,_eval_,batch-64,save-vis-1,dbg-0,dyn-1,seg-16_53:p-80:r-640,mc
<a id="on_54_126___p_80_mc_r_640_resnet_640_16_53_"></a>
#### on-54_126       @ p-80-mc/r-640/resnet-640-16_53-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_640-16_53-80_80-80_80-mc-batch_32,_eval_,batch-64,save-vis-1,dbg-0,dyn-1,seg-54_126:p-80:r-640,mc


<a id="r_320___resnet_640_16_53_"></a>
## r-320       @ resnet-640-16_53-->p2s_seg
<a id="p_80___r_320_resnet_640_16_53_"></a>
### p-80       @ r-320/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-80:r-320,batch-24,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1
<a id="on_train___p_80_r_320_resnet_640_16_5_3_"></a>
#### on-train       @ p-80/r-320/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_320-16_53-80_80-80_80-batch_24,_eval_,batch-16,save-vis-1,dbg-0,dyn-1,seg-16_53:p-80:r-320
<a id="on_54_126___p_80_r_320_resnet_640_16_5_3_"></a>
#### on-54_126       @ p-80/r-320/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_320-16_53-80_80-80_80-batch_24,_eval_,batch-16,save-vis-1,dbg-0,dyn-1,seg-54_126:p-80:r-320


<a id="p_80_aug___r_320_resnet_640_16_53_"></a>
### p-80-aug       @ r-320/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-80:r-320:strd-40_80:rot-15_345_4:flip-1,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1
<a id="on_train___p_80_aug_r_320_resnet_640_16_5_3_"></a>
#### on-train       @ p-80-aug/r-320/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_320-16_53-80_80-40_80-rot_15_345_4-flip-batch_32,_eval_,batch-16,save-vis-1,dbg-0,dyn-1,seg-16_53:p-80:r-320
<a id="on_54_126___p_80_aug_r_320_resnet_640_16_5_3_"></a>
#### on-54_126       @ p-80-aug/r-320/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_320-16_53-80_80-40_80-rot_15_345_4-flip-batch_32,_eval_,batch-16,save-vis-1,dbg-0,dyn-1,seg-54_126:p-80:r-320


<a id="p_160___r_320_resnet_640_16_53_"></a>
### p-160       @ r-320/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg_160-16_53:p-160:r-320,batch-8,dbg-0,dyn-1,dist-0,ep-10000,gz,pt-1
<a id="on_train___p_160_r_320_resnet_640_16_53_"></a>
#### on-train       @ p-160/r-320/resnet-640-16_53-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_320-16_53-160_160-160_160-batch_8,_eval_,batch-4,save-vis-1,dbg-0,dyn-1,seg_160-16_53:p-160:r-320
<a id="on_54_126___p_160_r_320_resnet_640_16_53_"></a>
#### on-54_126       @ p-160/r-320/resnet-640-16_53-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_320-16_53-160_160-160_160-batch_8,_eval_,batch-4,save-vis-1,dbg-0,dyn-1,seg_160-54_126:p-160:r-320


