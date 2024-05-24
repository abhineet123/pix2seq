<!-- MarkdownTOC -->

- [resnet-640-16_53](#resnet_640_16_5_3_)
    - [r-2560       @ resnet-640-16_53](#r_2560___resnet_640_16_53_)
        - [s-640-sub-8       @ r-2560/resnet-640-16_53](#s_640_sub_8___r_2560_resnet_640_16_5_3_)
        - [s-640-sub-8-mc       @ r-2560/resnet-640-16_53](#s_640_sub_8_mc___r_2560_resnet_640_16_5_3_)
        - [s-640-sub-8-lac       @ r-2560/resnet-640-16_53](#s_640_sub_8_lac___r_2560_resnet_640_16_5_3_)
        - [sz-640-aug-sub-8       @ r-2560/resnet-640-16_53](#sz_640_aug_sub_8___r_2560_resnet_640_16_5_3_)
        - [sz-640-aug-sub-4       @ r-2560/resnet-640-16_53](#sz_640_aug_sub_4___r_2560_resnet_640_16_5_3_)
        - [s-640-sub-4-lac       @ r-2560/resnet-640-16_53](#s_640_sub_4_lac___r_2560_resnet_640_16_5_3_)
    - [res-640       @ resnet-640-16_53](#res_640___resnet_640_16_53_)
        - [sz-80       @ res-640/resnet-640-16_53](#sz_80___res_640_resnet_640_16_53_)
            - [on-train       @ sz-80/res-640/resnet-640-16_53](#on_train___sz_80_res_640_resnet_640_16_53_)
            - [on-54_126       @ sz-80/res-640/resnet-640-16_53](#on_54_126___sz_80_res_640_resnet_640_16_53_)
        - [sz-80-seq-0       @ res-640/resnet-640-16_53](#sz_80_seq_0___res_640_resnet_640_16_53_)
            - [on-train       @ sz-80-seq-0/res-640/resnet-640-16_53](#on_train___sz_80_seq_0_res_640_resnet_640_16_53_)
            - [on-seq-1       @ sz-80-seq-0/res-640/resnet-640-16_53](#on_seq_1___sz_80_seq_0_res_640_resnet_640_16_53_)
        - [sz-80-mc       @ res-640/resnet-640-16_53](#sz_80_mc___res_640_resnet_640_16_53_)
    - [res-320       @ resnet-640-16_53](#res_320___resnet_640_16_53_)
        - [sz-80       @ res-320/resnet-640-16_53](#sz_80___res_320_resnet_640_16_53_)
            - [on-train       @ sz-80/res-320/resnet-640-16_53](#on_train___sz_80_res_320_resnet_640_16_53_)
            - [on-54_126       @ sz-80/res-320/resnet-640-16_53](#on_54_126___sz_80_res_320_resnet_640_16_53_)
        - [sz-80-aug       @ res-320/resnet-640-16_53](#sz_80_aug___res_320_resnet_640_16_53_)
        - [sz-160       @ res-320/resnet-640-16_53](#sz_160___res_320_resnet_640_16_53_)
            - [on-train       @ sz-160/res-320/resnet-640-16_53](#on_train___sz_160_res_320_resnet_640_16_5_3_)
            - [on-54_126       @ sz-160/res-320/resnet-640-16_53](#on_54_126___sz_160_res_320_resnet_640_16_5_3_)

<!-- /MarkdownTOC -->
<a id="resnet_640_16_5_3_"></a>
# resnet-640-16_53
<a id="r_2560___resnet_640_16_53_"></a>
## r-2560       @ resnet-640-16_53-->p2s_seg
<a id="s_640_sub_8___r_2560_resnet_640_16_5_3_"></a>
### s-640-sub-8       @ r-2560/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-640:r-2560:sub-8,batch-24,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1
`seq-0`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-640:r-2560:sub-8:seq-0,batch-8,dbg-0,dyn-1,dist-0,ep-10000,gz,pt-1

<a id="s_640_sub_8_mc___r_2560_resnet_640_16_5_3_"></a>
### s-640-sub-8-mc       @ r-2560/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-640:r-2560:sub-8,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,mc

<a id="s_640_sub_8_lac___r_2560_resnet_640_16_5_3_"></a>
### s-640-sub-8-lac       @ r-2560/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-640:r-2560:sub-8,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,lac

<a id="sz_640_aug_sub_8___r_2560_resnet_640_16_5_3_"></a>
### sz-640-aug-sub-8       @ r-2560/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-640:r-2560:rot-15_345_4:sub-8,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1
`seq-0`
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-640:r-2560:rot-15_345_4:seq-0:sub-8,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1

<a id="sz_640_aug_sub_4___r_2560_resnet_640_16_5_3_"></a>
### sz-640-aug-sub-4       @ r-2560/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg_160-16_53:p-640:r-2560:rot-15_345_4:sub-4,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1

<a id="s_640_sub_4_lac___r_2560_resnet_640_16_5_3_"></a>
### s-640-sub-4-lac       @ r-2560/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg_160-16_53:p-640:r-2560:sub-4,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,lac

<a id="res_640___resnet_640_16_53_"></a>
## res-640       @ resnet-640-16_53-->p2s_seg
<a id="sz_80___res_640_resnet_640_16_53_"></a>
### sz-80       @ res-640/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-80:r-640,batch-36,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1
<a id="on_train___sz_80_res_640_resnet_640_16_53_"></a>
#### on-train       @ sz-80/res-640/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_640-16_53-80_80-80_80-batch_36,_eval_,batch-18,save-vis-1,dbg-0,dyn-1,seg-16_53:p-80:r-640
<a id="on_54_126___sz_80_res_640_resnet_640_16_53_"></a>
#### on-54_126       @ sz-80/res-640/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_640-16_53-80_80-80_80-batch_36,_eval_,batch-18,save-vis-1,dbg-0,dyn-1,seg-54_126:p-80:r-640

<a id="sz_80_seq_0___res_640_resnet_640_16_53_"></a>
### sz-80-seq-0       @ res-640/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,batch-36,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,seg-16_53:p-80:r-640:seq-0
<a id="on_train___sz_80_seq_0_res_640_resnet_640_16_53_"></a>
#### on-train       @ sz-80-seq-0/res-640/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_640-16_53-80_80-80_80-seq_0_0-batch_36,_eval_,batch-18,save-vis-1,dbg-0,dyn-1,seg-16_53:p-80:r-640:seq-0
<a id="on_seq_1___sz_80_seq_0_res_640_resnet_640_16_53_"></a>
#### on-seq-1       @ sz-80-seq-0/res-640/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_640-16_53-80_80-80_80-seq_0_0-batch_36,_eval_,batch-18,save-vis-1,dbg-0,dyn-1,seg-16_53:p-80:r-640:seq-1

<a id="sz_80_mc___res_640_resnet_640_16_53_"></a>
### sz-80-mc       @ res-640/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-80:r-640,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,mc

<a id="res_320___resnet_640_16_53_"></a>
## res-320       @ resnet-640-16_53-->p2s_seg
<a id="sz_80___res_320_resnet_640_16_53_"></a>
### sz-80       @ res-320/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-80:r-320,batch-24,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1
<a id="on_train___sz_80_res_320_resnet_640_16_53_"></a>
#### on-train       @ sz-80/res-320/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_320-16_53-80_80-80_80-batch_24,_eval_,batch-18,save-vis-1,dbg-0,dyn-1,seg-16_53:p-80:r-320
<a id="on_54_126___sz_80_res_320_resnet_640_16_53_"></a>
#### on-54_126       @ sz-80/res-320/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_320-16_53-80_80-80_80-batch_24,_eval_,batch-18,save-vis-1,dbg-0,dyn-1,seg-54_126:p-80:r-320

<a id="sz_80_aug___res_320_resnet_640_16_53_"></a>
### sz-80-aug       @ res-320/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:p-80:r-320:strd-40_80:rot-15_345_4:flip-1,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1

<a id="sz_160___res_320_resnet_640_16_53_"></a>
### sz-160       @ res-320/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg_160-16_53:p-160:r-320,batch-8,dbg-0,dyn-1,dist-0,ep-10000,gz,pt-1
<a id="on_train___sz_160_res_320_resnet_640_16_5_3_"></a>
#### on-train       @ sz-160/res-320/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_320-16_53-160_160-160_160-batch_8,_eval_,batch-18,save-vis-1,dbg-0,dyn-1,seg_160-16_53:p-160:r-320
<a id="on_54_126___sz_160_res_320_resnet_640_16_5_3_"></a>
#### on-54_126       @ sz-160/res-320/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_320-16_53-160_160-160_160-batch_8,_eval_,batch-18,save-vis-1,dbg-0,dyn-1,seg_160-54_126:p-160:r-320


