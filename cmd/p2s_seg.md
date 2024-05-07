<!-- MarkdownTOC -->

- [resnet-640-16_53](#resnet_640_16_5_3_)
    - [res-640-sz-80       @ resnet-640-16_53](#res_640_sz_80___resnet_640_16_53_)
        - [on-train       @ res-640-sz-80/resnet-640-16_53](#on_train___res_640_sz_80_resnet_640_16_53_)
        - [on-54_126       @ res-640-sz-80/resnet-640-16_53](#on_54_126___res_640_sz_80_resnet_640_16_53_)
    - [res-640-sz-80-seq-0       @ resnet-640-16_53](#res_640_sz_80_seq_0___resnet_640_16_53_)
        - [on-train       @ res-640-sz-80-seq-0/resnet-640-16_53](#on_train___res_640_sz_80_seq_0_resnet_640_16_53_)
        - [on-seq-1       @ res-640-sz-80-seq-0/resnet-640-16_53](#on_seq_1___res_640_sz_80_seq_0_resnet_640_16_53_)
    - [res-320       @ resnet-640-16_53](#res_320___resnet_640_16_53_)
        - [sz-80       @ res-320/resnet-640-16_53](#sz_80___res_320_resnet_640_16_53_)
            - [on-train       @ sz-80/res-320/resnet-640-16_53](#on_train___sz_80_res_320_resnet_640_16_53_)
            - [on-54_126       @ sz-80/res-320/resnet-640-16_53](#on_54_126___sz_80_res_320_resnet_640_16_53_)
        - [sz-80-aug       @ res-320/resnet-640-16_53](#sz_80_aug___res_320_resnet_640_16_53_)
        - [sz-160       @ res-320/resnet-640-16_53](#sz_160___res_320_resnet_640_16_53_)

<!-- /MarkdownTOC -->
<a id="resnet_640_16_5_3_"></a>
# resnet-640-16_53
<a id="res_640_sz_80___resnet_640_16_53_"></a>
## res-640-sz-80       @ resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:sz-80:res-640,batch-36,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1
<a id="on_train___res_640_sz_80_resnet_640_16_53_"></a>
### on-train       @ res-640-sz-80/resnet-640-16_53-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ext_reorg_roi_g2-frame-16_53-batch_18,_eval_,ipsc-16_53,batch-36,save-vis-1,dbg-0,dyn-1
<a id="on_54_126___res_640_sz_80_resnet_640_16_53_"></a>
### on-54_126       @ res-640-sz-80/resnet-640-16_53-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ext_reorg_roi_g2-frame-16_53-batch_18,_eval_,ipsc-54_126,batch-36,save-vis-1,dbg-0,dyn-1

<a id="res_640_sz_80_seq_0___resnet_640_16_53_"></a>
## res-640-sz-80-seq-0       @ resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,batch-36,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,seg-16_53:sz-80:res-640:seq-0
<a id="on_train___res_640_sz_80_seq_0_resnet_640_16_53_"></a>
### on-train       @ res-640-sz-80-seq-0/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_640-16_53-80_80-80_80-seq_0_0-batch_36,_eval_,batch-18,save-vis-1,dbg-0,dyn-1,seg-16_53:sz-80:res-640:seq-0

<a id="on_seq_1___res_640_sz_80_seq_0_resnet_640_16_53_"></a>
### on-seq-1       @ res-640-sz-80-seq-0/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_640-16_53-80_80-80_80-seq_0_0-batch_36,_eval_,batch-18,save-vis-1,dbg-0,dyn-1,seg-16_53:sz-80:res-640:seq-1

<a id="res_320___resnet_640_16_53_"></a>
## res-320       @ resnet-640-16_53-->p2s_seg
<a id="sz_80___res_320_resnet_640_16_53_"></a>
### sz-80       @ res-320/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:sz-80:res-320,batch-24,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1
<a id="on_train___sz_80_res_320_resnet_640_16_53_"></a>
#### on-train       @ sz-80/res-320/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_320-16_53-80_80-80_80-batch_24,_eval_,batch-18,save-vis-1,dbg-0,dyn-1,seg-16_53:sz-80:res-320
<a id="on_54_126___sz_80_res_320_resnet_640_16_53_"></a>
#### on-54_126       @ sz-80/res-320/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_320-16_53-80_80-80_80-batch_24,_eval_,batch-18,save-vis-1,dbg-0,dyn-1,seg-54_126:sz-80:res-320

<a id="sz_80_aug___res_320_resnet_640_16_53_"></a>
### sz-80-aug       @ res-320/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:sz-80:res-smin-0:smax-0:rmin-15:rmax-345:rnum-0:flip-0:320,batch-24,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1

<a id="sz_160___res_320_resnet_640_16_53_"></a>
### sz-160       @ res-320/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg_160-16_53:sz-160:res-320,batch-8,dbg-0,dyn-1,dist-0,ep-10000,gz,pt-1



