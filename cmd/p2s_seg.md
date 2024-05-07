<!-- MarkdownTOC -->

- [resnet-640-16_53](#resnet_640_16_5_3_)
    - [resize-640-size-80       @ resnet-640-16_53](#resize_640_size_80___resnet_640_16_53_)
        - [on-train       @ resize-640-size-80/resnet-640-16_53](#on_train___resize_640_size_80_resnet_640_16_5_3_)
        - [on-54_126       @ resize-640-size-80/resnet-640-16_53](#on_54_126___resize_640_size_80_resnet_640_16_5_3_)
    - [resize-640-size-80-seq-0       @ resnet-640-16_53](#resize_640_size_80_seq_0___resnet_640_16_53_)
        - [on-train       @ resize-640-size-80-seq-0/resnet-640-16_53](#on_train___resize_640_size_80_seq_0_resnet_640_16_5_3_)
        - [on-seq-1       @ resize-640-size-80-seq-0/resnet-640-16_53](#on_seq_1___resize_640_size_80_seq_0_resnet_640_16_5_3_)
    - [resize-320       @ resnet-640-16_53](#resize_320___resnet_640_16_53_)
        - [size-80       @ resize-320/resnet-640-16_53](#size_80___resize_320_resnet_640_16_5_3_)
            - [on-train       @ size-80/resize-320/resnet-640-16_53](#on_train___size_80_resize_320_resnet_640_16_5_3_)
            - [on-54_126       @ size-80/resize-320/resnet-640-16_53](#on_54_126___size_80_resize_320_resnet_640_16_5_3_)
        - [size-80-aug       @ resize-320/resnet-640-16_53](#size_80_aug___resize_320_resnet_640_16_5_3_)
        - [size-160       @ resize-320/resnet-640-16_53](#size_160___resize_320_resnet_640_16_5_3_)

<!-- /MarkdownTOC -->
<a id="resnet_640_16_5_3_"></a>
# resnet-640-16_53
<a id="resize_640_size_80___resnet_640_16_53_"></a>
## resize-640-size-80       @ resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:size-80:resize-640,batch-36,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1
<a id="on_train___resize_640_size_80_resnet_640_16_5_3_"></a>
### on-train       @ resize-640-size-80/resnet-640-16_53-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ext_reorg_roi_g2-frame-16_53-batch_18,_eval_,ipsc-16_53,batch-36,save-vis-1,dbg-0,dyn-1
<a id="on_54_126___resize_640_size_80_resnet_640_16_5_3_"></a>
### on-54_126       @ resize-640-size-80/resnet-640-16_53-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ext_reorg_roi_g2-frame-16_53-batch_18,_eval_,ipsc-54_126,batch-36,save-vis-1,dbg-0,dyn-1

<a id="resize_640_size_80_seq_0___resnet_640_16_53_"></a>
## resize-640-size-80-seq-0       @ resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,batch-36,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,seg-16_53:size-80:resize-640:seq-0
<a id="on_train___resize_640_size_80_seq_0_resnet_640_16_5_3_"></a>
### on-train       @ resize-640-size-80-seq-0/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_640-16_53-80_80-80_80-seq_0_0-batch_36,_eval_,batch-18,save-vis-1,dbg-0,dyn-1,seg-16_53:size-80:resize-640:seq-0

<a id="on_seq_1___resize_640_size_80_seq_0_resnet_640_16_5_3_"></a>
### on-seq-1       @ resize-640-size-80-seq-0/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_640-16_53-80_80-80_80-seq_0_0-batch_36,_eval_,batch-18,save-vis-1,dbg-0,dyn-1,seg-16_53:size-80:resize-640:seq-1

<a id="resize_320___resnet_640_16_53_"></a>
## resize-320       @ resnet-640-16_53-->p2s_seg
<a id="size_80___resize_320_resnet_640_16_5_3_"></a>
### size-80       @ resize-320/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:size-80:resize-320,batch-24,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1
<a id="on_train___size_80_resize_320_resnet_640_16_5_3_"></a>
#### on-train       @ size-80/resize-320/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_320-16_53-80_80-80_80-batch_24,_eval_,batch-18,save-vis-1,dbg-0,dyn-1,seg-16_53:size-80:resize-320
<a id="on_54_126___size_80_resize_320_resnet_640_16_5_3_"></a>
#### on-54_126       @ size-80/resize-320/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_320-16_53-80_80-80_80-batch_24,_eval_,batch-18,save-vis-1,dbg-0,dyn-1,seg-54_126:size-80:resize-320

<a id="size_80_aug___resize_320_resnet_640_16_5_3_"></a>
### size-80-aug       @ resize-320/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:size-80:resize-320,batch-24,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1

<a id="size_160___resize_320_resnet_640_16_5_3_"></a>
### size-160       @ resize-320/resnet-640-16_53-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg_160-16_53:size-160:resize-320,batch-8,dbg-0,dyn-1,dist-0,ep-10000,gz,pt-1



