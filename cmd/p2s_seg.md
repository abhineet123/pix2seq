<!-- MarkdownTOC -->

- [resnet-640](#resnet_64_0_)
    - [ipsc-16_53       @ resnet-640](#ipsc_16_53___resnet_640_)
        - [resize-640-size-80       @ ipsc-16_53/resnet-640](#resize_640_size_80___ipsc_16_53_resnet_64_0_)
            - [on-16_53       @ resize-640-size-80/ipsc-16_53/resnet-640](#on_16_53___resize_640_size_80_ipsc_16_53_resnet_640_)
            - [on-54_126       @ resize-640-size-80/ipsc-16_53/resnet-640](#on_54_126___resize_640_size_80_ipsc_16_53_resnet_640_)
        - [resize-640-size-80-seq-0       @ ipsc-16_53/resnet-640](#resize_640_size_80_seq_0___ipsc_16_53_resnet_64_0_)
            - [on-train       @ resize-640-size-80-seq-0/ipsc-16_53/resnet-640](#on_train___resize_640_size_80_seq_0_ipsc_16_53_resnet_640_)
            - [on-seq-1       @ resize-640-size-80-seq-0/ipsc-16_53/resnet-640](#on_seq_1___resize_640_size_80_seq_0_ipsc_16_53_resnet_640_)
        - [resize-320-size-80       @ ipsc-16_53/resnet-640](#resize_320_size_80___ipsc_16_53_resnet_64_0_)
        - [resize-320-size-160       @ ipsc-16_53/resnet-640](#resize_320_size_160___ipsc_16_53_resnet_64_0_)

<!-- /MarkdownTOC -->
<a id="resnet_64_0_"></a>
# resnet-640 
<a id="ipsc_16_53___resnet_640_"></a>
## ipsc-16_53       @ resnet-640-->p2s_seg
<a id="resize_640_size_80___ipsc_16_53_resnet_64_0_"></a>
### resize-640-size-80       @ ipsc-16_53/resnet-640-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:size-80:smin-0:smax-0:rmin-15:rmax-345:rnum-0:flip-0:resize-640,batch-36,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1
<a id="on_16_53___resize_640_size_80_ipsc_16_53_resnet_640_"></a>
#### on-16_53       @ resize-640-size-80/ipsc-16_53/resnet-640-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ext_reorg_roi_g2-frame-16_53-batch_18,_eval_,ipsc-16_53,batch-36,save-vis-1,dbg-0,dyn-1
<a id="on_54_126___resize_640_size_80_ipsc_16_53_resnet_640_"></a>
#### on-54_126       @ resize-640-size-80/ipsc-16_53/resnet-640-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ext_reorg_roi_g2-frame-16_53-batch_18,_eval_,ipsc-54_126,batch-36,save-vis-1,dbg-0,dyn-1

<a id="resize_640_size_80_seq_0___ipsc_16_53_resnet_64_0_"></a>
### resize-640-size-80-seq-0       @ ipsc-16_53/resnet-640-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,batch-36,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,seg-16_53:size-80:smin-0:smax-0:rmin-15:rmax-345:rnum-0:flip-0:resize-640:seq-0
<a id="on_train___resize_640_size_80_seq_0_ipsc_16_53_resnet_640_"></a>
#### on-train       @ resize-640-size-80-seq-0/ipsc-16_53/resnet-640-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_640-16_53-80_80-80_80-seq_0_0-batch_36,_eval_,batch-18,save-vis-1,dbg-0,dyn-1,seg-16_53:size-80:smin-0:smax-0:rmin-15:rmax-345:rnum-0:flip-0:resize-640:seq-0

<a id="on_seq_1___resize_640_size_80_seq_0_ipsc_16_53_resnet_640_"></a>
#### on-seq-1       @ resize-640-size-80-seq-0/ipsc-16_53/resnet-640-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_640-16_53-80_80-80_80-seq_0_0-batch_36,_eval_,batch-18,save-vis-1,dbg-0,dyn-1,seg-16_53:size-80:smin-0:smax-0:rmin-15:rmax-345:rnum-0:flip-0:resize-640:seq-1

<a id="resize_320_size_80___ipsc_16_53_resnet_64_0_"></a>
### resize-320-size-80       @ ipsc-16_53/resnet-640-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-16_53:size-80:smin-0:smax-0:rmin-15:rmax-345:rnum-0:flip-0:resize-320,batch-24,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1
<a id="resize_320_size_160___ipsc_16_53_resnet_64_0_"></a>
### resize-320-size-160       @ ipsc-16_53/resnet-640-->p2s_seg
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg_160-16_53:size-160:smin-0:smax-0:rmin-15:rmax-345:rnum-0:flip-0:resize-320,batch-8,dbg-0,dyn-1,dist-0,ep-10000,gz,pt-1

