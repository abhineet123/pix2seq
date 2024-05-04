<!-- MarkdownTOC -->

- [resnet-640](#resnet_64_0_)
    - [ipsc-16_53       @ resnet-640](#ipsc_16_53___resnet_640_)
        - [on-16_53       @ ipsc-16_53/resnet-640](#on_16_53___ipsc_16_53_resnet_64_0_)
        - [on-54_126       @ ipsc-16_53/resnet-640](#on_54_126___ipsc_16_53_resnet_64_0_)
    - [ipsc-16_53-seq-0       @ resnet-640](#ipsc_16_53_seq_0___resnet_640_)

<!-- /MarkdownTOC -->
<a id="resnet_64_0_"></a>
# resnet-640 
<a id="ipsc_16_53___resnet_640_"></a>
## ipsc-16_53       @ resnet-640-->p2s_seg
python3 run.py --cfg=configs/config_sem_seg.py  --j5=train,resnet-640,seg-16_53:size-80:smin-0:smax-0:rmin-15:rmax-345:rnum-0:flip-0:resize-640,batch-36,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1
<a id="on_16_53___ipsc_16_53_resnet_64_0_"></a>
### on-16_53       @ ipsc-16_53/resnet-640-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_sem_seg.py  --j5=m-resnet_640_ext_reorg_roi_g2-frame-16_53-batch_18,_eval_,ipsc-16_53,batch-36,save-vis-1,dbg-0,dyn-1
<a id="on_54_126___ipsc_16_53_resnet_64_0_"></a>
### on-54_126       @ ipsc-16_53/resnet-640-->p2s_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_sem_seg.py  --j5=m-resnet_640_ext_reorg_roi_g2-frame-16_53-batch_18,_eval_,ipsc-54_126,batch-36,save-vis-1,dbg-0,dyn-1

<a id="ipsc_16_53_seq_0___resnet_640_"></a>
## ipsc-16_53-seq-0       @ resnet-640-->p2s_seg
python3 run.py --cfg=configs/config_sem_seg.py  --j5=train,resnet-640,batch-36,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,seg-16_53:size-80:smin-0:smax-0:rmin-15:rmax-345:rnum-0:flip-0:resize-640:seq-0

