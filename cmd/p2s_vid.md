<!-- MarkdownTOC -->

- [resnet-640](#resnet_64_0_)
    - [g2_0_4       @ resnet-640](#g2_0_4___resnet_640_)
        - [batch-3       @ g2_0_4/resnet-640](#batch_3___g2_0_4_resnet_64_0_)
        - [batch-6       @ g2_0_4/resnet-640](#batch_6___g2_0_4_resnet_64_0_)
            - [on-g2_5_9       @ batch-6/g2_0_4/resnet-640](#on_g2_5_9___batch_6_g2_0_4_resnet_64_0_)
            - [on-g2_0_4       @ batch-6/g2_0_4/resnet-640](#on_g2_0_4___batch_6_g2_0_4_resnet_64_0_)

<!-- /MarkdownTOC -->

<a id="resnet_64_0_"></a>
# resnet-640 
<a id="g2_0_4___resnet_640_"></a>
## g2_0_4       @ resnet-640-->p2s_vid
<a id="batch_3___g2_0_4_resnet_64_0_"></a>
### batch-3       @ g2_0_4/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det_ipsc.py --j5=train,resnet-640,vid_det,pt-0,ipsc-g2_0_4,batch-3,dbg-1,dyn-1,ep-4000
<a id="batch_6___g2_0_4_resnet_64_0_"></a>
### batch-6       @ g2_0_4/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det_ipsc.py --j5=train,resnet-640,vid_det,pt-0,ipsc-g2_0_4,batch-6,dbg-0,dyn-1,ep-4000,dist-1
<a id="on_g2_5_9___batch_6_g2_0_4_resnet_64_0_"></a>
#### on-g2_5_9       @ batch-6/g2_0_4/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,vid_det,m-resnet_640_ext_reorg_roi_g2_0_4-length-2-stride-1-batch_6,ipsc-g2_5_9,batch-3,save-vis-1,dbg-1,dyn-1
<a id="on_g2_0_4___batch_6_g2_0_4_resnet_64_0_"></a>
#### on-g2_0_4       @ batch-6/g2_0_4/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=eval,vid_det,m-resnet_640_ext_reorg_roi_g2_0_4-length-2-stride-1-batch_6,ipsc-g2_0_4,batch-1,save-vis-1,dbg-1,dyn-1