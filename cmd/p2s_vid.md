<!-- MarkdownTOC -->

- [resnet-640](#resnet_64_0_)
    - [mnist-640-1-12_1000       @ resnet-640](#mnist_640_1_12_1000___resnet_640_)
        - [len-2       @ mnist-640-1-12_1000/resnet-640](#len_2___mnist_640_1_12_1000_resnet_640_)
        - [len-3       @ mnist-640-1-12_1000/resnet-640](#len_3___mnist_640_1_12_1000_resnet_640_)
    - [g2_0_4       @ resnet-640](#g2_0_4___resnet_640_)
        - [batch-3       @ g2_0_4/resnet-640](#batch_3___g2_0_4_resnet_64_0_)
            - [on-g2_0_4       @ batch-3/g2_0_4/resnet-640](#on_g2_0_4___batch_3_g2_0_4_resnet_64_0_)
        - [batch-6       @ g2_0_4/resnet-640](#batch_6___g2_0_4_resnet_64_0_)
            - [on-g2_5_9       @ batch-6/g2_0_4/resnet-640](#on_g2_5_9___batch_6_g2_0_4_resnet_64_0_)
            - [on-g2_0_4       @ batch-6/g2_0_4/resnet-640](#on_g2_0_4___batch_6_g2_0_4_resnet_64_0_)
    - [g2_5_9       @ resnet-640](#g2_5_9___resnet_640_)
        - [batch-8       @ g2_5_9/resnet-640](#batch_8___g2_5_9_resnet_64_0_)
            - [on-g2_0_4       @ batch-8/g2_5_9/resnet-640](#on_g2_0_4___batch_8_g2_5_9_resnet_64_0_)
            - [on-g2_5_9       @ batch-8/g2_5_9/resnet-640](#on_g2_5_9___batch_8_g2_5_9_resnet_64_0_)
        - [fg-4       @ g2_5_9/resnet-640](#fg_4___g2_5_9_resnet_64_0_)
    - [g2_0_37       @ resnet-640](#g2_0_37___resnet_640_)
        - [fg-4       @ g2_0_37/resnet-640](#fg_4___g2_0_37_resnet_640_)

<!-- /MarkdownTOC -->

<a id="resnet_64_0_"></a>
# resnet-640 
<a id="mnist_640_1_12_1000___resnet_640_"></a>
## mnist-640-1-12_1000       @ resnet-640-->p2s_vid
<a id="len_2___mnist_640_1_12_1000_resnet_640_"></a>
### len-2       @ mnist-640-1-12_1000/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,mnist-640-1-12_1000,batch-16,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1
<a id="len_3___mnist_640_1_12_1000_resnet_640_"></a>
### len-3       @ mnist-640-1-12_1000/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,mnist-640-1-12_1000,len-3,batch-12,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1

<a id="g2_0_4___resnet_640_"></a>
## g2_0_4       @ resnet-640-->p2s_vid
<a id="batch_3___g2_0_4_resnet_64_0_"></a>
### batch-3       @ g2_0_4/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-0,ipsc-g2_0_4,batch-3,dbg-1,dyn-1,ep-4000
<a id="on_g2_0_4___batch_3_g2_0_4_resnet_64_0_"></a>
#### on-g2_0_4       @ batch-3/g2_0_4/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=eval,vid_det,m-resnet_640_ext_reorg_roi_g2_0_4-length-2-stride-1-batch_3,ipsc-g2_0_4,batch-1,save-vis-1,dbg-1,dyn-1

<a id="batch_6___g2_0_4_resnet_64_0_"></a>
### batch-6       @ g2_0_4/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-0,ipsc-g2_0_4,batch-6,dbg-0,dyn-1,ep-4000,dist-1
<a id="on_g2_5_9___batch_6_g2_0_4_resnet_64_0_"></a>
#### on-g2_5_9       @ batch-6/g2_0_4/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,vid_det,m-resnet_640_ext_reorg_roi_g2_0_4-length-2-stride-1-batch_6,ipsc-g2_5_9,batch-3,save-vis-1,dbg-1,dyn-1
<a id="on_g2_0_4___batch_6_g2_0_4_resnet_64_0_"></a>
#### on-g2_0_4       @ batch-6/g2_0_4/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=eval,vid_det,m-resnet_640_ext_reorg_roi_g2_0_4-length-2-stride-1-batch_6,ipsc-g2_0_4,batch-1,save-vis-1,dbg-1,dyn-1

<a id="g2_5_9___resnet_640_"></a>
## g2_5_9       @ resnet-640-->p2s_vid
<a id="batch_8___g2_5_9_resnet_64_0_"></a>
### batch-8       @ g2_5_9/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-g2_5_9,batch-3,dbg-1,dyn-1,ep-1000000,dist-0,ckpt_ep-20
<a id="on_g2_0_4___batch_8_g2_5_9_resnet_64_0_"></a>
#### on-g2_0_4       @ batch-8/g2_5_9/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,vid_det,m-resnet_640_ext_reorg_roi_g2_5_9-length-2-stride-1-batch_8,ipsc-g2_0_4,batch-4,save-vis-1,dbg-1,dyn-1
**12094**
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,vid_det,m-resnet_640_ext_reorg_roi_g2_5_9-length-2-stride-1-batch_8,ipsc-12094_0_4,batch-4,save-vis-1,dbg-1,dyn-1
**12094_short**
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,vid_det,m-resnet_640_ext_reorg_roi_g2_5_9-length-2-stride-1-batch_8,ipsc-12094_short_0_4,batch-4,save-vis-1,dbg-1,dyn-1
<a id="on_g2_5_9___batch_8_g2_5_9_resnet_64_0_"></a>
#### on-g2_5_9       @ batch-8/g2_5_9/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,vid_det,m-resnet_640_ext_reorg_roi_g2_5_9-length-2-stride-1-batch_8,ipsc-g2_5_9,batch-4,save-vis-1,dbg-1,dyn-1

<a id="fg_4___g2_5_9_resnet_64_0_"></a>
### fg-4       @ g2_5_9/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-g2_5_9,batch-4,dbg-1,dyn-1,ep-10000,dist-1,ckpt_ep-20,fg-4

<a id="g2_0_37___resnet_640_"></a>
## g2_0_37       @ resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-g2_0_37,len-2,strd-1,batch-16,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1
<a id="fg_4___g2_0_37_resnet_640_"></a>
### fg-4       @ g2_0_37/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-g2_0_37,len-2,strd-1,fg-4,batch-16,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1