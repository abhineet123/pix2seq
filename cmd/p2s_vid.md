<!-- MarkdownTOC -->

- [swin-t](#swin_t_)
    - [mnist-640-1-12_1000       @ swin-t](#mnist_640_1_12_1000___swin_t_)
    - [len-2       @ swin-t](#len_2___swin_t_)
        - [len-3       @ len-2/swin-t](#len_3___len_2_swin_t_)
    - [mnist-640-5-12_1000       @ swin-t](#mnist_640_5_12_1000___swin_t_)
    - [len-2       @ swin-t](#len_2___swin_t__1)
    - [len-3       @ swin-t](#len_3___swin_t_)
    - [len-4       @ swin-t](#len_4___swin_t_)
- [swin-s](#swin_s_)
    - [mnist-640-1-12_1000       @ swin-s](#mnist_640_1_12_1000___swin_s_)
        - [len-2       @ mnist-640-1-12_1000/swin-s](#len_2___mnist_640_1_12_1000_swin_s_)
- [resnet-640](#resnet_64_0_)
    - [detrac-non_empty       @ resnet-640](#detrac_non_empty___resnet_640_)
        - [0_19       @ detrac-non_empty/resnet-640](#0_19___detrac_non_empty_resnet_64_0_)
        - [0_9       @ detrac-non_empty/resnet-640](#0_9___detrac_non_empty_resnet_64_0_)
            - [on-train       @ 0_9/detrac-non_empty/resnet-640](#on_train___0_9_detrac_non_empty_resnet_64_0_)
            - [on-test       @ 0_9/detrac-non_empty/resnet-640](#on_test___0_9_detrac_non_empty_resnet_64_0_)
            - [on-test-strd-2       @ 0_9/detrac-non_empty/resnet-640](#on_test_strd_2___0_9_detrac_non_empty_resnet_64_0_)
    - [mnist-640-1-12_1000       @ resnet-640](#mnist_640_1_12_1000___resnet_640_)
        - [len-2       @ mnist-640-1-12_1000/resnet-640](#len_2___mnist_640_1_12_1000_resnet_640_)
            - [on-train       @ len-2/mnist-640-1-12_1000/resnet-640](#on_train___len_2_mnist_640_1_12_1000_resnet_640_)
            - [on-test       @ len-2/mnist-640-1-12_1000/resnet-640](#on_test___len_2_mnist_640_1_12_1000_resnet_640_)
        - [len-3       @ mnist-640-1-12_1000/resnet-640](#len_3___mnist_640_1_12_1000_resnet_640_)
            - [swin-t       @ len-3/mnist-640-1-12_1000/resnet-640](#swin_t___len_3_mnist_640_1_12_1000_resnet_640_)
        - [len-9       @ mnist-640-1-12_1000/resnet-640](#len_9___mnist_640_1_12_1000_resnet_640_)
    - [mnist-640-5-12_1000       @ resnet-640](#mnist_640_5_12_1000___resnet_640_)
        - [len-2       @ mnist-640-5-12_1000/resnet-640](#len_2___mnist_640_5_12_1000_resnet_640_)
            - [on-train       @ len-2/mnist-640-5-12_1000/resnet-640](#on_train___len_2_mnist_640_5_12_1000_resnet_640_)
            - [on-test       @ len-2/mnist-640-5-12_1000/resnet-640](#on_test___len_2_mnist_640_5_12_1000_resnet_640_)
        - [len-9       @ mnist-640-5-12_1000/resnet-640](#len_9___mnist_640_5_12_1000_resnet_640_)
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
<a id="swin_t_"></a>
# swin-t 
<a id="mnist_640_1_12_1000___swin_t_"></a>
## mnist-640-1-12_1000       @ swin-t-->p2s_vid
<a id="len_2___swin_t_"></a>
## len-2       @ swin-t-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,mnist-640-1-12_1000,batch-4,dbg-0,dyn-1,dist-0,ep-10000,ckpt_ep-1,swin-t
<a id="len_3___len_2_swin_t_"></a>
### len-3       @ len-2/swin-t-->p2s_vid
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,mnist-640-1-12_1000,len-3,batch-4,dbg-1,dyn-1,dist-0,ep-10000,ckpt_ep-1,swin-t

<a id="mnist_640_5_12_1000___swin_t_"></a>
## mnist-640-5-12_1000       @ swin-t-->p2s_vid
<a id="len_2___swin_t__1"></a>
## len-2       @ swin-t-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,mnist-640-5-12_1000,batch-8,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1,swin-t
<a id="len_3___swin_t_"></a>
## len-3       @ swin-t-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,mnist-640-5-12_1000,batch-3,dbg-1,dyn-1,dist-0,ep-10000,ckpt_ep-1,swin-t
<a id="len_4___swin_t_"></a>
## len-4       @ swin-t-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,mnist-640-5-12_1000,batch-3,dbg-1,dyn-1,dist-0,ep-10000,ckpt_ep-1,swin-t

<a id="swin_s_"></a>
# swin-s 
<a id="mnist_640_1_12_1000___swin_s_"></a>
## mnist-640-1-12_1000       @ swin-s-->p2s_vid
<a id="len_2___mnist_640_1_12_1000_swin_s_"></a>
### len-2       @ mnist-640-1-12_1000/swin-s-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-0,mnist-640-1-12_1000,batch-3,dbg-1,dyn-1,ep-10000,ckpt_ep-1,swin-s

<a id="resnet_64_0_"></a>
# resnet-640 
<a id="detrac_non_empty___resnet_640_"></a>
## detrac-non_empty       @ resnet-640-->p2s_vid
<a id="0_19___detrac_non_empty_resnet_64_0_"></a>
### 0_19       @ detrac-non_empty/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,detrac-non_empty-0_19,batch-18,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1
<a id="0_9___detrac_non_empty_resnet_64_0_"></a>
### 0_9       @ detrac-non_empty/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,detrac-non_empty-0_9,batch-18,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1
<a id="on_train___0_9_detrac_non_empty_resnet_64_0_"></a>
#### on-train       @ 0_9/detrac-non_empty/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_9-batch_18,detrac-non_empty-0_9,batch-8,save-vis-1,dbg-0,dyn-1,dist-0
`strd-2`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_9-batch_18,detrac-non_empty-0_9,strd-2,batch-48,save-vis-1,dbg-0,dyn-1,dist-0

<a id="on_test___0_9_detrac_non_empty_resnet_64_0_"></a>
#### on-test       @ 0_9/detrac-non_empty/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_9-batch_18,detrac-non_empty-49_68,batch-48,save-vis-1,dbg-0,dyn-1,dist-0
<a id="on_test_strd_2___0_9_detrac_non_empty_resnet_64_0_"></a>
#### on-test-strd-2       @ 0_9/detrac-non_empty/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-2-non_empty-seq-0_9-batch_18,detrac-non_empty-49_68,batch-48,save-vis-1,dbg-0,dyn-1,dist-0

<a id="mnist_640_1_12_1000___resnet_640_"></a>
## mnist-640-1-12_1000       @ resnet-640-->p2s_vid
<a id="len_2___mnist_640_1_12_1000_resnet_640_"></a>
### len-2       @ mnist-640-1-12_1000/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-0,mnist-640-1-12_1000,batch-3,dbg-1,dyn-1,ep-10000,ckpt_ep-1
<a id="on_train___len_2_mnist_640_1_12_1000_resnet_640_"></a>
#### on-train       @ len-2/mnist-640-1-12_1000/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=eval,vid_det,m-resnet_640_mnist_640_1_12_1000_var-length-2-stride-1-train-batch_18,mnist-640-1-12_1000,batch-1,save-vis-1,dbg-1,dyn-1,dist-0,suffix-train
<a id="on_test___len_2_mnist_640_1_12_1000_resnet_640_"></a>
#### on-test       @ len-2/mnist-640-1-12_1000/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=eval,vid_det,m-resnet_640_mnist_640_1_12_1000_var-length-2-stride-1-train-batch_18,mnist-640-1-12_1000,batch-48,save-vis-1,dbg-1,dyn-1,dist-0,suffix-test
`strd-2`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=eval,vid_det,m-resnet_640_mnist_640_1_12_1000_var-length-2-stride-1-train-batch_18,mnist-640-1-12_1000,strd-2,batch-48,save-vis-1,dbg-0,dyn-1,dist-0,suffix-test


<a id="len_3___mnist_640_1_12_1000_resnet_640_"></a>
### len-3       @ mnist-640-1-12_1000/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,mnist-640-1-12_1000,len-3,batch-12,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1
<a id="swin_t___len_3_mnist_640_1_12_1000_resnet_640_"></a>
#### swin-t       @ len-3/mnist-640-1-12_1000/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,mnist-640-1-12_1000,len-3,batch-4,dbg-1,dyn-1,dist-0,ep-10000,ckpt_ep-1,swin-t


<a id="len_9___mnist_640_1_12_1000_resnet_640_"></a>
### len-9       @ mnist-640-1-12_1000/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,mnist-640-1-12_1000,len-9,batch-4,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1

<a id="mnist_640_5_12_1000___resnet_640_"></a>
## mnist-640-5-12_1000       @ resnet-640-->p2s_vid
<a id="len_2___mnist_640_5_12_1000_resnet_640_"></a>
### len-2       @ mnist-640-5-12_1000/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,mnist-640-5-12_1000,batch-18,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1
<a id="on_train___len_2_mnist_640_5_12_1000_resnet_640_"></a>
#### on-train       @ len-2/mnist-640-5-12_1000/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=eval,vid_det,m-resnet_640_mnist_640_5_12_1000_var-length-2-stride-1-train-batch_18,mnist-640-5-12_1000,batch-36,save-vis-1,dbg-0,dyn-1,dist-0,suffix-train
<a id="on_test___len_2_mnist_640_5_12_1000_resnet_640_"></a>
#### on-test       @ len-2/mnist-640-5-12_1000/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=eval,vid_det,m-resnet_640_mnist_640_5_12_1000_var-length-2-stride-1-train-batch_18,mnist-640-5-12_1000,batch-36,save-vis-1,dbg-0,dyn-1,dist-0,suffix-test
`strd-2`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=eval,vid_det,m-resnet_640_mnist_640_5_12_1000_var-length-2-stride-1-train-batch_18,mnist-640-5-12_1000,strd-2,batch-48,save-vis-1,dbg-0,dyn-1,dist-0,suffix-test

<a id="len_9___mnist_640_5_12_1000_resnet_640_"></a>
### len-9       @ mnist-640-5-12_1000/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,mnist-640-5-12_1000,len-9,batch-4,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1

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