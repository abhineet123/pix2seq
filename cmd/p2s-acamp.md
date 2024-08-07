<!-- MarkdownTOC -->

- [resnet-640](#resnet_64_0_)
    - [1k8_vid_entire_seq       @ resnet-640](#1k8_vid_entire_seq___resnet_640_)
        - [on-train       @ 1k8_vid_entire_seq/resnet-640](#on_train___1k8_vid_entire_seq_resnet_64_0_)
        - [on-inv       @ 1k8_vid_entire_seq/resnet-640](#on_inv___1k8_vid_entire_seq_resnet_64_0_)
    - [10k6_vid_entire_seq       @ resnet-640](#10k6_vid_entire_seq___resnet_640_)
        - [on-train       @ 10k6_vid_entire_seq/resnet-640](#on_train___10k6_vid_entire_seq_resnet_640_)
        - [on-inv       @ 10k6_vid_entire_seq/resnet-640](#on_inv___10k6_vid_entire_seq_resnet_640_)
    - [20k6_5_video       @ resnet-640](#20k6_5_video___resnet_640_)
        - [on-train       @ 20k6_5_video/resnet-640](#on_train___20k6_5_video_resnet_64_0_)
        - [on-inv       @ 20k6_5_video/resnet-640](#on_inv___20k6_5_video_resnet_64_0_)
    - [1k8_vid_entire_seq-aug       @ resnet-640](#1k8_vid_entire_seq_aug___resnet_640_)
    - [10k6_vid_entire_seq-aug       @ resnet-640](#10k6_vid_entire_seq_aug___resnet_640_)
        - [on-inv       @ 10k6_vid_entire_seq-aug/resnet-640](#on_inv___10k6_vid_entire_seq_aug_resnet_640_)
    - [20k6_5_video-aug       @ resnet-640](#20k6_5_video_aug___resnet_640_)

<!-- /MarkdownTOC -->
<a id="resnet_64_0_"></a>
# resnet-640 

<a id="1k8_vid_entire_seq___resnet_640_"></a>
## 1k8_vid_entire_seq       @ resnet-640-->p2s-acamp
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,acamp-1k8_vid_entire_seq,batch-32,dbg-0,dyn-1,dist-1
<a id="on_train___1k8_vid_entire_seq_resnet_64_0_"></a>
### on-train       @ 1k8_vid_entire_seq/resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_1k8_vid_entire_seq-batch_32,_eval_,acamp-1k8_vid_entire_seq,batch-4,save-vis-1,dbg-0,dyn-1
<a id="on_inv___1k8_vid_entire_seq_resnet_64_0_"></a>
### on-inv       @ 1k8_vid_entire_seq/resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_1k8_vid_entire_seq-batch_32,_eval_,acamp-1k8_vid_entire_seq_inv,batch-8,save-vis-1,dbg-0,dyn-1

<a id="10k6_vid_entire_seq___resnet_640_"></a>
## 10k6_vid_entire_seq       @ resnet-640-->p2s-acamp
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,acamp-10k6_vid_entire_seq,batch-32,dbg-0,dyn-1,dist-1
<a id="on_train___10k6_vid_entire_seq_resnet_640_"></a>
### on-train       @ 10k6_vid_entire_seq/resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_10k6_vid_entire_seq-batch_32,_eval_,acamp-10k6_vid_entire_seq,batch-4,save-vis-1,dbg-0,dyn-1
<a id="on_inv___10k6_vid_entire_seq_resnet_640_"></a>
### on-inv       @ 10k6_vid_entire_seq/resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_10k6_vid_entire_seq-batch_32,_eval_,acamp-10k6_vid_entire_seq_inv,batch-8,save-vis-1,dbg-0,dyn-1

<a id="20k6_5_video___resnet_640_"></a>
## 20k6_5_video       @ resnet-640-->p2s-acamp
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,acamp-20k6_5_video,batch-32,dbg-0,dyn-1,dist-1
<a id="on_train___20k6_5_video_resnet_64_0_"></a>
### on-train       @ 20k6_5_video/resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_20k6_5_video-batch_32,_eval_,acamp-20k6_5_video,batch-4,save-vis-1,dbg-0,dyn-1
<a id="on_inv___20k6_5_video_resnet_64_0_"></a>
### on-inv       @ 20k6_5_video/resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_20k6_5_video-batch_32,_eval_,acamp-20k6_5_video_inv,batch-12,save-vis-1,dbg-0,dyn-1

<a id="1k8_vid_entire_seq_aug___resnet_640_"></a>
## 1k8_vid_entire_seq-aug       @ resnet-640-->p2s-acamp
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,acamp-1k8_vid_entire_seq,batch-32,dbg-0,dyn-1,dist-1,jtr,res-1280

<a id="10k6_vid_entire_seq_aug___resnet_640_"></a>
## 10k6_vid_entire_seq-aug       @ resnet-640-->p2s-acamp
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,acamp-10k6_vid_entire_seq,batch-32,dbg-0,dyn-1,dist-1,jtr,res-1280
<a id="on_inv___10k6_vid_entire_seq_aug_resnet_640_"></a>
### on-inv       @ 10k6_vid_entire_seq-aug/resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_10k6_vid_entire_seq-batch_32,_eval_,acamp-10k6_vid_entire_seq_inv,batch-8,save-vis-1,dbg-0,dyn-1

<a id="20k6_5_video_aug___resnet_640_"></a>
## 20k6_5_video-aug       @ resnet-640-->p2s-acamp
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,acamp-20k6_5_video,batch-32,dbg-0,dyn-1,dist-1,jtr,res-1280
