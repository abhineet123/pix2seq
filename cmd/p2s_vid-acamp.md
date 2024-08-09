<!-- MarkdownTOC -->

- [mid       @ resnet-640](#mid___resnet_640_)
    - [1k8_vid_entire_seq       @ mid](#1k8_vid_entire_seq___mi_d_)
        - [on-train       @ 1k8_vid_entire_seq/mid](#on_train___1k8_vid_entire_seq_mid_)
        - [on-inv       @ 1k8_vid_entire_seq/mid](#on_inv___1k8_vid_entire_seq_mid_)
    - [10k6_vid_entire_seq       @ mid](#10k6_vid_entire_seq___mi_d_)
        - [on-train       @ 10k6_vid_entire_seq/mid](#on_train___10k6_vid_entire_seq_mi_d_)
        - [on-inv       @ 10k6_vid_entire_seq/mid](#on_inv___10k6_vid_entire_seq_mi_d_)
    - [1k8_vid_entire_seq-aug       @ mid](#1k8_vid_entire_seq_aug___mi_d_)
        - [on-train       @ 1k8_vid_entire_seq-aug/mid](#on_train___1k8_vid_entire_seq_aug_mid_)
        - [on-inv       @ 1k8_vid_entire_seq-aug/mid](#on_inv___1k8_vid_entire_seq_aug_mid_)
    - [10k6_vid_entire_seq-aug       @ mid](#10k6_vid_entire_seq_aug___mi_d_)
        - [on-train       @ 10k6_vid_entire_seq-aug/mid](#on_train___10k6_vid_entire_seq_aug_mi_d_)
        - [on-inv       @ 10k6_vid_entire_seq-aug/mid](#on_inv___10k6_vid_entire_seq_aug_mi_d_)
    - [20k6_5_video-aug       @ mid](#20k6_5_video_aug___mi_d_)
        - [on-train       @ 20k6_5_video-aug/mid](#on_train___20k6_5_video_aug_mid_)
        - [on-inv       @ 20k6_5_video-aug/mid](#on_inv___20k6_5_video_aug_mid_)

<!-- /MarkdownTOC -->

<a id="mid___resnet_640_"></a>
# mid       @ resnet-640-->p2s_vid
<a id="1k8_vid_entire_seq___mi_d_"></a>
## 1k8_vid_entire_seq       @ mid-->p2s_vid-acamp
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,acamp-1k8_vid_entire_seq,batch-18,dbg-0,dyn-1,dist-1
<a id="on_train___1k8_vid_entire_seq_mid_"></a>
### on-train       @ 1k8_vid_entire_seq/mid-->p2s_vid-acamp
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_1k8_vid_entire_seq-length-2-stride-1-batch_18,_eval_,vid_det,acamp-1k8_vid_entire_seq,vstrd-1,batch-2,save-vis-1,dbg-0,dyn-1
<a id="on_inv___1k8_vid_entire_seq_mid_"></a>
### on-inv       @ 1k8_vid_entire_seq/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_1k8_vid_entire_seq-length-2-stride-1-batch_18,_eval_,vid_det,acamp-1k8_vid_entire_seq_inv,vstrd-1,batch-8,save-vis-1,dbg-0,dyn-1

<a id="10k6_vid_entire_seq___mi_d_"></a>
## 10k6_vid_entire_seq       @ mid-->p2s_vid-acamp
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,acamp-10k6_vid_entire_seq,batch-18,dbg-0,dyn-1,dist-1
<a id="on_train___10k6_vid_entire_seq_mi_d_"></a>
### on-train       @ 10k6_vid_entire_seq/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_10k610k6_vid_entire_seq_vid_entire_seq-length-2-stride-1-batch_18,_eval_,vid_det,acamp-10k6_vid_entire_seq,vstrd-1,batch-3,save-vis-1,dbg-0,dyn-1
<a id="on_inv___10k6_vid_entire_seq_mi_d_"></a>
### on-inv       @ 10k6_vid_entire_seq/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_10k6_vid_entire_seq-length-2-stride-1-batch_18,_eval_,vid_det,acamp-10k6_vid_entire_seq_inv,vstrd-1,batch-6,save-vis-1,dbg-0,dyn-1


<a id="1k8_vid_entire_seq_aug___mi_d_"></a>
## 1k8_vid_entire_seq-aug       @ mid-->p2s_vid-acamp
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,acamp-1k8_vid_entire_seq,batch-18,dbg-0,dyn-1,dist-1,jtr,res-1280
<a id="on_train___1k8_vid_entire_seq_aug_mid_"></a>
### on-train       @ 1k8_vid_entire_seq-aug/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_1k8_vid_entire_seq-length-2-stride-1-batch_18-jtr-res_1280,_eval_,vid_det,acamp-1k8_vid_entire_seq,vstrd-1,batch-1,save-vis-1,dbg-0,dyn-1
<a id="on_inv___1k8_vid_entire_seq_aug_mid_"></a>
### on-inv       @ 1k8_vid_entire_seq-aug/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_1k8_vid_entire_seq-length-2-stride-1-batch_18-jtr-res_1280,_eval_,vid_det,acamp-1k8_vid_entire_seq_inv,vstrd-1,batch-12,save-vis-1,dbg-0,dyn-1

<a id="10k6_vid_entire_seq_aug___mi_d_"></a>
## 10k6_vid_entire_seq-aug       @ mid-->p2s_vid-acamp
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,acamp-10k6_vid_entire_seq,batch-18,dbg-0,dyn-1,dist-1,jtr,res-1280
`2_per_seq_dbg_bear`
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,acamp-2_per_seq_dbg_bear,batch-1,dbg-1,dyn-1,dist-0,jtr,res-1280
<a id="on_train___10k6_vid_entire_seq_aug_mi_d_"></a>
### on-train       @ 10k6_vid_entire_seq-aug/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_10k6_vid_entire_seq-length-2-stride-1-batch_18-jtr-res_1280,_eval_,vid_det,acamp-10k6_vid_entire_seq,vstrd-1,batch-32,save-vis-1,dbg-0,dyn-1
<a id="on_inv___10k6_vid_entire_seq_aug_mi_d_"></a>
### on-inv       @ 10k6_vid_entire_seq-aug/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_10k6_vid_entire_seq-length-2-stride-1-batch_18-jtr-res_1280,_eval_,vid_det,acamp-10k6_vid_entire_seq_inv,vstrd-1,batch-12,save-vis-1,dbg-0,dyn-1

<a id="20k6_5_video_aug___mi_d_"></a>
## 20k6_5_video-aug       @ mid-->p2s_vid-acamp
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,acamp-20k6_5_video,batch-18,dbg-0,dyn-1,dist-1,jtr,res-1280
<a id="on_train___20k6_5_video_aug_mid_"></a>
### on-train       @ 20k6_5_video-aug/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_20k6_5_video-length-2-stride-1-batch_18-jtr-res_1280,_eval_,vid_det,acamp-20k6_5_video,vstrd-1,batch-4,save-vis-1,dbg-0,dyn-1
<a id="on_inv___20k6_5_video_aug_mid_"></a>
### on-inv       @ 20k6_5_video-aug/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_20k6_5_video-length-2-stride-1-batch_18-jtr-res_1280,_eval_,vid_det,acamp-20k6_5_video_inv,vstrd-1,batch-20,save-vis-1,dbg-0,dyn-1


