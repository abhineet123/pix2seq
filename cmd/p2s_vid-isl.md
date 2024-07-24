<!-- MarkdownTOC -->

- [swin](#swi_n_)
    - [gram-0_1       @ swin](#gram_0_1___swin_)
        - [len-2       @ gram-0_1/swin](#len_2___gram_0_1_swi_n_)
- [lfn](#lfn_)
    - [gram-0_1       @ lfn](#gram_0_1___lf_n_)
        - [len-9       @ gram-0_1/lfn](#len_9___gram_0_1_lfn_)
            - [on-train       @ len-9/gram-0_1/lfn](#on_train___len_9_gram_0_1_lfn_)
        - [len-14-0_2000       @ gram-0_1/lfn](#len_14_0_2000___gram_0_1_lfn_)
            - [on-3000_5000       @ len-14-0_2000/gram-0_1/lfn](#on_3000_5000___len_14_0_2000_gram_0_1_lfn_)
        - [len-16-0_2000       @ gram-0_1/lfn](#len_16_0_2000___gram_0_1_lfn_)
    - [detrac-non_empty       @ lfn](#detrac_non_empty___lf_n_)
        - [0_19-jtr       @ detrac-non_empty/lfn](#0_19_jtr___detrac_non_empty_lfn_)
            - [on-train       @ 0_19-jtr/detrac-non_empty/lfn](#on_train___0_19_jtr_detrac_non_empty_lf_n_)
                - [strd-2       @ on-train/0_19-jtr/detrac-non_empty/lfn](#strd_2___on_train_0_19_jtr_detrac_non_empty_lfn_)
        - [0_19-len-9       @ detrac-non_empty/lfn](#0_19_len_9___detrac_non_empty_lfn_)
- [mid](#mid_)
    - [detrac-0_19       @ mid](#detrac_0_19___mi_d_)
        - [on-train       @ detrac-0_19/mid](#on_train___detrac_0_19_mi_d_)
            - [strd-1       @ on-train/detrac-0_19/mid](#strd_1___on_train_detrac_0_19_mid_)
            - [strd-2       @ on-train/detrac-0_19/mid](#strd_2___on_train_detrac_0_19_mid_)
        - [on-49_68       @ detrac-0_19/mid](#on_49_68___detrac_0_19_mi_d_)
            - [strd-1       @ on-49_68/detrac-0_19/mid](#strd_1___on_49_68_detrac_0_19_mid_)
            - [strd-2       @ on-49_68/detrac-0_19/mid](#strd_2___on_49_68_detrac_0_19_mid_)
    - [detrac-0_9       @ mid](#detrac_0_9___mi_d_)
        - [on-train       @ detrac-0_9/mid](#on_train___detrac_0_9_mid_)
            - [strd-2       @ on-train/detrac-0_9/mid](#strd_2___on_train_detrac_0_9_mi_d_)
        - [on-49_68       @ detrac-0_9/mid](#on_49_68___detrac_0_9_mid_)
            - [strd-2       @ on-49_68/detrac-0_9/mid](#strd_2___on_49_68_detrac_0_9_mi_d_)

<!-- /MarkdownTOC -->
<a id="swi_n_"></a>
# swin
<a id="gram_0_1___swin_"></a>
## gram-0_1       @ swin-->p2s_vid-isl
<a id="len_2___gram_0_1_swi_n_"></a>
### len-2       @ gram-0_1/swin-->p2s_vid-isl
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,gram-0_1,len-2,batch-3,dbg-1,dyn-1,dist-0,swin-t

<a id="lfn_"></a>
# lfn 
<a id="gram_0_1___lf_n_"></a>
## gram-0_1       @ lfn-->p2s_vid-isl
<a id="len_9___gram_0_1_lfn_"></a>
### len-9       @ gram-0_1/lfn-->p2s_vid-isl
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,gram-0_1,len-9,batch-2,dbg-0,dyn-1,dist-0,lfn
<a id="on_train___len_9_gram_0_1_lfn_"></a>
#### on-train       @ len-9/gram-0_1/lfn-->p2s_vid-isl
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=vid_det,m-resnet_640_gram_only-length-9-stride-1-seq-0_1-batch_2-lfn,_eval_,gram-0_1,len-9,vstrd-9,batch-2,save-vis-1,dbg-0,dyn-1,dist-0

<a id="len_14_0_2000___gram_0_1_lfn_"></a>
### len-14-0_2000       @ gram-0_1/lfn-->p2s_vid-isl
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,gram-0_1-0_2000,len-14,batch-2,dbg-0,dyn-1,dist-1,lfn
<a id="on_3000_5000___len_14_0_2000_gram_0_1_lfn_"></a>
#### on-3000_5000       @ len-14-0_2000/gram-0_1/lfn-->p2s_vid-isl
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=vid_det,m-resnet_640_gram_only-0_2000-length-14-stride-1-seq-0_1-batch_2-lfn,_eval_,gram-0_1-3000_5000,len-14,vstrd-14,batch-1,save-vis-1,dbg-0,dyn-1,dist-0

<a id="len_16_0_2000___gram_0_1_lfn_"></a>
### len-16-0_2000       @ gram-0_1/lfn-->p2s_vid-isl
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,gram-0_1-0_2000,len-16,batch-2,dbg-0,dyn-1,dist-1,lfn

<a id="detrac_non_empty___lf_n_"></a>
## detrac-non_empty       @ lfn-->p2s_vid-isl
<a id="0_19_jtr___detrac_non_empty_lfn_"></a>
### 0_19-jtr       @ detrac-non_empty/lfn-->p2s_vid-isl
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,detrac-non_empty-0_19,len-6,batch-6,dbg-0,dyn-1,dist-1,lfn,jtr
<a id="on_train___0_19_jtr_detrac_non_empty_lf_n_"></a>
#### on-train       @ 0_19-jtr/detrac-non_empty/lfn-->p2s_vid-isl
<a id="strd_2___on_train_0_19_jtr_detrac_non_empty_lfn_"></a>
##### strd-2       @ on-train/0_19-jtr/detrac-non_empty/lfn-->p2s_vid-isl
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_19-batch_18,detrac-non_empty-0_19,vstrd-2,batch-12,save-vis-1,dbg-0,dyn-1,dist-0

<a id="0_19_len_9___detrac_non_empty_lfn_"></a>
### 0_19-len-9       @ detrac-non_empty/lfn-->p2s_vid-isl
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,detrac-non_empty-0_19,len-9,batch-2,dbg-0,dyn-1,dist-1,lfn

<a id="mid_"></a>
# mid
<a id="detrac_0_19___mi_d_"></a>
## detrac-0_19       @ mid-->p2s_vid-isl
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,detrac-non_empty-0_19,batch-3,dbg-0,dyn-1,dist-1
<a id="on_train___detrac_0_19_mi_d_"></a>
### on-train       @ detrac-0_19/mid-->p2s_vid-isl
<a id="strd_1___on_train_detrac_0_19_mid_"></a>
#### strd-1       @ on-train/detrac-0_19/mid-->p2s_vid-isl
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_19-batch_18,detrac-non_empty-0_19,batch-48,save-vis-1,dbg-0,dyn-1,dist-0
<a id="strd_2___on_train_detrac_0_19_mid_"></a>
#### strd-2       @ on-train/detrac-0_19/mid-->p2s_vid-isl
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_19-batch_18,detrac-non_empty-0_19,vstrd-2,batch-12,save-vis-1,dbg-0,dyn-1,dist-0
<a id="on_49_68___detrac_0_19_mi_d_"></a>
### on-49_68       @ detrac-0_19/mid-->p2s_vid-isl
<a id="strd_1___on_49_68_detrac_0_19_mid_"></a>
#### strd-1       @ on-49_68/detrac-0_19/mid-->p2s_vid-isl
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_19-batch_18,detrac-non_empty-49_68,batch-48,save-vis-1,dbg-0,dyn-1,dist-0
<a id="strd_2___on_49_68_detrac_0_19_mid_"></a>
#### strd-2       @ on-49_68/detrac-0_19/mid-->p2s_vid-isl
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_19-batch_18,eval,vid_det,detrac-non_empty-49_68,vstrd-2,batch-12,save-vis-1,dbg-0,dyn-1,dist-0

<a id="detrac_0_9___mi_d_"></a>
## detrac-0_9       @ mid-->p2s_vid-isl
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,detrac-non_empty-0_9,batch-18,dbg-0,dyn-1,dist-1
<a id="on_train___detrac_0_9_mid_"></a>
### on-train       @ detrac-0_9/mid-->p2s_vid-isl
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_9-batch_18,detrac-non_empty-0_9,batch-48,save-vis-1,dbg-0,dyn-1,dist-0
<a id="strd_2___on_train_detrac_0_9_mi_d_"></a>
#### strd-2       @ on-train/detrac-0_9/mid-->p2s_vid-isl
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_9-batch_18,detrac-non_empty-0_9,vstrd-2,batch-48,save-vis-1,dbg-0,dyn-1,dist-0
<a id="on_49_68___detrac_0_9_mid_"></a>
### on-49_68       @ detrac-0_9/mid-->p2s_vid-isl
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_9-batch_18,detrac-non_empty-49_68,batch-1,save-vis-1,dbg-0,dyn-1,dist-0
<a id="strd_2___on_49_68_detrac_0_9_mi_d_"></a>
#### strd-2       @ on-49_68/detrac-0_9/mid-->p2s_vid-isl
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_9-batch_18,eval,vid_det,detrac-non_empty-49_68,vstrd-2,batch-48,save-vis-1,dbg-0,dyn-1,dist-0


