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
    - [detrac-0_19       @ lfn](#detrac_0_19___lf_n_)
        - [len-6-aug       @ detrac-0_19/lfn](#len_6_aug___detrac_0_19_lf_n_)
            - [on-train       @ len-6-aug/detrac-0_19/lfn](#on_train___len_6_aug_detrac_0_19_lf_n_)
            - [on-49_68       @ len-6-aug/detrac-0_19/lfn](#on_49_68___len_6_aug_detrac_0_19_lf_n_)
        - [len-9       @ detrac-0_19/lfn](#len_9___detrac_0_19_lf_n_)
            - [on-49_68       @ len-9/detrac-0_19/lfn](#on_49_68___len_9_detrac_0_19_lf_n_)
- [mid](#mid_)
    - [detrac-0_19       @ mid](#detrac_0_19___mi_d_)
        - [on-train       @ detrac-0_19/mid](#on_train___detrac_0_19_mi_d_)
        - [on-49_68       @ detrac-0_19/mid](#on_49_68___detrac_0_19_mi_d_)
    - [detrac-0_9       @ mid](#detrac_0_9___mi_d_)
        - [on-train       @ detrac-0_9/mid](#on_train___detrac_0_9_mid_)
        - [on-49_68       @ detrac-0_9/mid](#on_49_68___detrac_0_9_mid_)
    - [detrac-0_48-len-32       @ mid](#detrac_0_48_len_32___mi_d_)
        - [on-49_85       @ detrac-0_48-len-32/mid](#on_49_85___detrac_0_48_len_32_mid_)
    - [detrac-0_48-len-40       @ mid](#detrac_0_48_len_40___mi_d_)
    - [detrac-0_48-len-48       @ mid](#detrac_0_48_len_48___mi_d_)
    - [detrac-0_48-len-56       @ mid](#detrac_0_48_len_56___mi_d_)

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
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=vid_det,m-resnet_640_gram_only-length-9-stride-1-seq-0_1-batch_2-lfn,_eval_,gram-0_1,len-9,vstrd-9,batch-2,save-vis-0,dbg-0,dyn-1,dist-0

<a id="len_14_0_2000___gram_0_1_lfn_"></a>
### len-14-0_2000       @ gram-0_1/lfn-->p2s_vid-isl
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,gram-0_1-0_2000,len-14,batch-2,dbg-0,dyn-1,dist-1,lfn
<a id="on_3000_5000___len_14_0_2000_gram_0_1_lfn_"></a>
#### on-3000_5000       @ len-14-0_2000/gram-0_1/lfn-->p2s_vid-isl
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=vid_det,m-resnet_640_gram_only-0_2000-length-14-stride-1-seq-0_1-batch_2-lfn,_eval_,gram-0_1-3000_5000,len-14,vstrd-14,batch-1,save-vis-0,dbg-0,dyn-1,dist-0

<a id="len_16_0_2000___gram_0_1_lfn_"></a>
### len-16-0_2000       @ gram-0_1/lfn-->p2s_vid-isl
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,gram-0_1-0_2000,len-16,batch-2,dbg-0,dyn-1,dist-1,lfn

<a id="detrac_0_19___lf_n_"></a>
## detrac-0_19       @ lfn-->p2s_vid-isl
<a id="len_6_aug___detrac_0_19_lf_n_"></a>
### len-6-aug       @ detrac-0_19/lfn-->p2s_vid-isl
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,detrac-non_empty-0_19,len-6,batch-6,dbg-0,dyn-1,dist-1,lfn,jtr
<a id="on_train___len_6_aug_detrac_0_19_lf_n_"></a>
#### on-train       @ len-6-aug/detrac-0_19/lfn-->p2s_vid-isl
`strd-2`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_19-batch_18,_eval_,vid_det,detrac-non_empty-0_19,vstrd-2,batch-12,save-vis-0,dbg-0,dyn-1,dist-0
<a id="on_49_68___len_6_aug_detrac_0_19_lf_n_"></a>
#### on-49_68       @ len-6-aug/detrac-0_19/lfn-->p2s_vid-isl
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=m-resnet_640_detrac-length-6-stride-1-non_empty-seq-0_19-batch_6-lfn-jtr,_eval_,vid_det,detrac-non_empty-49_68,len-6,batch-12,save-vis-0,dbg-0,dyn-1,dist-0,lfn,asi-0

<a id="len_9___detrac_0_19_lf_n_"></a>
### len-9       @ detrac-0_19/lfn-->p2s_vid-isl
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,detrac-non_empty-0_19,len-9,batch-2,dbg-0,dyn-1,dist-1,lfn
<a id="on_49_68___len_9_detrac_0_19_lf_n_"></a>
#### on-49_68       @ len-9/detrac-0_19/lfn-->p2s_vid-isl
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=m-resnet_640_detrac-length-9-stride-1-non_empty-seq-0_19-batch_2-lfn,_eval_,vid_det,detrac-non_empty-49_68,len-9,batch-10,save-vis-0,dbg-0,dyn-1,dist-0,lfn,asi-0

<a id="mid_"></a>
# mid
<a id="detrac_0_19___mi_d_"></a>
## detrac-0_19       @ mid-->p2s_vid-isl
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,detrac-non_empty-0_19,batch-3,dbg-0,dyn-1,dist-1
<a id="on_train___detrac_0_19_mi_d_"></a>
### on-train       @ detrac-0_19/mid-->p2s_vid-isl
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_19-batch_18,detrac-non_empty-0_19,batch-48,save-vis-0,dbg-0,dyn-1,dist-0
`strd-2` 
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_19-batch_18,detrac-non_empty-0_19,vstrd-2,batch-12,save-vis-0,dbg-0,dyn-1,dist-0
<a id="on_49_68___detrac_0_19_mi_d_"></a>
### on-49_68       @ detrac-0_19/mid-->p2s_vid-isl
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_19-batch_18,detrac-non_empty-49_68,batch-12,save-vis-0,dbg-0,dyn-1,dist-0
`strd-2`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py --j5=m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_19-batch_18,eval,vid_det,detrac-non_empty-49_68,vstrd-2,batch-12,save-vis-0,dbg-0,dyn-1,dist-0

<a id="detrac_0_9___mi_d_"></a>
## detrac-0_9       @ mid-->p2s_vid-isl
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,detrac-non_empty-0_9,batch-18,dbg-0,dyn-1,dist-1
<a id="on_train___detrac_0_9_mid_"></a>
### on-train       @ detrac-0_9/mid-->p2s_vid-isl
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_9-batch_18,detrac-non_empty-0_9,batch-48,save-vis-0,dbg-0,dyn-1,dist-0
`strd-2`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_9-batch_18,detrac-non_empty-0_9,vstrd-2,batch-48,save-vis-0,dbg-0,dyn-1,dist-0
<a id="on_49_68___detrac_0_9_mid_"></a>
### on-49_68       @ detrac-0_9/mid-->p2s_vid-isl
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_9-batch_18,detrac-non_empty-49_68,batch-1,save-vis-0,dbg-0,dyn-1,dist-0
`strd-2`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_9-batch_18,eval,vid_det,detrac-non_empty-49_68,vstrd-2,batch-48,save-vis-0,dbg-0,dyn-1,dist-0

<a id="detrac_0_48_len_32___mi_d_"></a>
## detrac-0_48-len-32       @ mid-->p2s_vid-isl
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,detrac-non_empty-0_48,batch-6,dbg-0,dyn-1,dist-2,len-32,seq5k,fbb,gxe
`dbg`
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,detrac-non_empty-0_48,batch-2,dbg-0,dyn-1,dist-0,len-32,seq5k,fbb
<a id="on_49_85___detrac_0_48_len_32_mid_"></a>
### on-49_85       @ detrac-0_48-len-32/mid-->p2s_vid-isl
`vstrd-32`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-32-stride-1-non_empty-seq-0_48-batch_6-seq5k-fbb-gxe,detrac-non_empty-49_85,batch-8,save-vis-0,dbg-0,dyn-1,dist-0,len-32,vstrd-32,asi-0,grs
`vstrd-1`
python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-32-stride-1-non_empty-seq-0_48-batch_6-seq5k-fbb-gxe,detrac-non_empty-49_85,batch-9,save-vis-0,dbg-0,dyn-1,dist-1,len-32,vstrd-1,asi-0,iter-186915

<a id="detrac_0_48_len_40___mi_d_"></a>
## detrac-0_48-len-40       @ mid-->p2s_vid-isl
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,detrac-non_empty-0_48,batch-6,dbg-0,dyn-1,dist-2,len-40,seq6k,fbb,gxe


<a id="detrac_0_48_len_48___mi_d_"></a>
## detrac-0_48-len-48       @ mid-->p2s_vid-isl
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,detrac-non_empty-0_48,batch-2,dbg-0,dyn-1,dist-1,len-48,seq7k,fbb


<a id="detrac_0_48_len_56___mi_d_"></a>
## detrac-0_48-len-56       @ mid-->p2s_vid-isl
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,detrac-non_empty-0_48,batch-2,dbg-0,dyn-1,dist-1,len-56,seq8k,fbb


