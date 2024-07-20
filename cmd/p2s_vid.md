<!-- MarkdownTOC -->

- [swin-t](#swin_t_)
    - [ipsc-16_53-jtr-res-1280-pt-len-2       @ swin-t](#ipsc_16_53_jtr_res_1280_pt_len_2___swin_t_)
    - [ipsc-16_53-jtr-res-1280-pt-len-6       @ swin-t](#ipsc_16_53_jtr_res_1280_pt_len_6___swin_t_)
    - [ipsc-16_53-len-2-pt       @ swin-t](#ipsc_16_53_len_2_pt___swin_t_)
        - [on-train       @ ipsc-16_53-len-2-pt/swin-t](#on_train___ipsc_16_53_len_2_pt_swin_t_)
        - [on-54_126       @ ipsc-16_53-len-2-pt/swin-t](#on_54_126___ipsc_16_53_len_2_pt_swin_t_)
    - [ipsc-16_53-len-6-pt       @ swin-t](#ipsc_16_53_len_6_pt___swin_t_)
        - [on-train       @ ipsc-16_53-len-6-pt/swin-t](#on_train___ipsc_16_53_len_6_pt_swin_t_)
        - [on-54_126       @ ipsc-16_53-len-6-pt/swin-t](#on_54_126___ipsc_16_53_len_6_pt_swin_t_)
    - [ipsc-16_53-len-2       @ swin-t](#ipsc_16_53_len_2___swin_t_)
        - [on-train       @ ipsc-16_53-len-2/swin-t](#on_train___ipsc_16_53_len_2_swin_t_)
        - [on-54_126       @ ipsc-16_53-len-2/swin-t](#on_54_126___ipsc_16_53_len_2_swin_t_)
    - [ipsc-16_53-len-6       @ swin-t](#ipsc_16_53_len_6___swin_t_)
        - [on-train       @ ipsc-16_53-len-6/swin-t](#on_train___ipsc_16_53_len_6_swin_t_)
    - [ipsc-16_53-len-6-spd-2       @ swin-t](#ipsc_16_53_len_6_spd_2___swin_t_)
        - [on-train       @ ipsc-16_53-len-6-spd-2/swin-t](#on_train___ipsc_16_53_len_6_spd_2_swin_t_)
        - [on-54_126       @ ipsc-16_53-len-6-spd-2/swin-t](#on_54_126___ipsc_16_53_len_6_spd_2_swin_t_)
    - [ipsc-16_53-len-6-jtr-res-1280       @ swin-t](#ipsc_16_53_len_6_jtr_res_1280___swin_t_)
    - [gram-0_1       @ swin-t](#gram_0_1___swin_t_)
        - [len-2       @ gram-0_1/swin-t](#len_2___gram_0_1_swin_t_)
- [resnet-640-lfn](#resnet_640_lf_n_)
    - [gram-0_1       @ resnet-640-lfn](#gram_0_1___resnet_640_lfn_)
        - [len-9       @ gram-0_1/resnet-640-lfn](#len_9___gram_0_1_resnet_640_lf_n_)
            - [on-train       @ len-9/gram-0_1/resnet-640-lfn](#on_train___len_9_gram_0_1_resnet_640_lf_n_)
        - [len-14-0_2000       @ gram-0_1/resnet-640-lfn](#len_14_0_2000___gram_0_1_resnet_640_lf_n_)
            - [on-3000_5000       @ len-14-0_2000/gram-0_1/resnet-640-lfn](#on_3000_5000___len_14_0_2000_gram_0_1_resnet_640_lf_n_)
        - [len-16-0_2000       @ gram-0_1/resnet-640-lfn](#len_16_0_2000___gram_0_1_resnet_640_lf_n_)
    - [detrac-non_empty       @ resnet-640-lfn](#detrac_non_empty___resnet_640_lfn_)
        - [0_19-jtr       @ detrac-non_empty/resnet-640-lfn](#0_19_jtr___detrac_non_empty_resnet_640_lf_n_)
            - [on-train       @ 0_19-jtr/detrac-non_empty/resnet-640-lfn](#on_train___0_19_jtr_detrac_non_empty_resnet_640_lfn_)
                - [strd-2       @ on-train/0_19-jtr/detrac-non_empty/resnet-640-lfn](#strd_2___on_train_0_19_jtr_detrac_non_empty_resnet_640_lf_n_)
        - [0_19-len-9       @ detrac-non_empty/resnet-640-lfn](#0_19_len_9___detrac_non_empty_resnet_640_lf_n_)
    - [ipsc-16_53-len-2       @ resnet-640-lfn](#ipsc_16_53_len_2___resnet_640_lfn_)
        - [on-train       @ ipsc-16_53-len-2/resnet-640-lfn](#on_train___ipsc_16_53_len_2_resnet_640_lf_n_)
            - [strd-1       @ on-train/ipsc-16_53-len-2/resnet-640-lfn](#strd_1___on_train_ipsc_16_53_len_2_resnet_640_lfn_)
            - [strd-2       @ on-train/ipsc-16_53-len-2/resnet-640-lfn](#strd_2___on_train_ipsc_16_53_len_2_resnet_640_lfn_)
        - [on-54_126       @ ipsc-16_53-len-2/resnet-640-lfn](#on_54_126___ipsc_16_53_len_2_resnet_640_lf_n_)
            - [strd-1       @ on-54_126/ipsc-16_53-len-2/resnet-640-lfn](#strd_1___on_54_126_ipsc_16_53_len_2_resnet_640_lf_n_)
            - [strd-2       @ on-54_126/ipsc-16_53-len-2/resnet-640-lfn](#strd_2___on_54_126_ipsc_16_53_len_2_resnet_640_lf_n_)
    - [ipsc-16_53-len-6       @ resnet-640-lfn](#ipsc_16_53_len_6___resnet_640_lfn_)
        - [on-train       @ ipsc-16_53-len-6/resnet-640-lfn](#on_train___ipsc_16_53_len_6_resnet_640_lf_n_)
            - [strd-1       @ on-train/ipsc-16_53-len-6/resnet-640-lfn](#strd_1___on_train_ipsc_16_53_len_6_resnet_640_lfn_)
            - [strd-6       @ on-train/ipsc-16_53-len-6/resnet-640-lfn](#strd_6___on_train_ipsc_16_53_len_6_resnet_640_lfn_)
        - [on-54_126       @ ipsc-16_53-len-6/resnet-640-lfn](#on_54_126___ipsc_16_53_len_6_resnet_640_lf_n_)
            - [strd-1       @ on-54_126/ipsc-16_53-len-6/resnet-640-lfn](#strd_1___on_54_126_ipsc_16_53_len_6_resnet_640_lf_n_)
            - [strd-6       @ on-54_126/ipsc-16_53-len-6/resnet-640-lfn](#strd_6___on_54_126_ipsc_16_53_len_6_resnet_640_lf_n_)
    - [ipsc-16_53-len-3       @ resnet-640-lfn](#ipsc_16_53_len_3___resnet_640_lfn_)
        - [on-train       @ ipsc-16_53-len-3/resnet-640-lfn](#on_train___ipsc_16_53_len_3_resnet_640_lf_n_)
            - [strd-1       @ on-train/ipsc-16_53-len-3/resnet-640-lfn](#strd_1___on_train_ipsc_16_53_len_3_resnet_640_lfn_)
            - [strd-3       @ on-train/ipsc-16_53-len-3/resnet-640-lfn](#strd_3___on_train_ipsc_16_53_len_3_resnet_640_lfn_)
        - [on-54_126       @ ipsc-16_53-len-3/resnet-640-lfn](#on_54_126___ipsc_16_53_len_3_resnet_640_lf_n_)
            - [strd-1       @ on-54_126/ipsc-16_53-len-3/resnet-640-lfn](#strd_1___on_54_126_ipsc_16_53_len_3_resnet_640_lf_n_)
            - [strd-3       @ on-54_126/ipsc-16_53-len-3/resnet-640-lfn](#strd_3___on_54_126_ipsc_16_53_len_3_resnet_640_lf_n_)
    - [ipsc-16_53-jtr-res-1280-len-2       @ resnet-640-lfn](#ipsc_16_53_jtr_res_1280_len_2___resnet_640_lfn_)
    - [ipsc-16_53-jtr-res-1280-len-6       @ resnet-640-lfn](#ipsc_16_53_jtr_res_1280_len_6___resnet_640_lfn_)
- [resnet-640](#resnet_64_0_)
    - [detrac-0_19       @ resnet-640](#detrac_0_19___resnet_640_)
        - [on-train       @ detrac-0_19/resnet-640](#on_train___detrac_0_19_resnet_640_)
            - [strd-1       @ on-train/detrac-0_19/resnet-640](#strd_1___on_train_detrac_0_19_resnet_64_0_)
            - [strd-2       @ on-train/detrac-0_19/resnet-640](#strd_2___on_train_detrac_0_19_resnet_64_0_)
        - [on-49_68       @ detrac-0_19/resnet-640](#on_49_68___detrac_0_19_resnet_640_)
            - [strd-1       @ on-49_68/detrac-0_19/resnet-640](#strd_1___on_49_68_detrac_0_19_resnet_64_0_)
            - [strd-2       @ on-49_68/detrac-0_19/resnet-640](#strd_2___on_49_68_detrac_0_19_resnet_64_0_)
    - [detrac-0_9       @ resnet-640](#detrac_0_9___resnet_640_)
        - [on-train       @ detrac-0_9/resnet-640](#on_train___detrac_0_9_resnet_64_0_)
            - [strd-2       @ on-train/detrac-0_9/resnet-640](#strd_2___on_train_detrac_0_9_resnet_640_)
        - [on-49_68       @ detrac-0_9/resnet-640](#on_49_68___detrac_0_9_resnet_64_0_)
            - [strd-2       @ on-49_68/detrac-0_9/resnet-640](#strd_2___on_49_68_detrac_0_9_resnet_640_)
    - [0_4       @ resnet-640](#0_4___resnet_640_)
        - [batch-3       @ 0_4/resnet-640](#batch_3___0_4_resnet_640_)
            - [on-g2_0_4       @ batch-3/0_4/resnet-640](#on_g2_0_4___batch_3_0_4_resnet_640_)
        - [batch-6       @ 0_4/resnet-640](#batch_6___0_4_resnet_640_)
            - [on-g2_5_9       @ batch-6/0_4/resnet-640](#on_g2_5_9___batch_6_0_4_resnet_640_)
            - [on-g2_0_4       @ batch-6/0_4/resnet-640](#on_g2_0_4___batch_6_0_4_resnet_640_)
    - [5_9       @ resnet-640](#5_9___resnet_640_)
        - [batch-8       @ 5_9/resnet-640](#batch_8___5_9_resnet_640_)
            - [on-g2_0_4       @ batch-8/5_9/resnet-640](#on_g2_0_4___batch_8_5_9_resnet_640_)
            - [on-5_9       @ batch-8/5_9/resnet-640](#on_5_9___batch_8_5_9_resnet_640_)
        - [fg-4       @ 5_9/resnet-640](#fg_4___5_9_resnet_640_)
    - [16_53-len-2       @ resnet-640](#16_53_len_2___resnet_640_)
        - [on-train       @ 16_53-len-2/resnet-640](#on_train___16_53_len_2_resnet_640_)
        - [on-54_126       @ 16_53-len-2/resnet-640](#on_54_126___16_53_len_2_resnet_640_)
    - [16_53-len-6       @ resnet-640](#16_53_len_6___resnet_640_)
        - [on-train       @ 16_53-len-6/resnet-640](#on_train___16_53_len_6_resnet_640_)
        - [on-54_126       @ 16_53-len-6/resnet-640](#on_54_126___16_53_len_6_resnet_640_)
    - [0_37       @ resnet-640](#0_37___resnet_640_)
        - [on-54_126       @ 0_37/resnet-640](#on_54_126___0_37_resnet_64_0_)
            - [strd-1       @ on-54_126/0_37/resnet-640](#strd_1___on_54_126_0_37_resnet_64_0_)
            - [strd-2       @ on-54_126/0_37/resnet-640](#strd_2___on_54_126_0_37_resnet_64_0_)
    - [0_37-fg-4       @ resnet-640](#0_37_fg_4___resnet_640_)
    - [16_53-jtr-res-1280       @ resnet-640](#16_53_jtr_res_1280___resnet_640_)
        - [on-train       @ 16_53-jtr-res-1280/resnet-640](#on_train___16_53_jtr_res_1280_resnet_64_0_)
            - [strd-1       @ on-train/16_53-jtr-res-1280/resnet-640](#strd_1___on_train_16_53_jtr_res_1280_resnet_640_)
            - [strd-2       @ on-train/16_53-jtr-res-1280/resnet-640](#strd_2___on_train_16_53_jtr_res_1280_resnet_640_)
        - [on-54_126       @ 16_53-jtr-res-1280/resnet-640](#on_54_126___16_53_jtr_res_1280_resnet_64_0_)
            - [strd-1       @ on-54_126/16_53-jtr-res-1280/resnet-640](#strd_1___on_54_126_16_53_jtr_res_1280_resnet_64_0_)
            - [strd-2       @ on-54_126/16_53-jtr-res-1280/resnet-640](#strd_2___on_54_126_16_53_jtr_res_1280_resnet_64_0_)
    - [16_53-jtr-res-1280-len-6       @ resnet-640](#16_53_jtr_res_1280_len_6___resnet_640_)
    - [16_53-jtr-res-1280-len-6-val       @ resnet-640](#16_53_jtr_res_1280_len_6_val___resnet_640_)
        - [on-train       @ 16_53-jtr-res-1280-len-6-val/resnet-640](#on_train___16_53_jtr_res_1280_len_6_val_resnet_64_0_)
            - [strd-1       @ on-train/16_53-jtr-res-1280-len-6-val/resnet-640](#strd_1___on_train_16_53_jtr_res_1280_len_6_val_resnet_640_)
            - [strd-6       @ on-train/16_53-jtr-res-1280-len-6-val/resnet-640](#strd_6___on_train_16_53_jtr_res_1280_len_6_val_resnet_640_)
        - [on-54_126       @ 16_53-jtr-res-1280-len-6-val/resnet-640](#on_54_126___16_53_jtr_res_1280_len_6_val_resnet_64_0_)
            - [strd-1       @ on-54_126/16_53-jtr-res-1280-len-6-val/resnet-640](#strd_1___on_54_126_16_53_jtr_res_1280_len_6_val_resnet_64_0_)
            - [strd-6       @ on-54_126/16_53-jtr-res-1280-len-6-val/resnet-640](#strd_6___on_54_126_16_53_jtr_res_1280_len_6_val_resnet_64_0_)
    - [acamp       @ resnet-640](#acamp___resnet_640_)
        - [1k8_vid_entire_seq       @ acamp/resnet-640](#1k8_vid_entire_seq___acamp_resnet_640_)
        - [on-train       @ acamp/resnet-640](#on_train___acamp_resnet_640_)
        - [on-inv       @ acamp/resnet-640](#on_inv___acamp_resnet_640_)
            - [vstrd-1       @ on-inv/acamp/resnet-640](#vstrd_1___on_inv_acamp_resnet_64_0_)
            - [vstrd-2       @ on-inv/acamp/resnet-640](#vstrd_2___on_inv_acamp_resnet_64_0_)
        - [1k8_vid_entire_seq-jtr-res-1280       @ acamp/resnet-640](#1k8_vid_entire_seq_jtr_res_1280___acamp_resnet_640_)
    - [10k6_vid_entire_seq       @ resnet-640](#10k6_vid_entire_seq___resnet_640_)
    - [10k6_vid_entire_seq-jtr-res-1280       @ resnet-640](#10k6_vid_entire_seq_jtr_res_1280___resnet_640_)

<!-- /MarkdownTOC -->
<a id="swin_t_"></a>
# swin-t 
<a id="ipsc_16_53_jtr_res_1280_pt_len_2___swin_t_"></a>
## ipsc-16_53-jtr-res-1280-pt-len-2       @ swin-t-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-16_53,len-2,vstrd-1,batch-9,dbg-0,dyn-1,dist-0,swin-t,spt,jtr,res-1280

<a id="ipsc_16_53_jtr_res_1280_pt_len_6___swin_t_"></a>
## ipsc-16_53-jtr-res-1280-pt-len-6       @ swin-t-->p2s_vid
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-16_53,len-6,vstrd-1,batch-9,dbg-0,dyn-1,dist-0,swin-t,spt,jtr,res-1280

<a id="ipsc_16_53_len_2_pt___swin_t_"></a>
## ipsc-16_53-len-2-pt       @ swin-t-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-16_53,len-2,vstrd-1,batch-9,dbg-0,dyn-1,dist-0,swin-t,spt
<a id="on_train___ipsc_16_53_len_2_pt_swin_t_"></a>
### on-train       @ ipsc-16_53-len-2-pt/swin-t-->p2s_vid
`strd-1` 
python3 run.py --cfg=configs/config_video_det.py  --j5=m-swin_t_pt_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_9,_eval_,vid_det,ipsc-16_53,len-2,vstrd-1,batch-6,save-vis-1,dbg-0,dyn-1
`strd-2` 
python3 run.py --cfg=configs/config_video_det.py  --j5=m-swin_t_pt_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_9,_eval_,vid_det,ipsc-16_53,len-2,vstrd-2,batch-6,save-vis-1,dbg-0,dyn-1
<a id="on_54_126___ipsc_16_53_len_2_pt_swin_t_"></a>
### on-54_126       @ ipsc-16_53-len-2-pt/swin-t-->p2s_vid
`strd-1` 
python3 run.py --cfg=configs/config_video_det.py  --j5=m-swin_t_pt_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_9,_eval_,vid_det,ipsc-54_126,len-2,vstrd-1,batch-6,save-vis-1,dbg-0,dyn-1
`strd-2` 
python3 run.py --cfg=configs/config_video_det.py  --j5=m-swin_t_pt_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_9,_eval_,vid_det,ipsc-54_126,len-2,vstrd-2,batch-6,save-vis-1,dbg-0,dyn-1

<a id="ipsc_16_53_len_6_pt___swin_t_"></a>
## ipsc-16_53-len-6-pt       @ swin-t-->p2s_vid
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-16_53,len-6,vstrd-1,batch-9,dbg-0,dyn-1,dist-0,swin-t,spt
<a id="on_train___ipsc_16_53_len_6_pt_swin_t_"></a>
### on-train       @ ipsc-16_53-len-6-pt/swin-t-->p2s_vid
`strd-1` 
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py  --j5=m-swin_t_pt_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_9,_eval_,vid_det,ipsc-16_53,len-6,vstrd-1,batch-6,save-vis-1,dbg-0,dyn-1
`strd-2` 
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py  --j5=m-swin_t_pt_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_9,_eval_,vid_det,ipsc-16_53,len-6,vstrd-6,batch-6,save-vis-1,dbg-0,dyn-1
<a id="on_54_126___ipsc_16_53_len_6_pt_swin_t_"></a>
### on-54_126       @ ipsc-16_53-len-6-pt/swin-t-->p2s_vid
`strd-1` 
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py  --j5=m-swin_t_pt_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_9,_eval_,vid_det,ipsc-54_126,len-6,vstrd-1,batch-6,save-vis-1,dbg-0,dyn-1
`strd-2` 
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py  --j5=m-swin_t_pt_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_9,_eval_,vid_det,ipsc-54_126,len-6,vstrd-6,batch-6,save-vis-1,dbg-0,dyn-1

<a id="ipsc_16_53_len_2___swin_t_"></a>
## ipsc-16_53-len-2       @ swin-t-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-16_53,len-2,vstrd-1,batch-9,dbg-0,dyn-1,dist-0,swin-t
<a id="on_train___ipsc_16_53_len_2_swin_t_"></a>
### on-train       @ ipsc-16_53-len-2/swin-t-->p2s_vid
`strd-1` 
python3 run.py --cfg=configs/config_video_det.py  --j5=m-swin_t_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_9,_eval_,vid_det,ipsc-16_53,len-2,vstrd-1,batch-4,save-vis-1,dbg-0,dyn-1
`strd-2` 
python3 run.py --cfg=configs/config_video_det.py  --j5=m-swin_t_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_9,_eval_,vid_det,ipsc-16_53,len-2,vstrd-2,batch-4,save-vis-1,dbg-0,dyn-1
<a id="on_54_126___ipsc_16_53_len_2_swin_t_"></a>
### on-54_126       @ ipsc-16_53-len-2/swin-t-->p2s_vid
`strd-1` 
python3 run.py --cfg=configs/config_video_det.py  --j5=m-swin_t_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_9,_eval_,vid_det,ipsc-54_126,len-2,vstrd-1,batch-4,save-vis-1,dbg-0,dyn-1
`strd-2` 
python3 run.py --cfg=configs/config_video_det.py  --j5=m-swin_t_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_9,_eval_,vid_det,ipsc-54_126,len-2,vstrd-2,batch-4,save-vis-1,dbg-0,dyn-1
`batch-16`
python3 run.py --cfg=configs/config_video_det.py  --j5=m-swin_t_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_9,_eval_,vid_det,ipsc-54_126,len-2,vstrd-2,batch-16,save-vis-1,dbg-0,dyn-1
`batch-64`
python3 run.py --cfg=configs/config_video_det.py  --j5=m-swin_t_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_9,_eval_,vid_det,ipsc-54_126,len-2,vstrd-2,batch-64,save-vis-1,dbg-0,dyn-1
`batch-1`
python3 run.py --cfg=configs/config_video_det.py  --j5=m-swin_t_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_9,_eval_,vid_det,ipsc-54_126,len-2,vstrd-2,batch-1,save-vis-1,dbg-0,dyn-1

<a id="ipsc_16_53_len_6___swin_t_"></a>
## ipsc-16_53-len-6       @ swin-t-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-16_53,len-6,vstrd-1,batch-1,dbg-0,dyn-1,dist-0,swin-t
<a id="on_train___ipsc_16_53_len_6_swin_t_"></a>
### on-train       @ ipsc-16_53-len-6/swin-t-->p2s_vid
`strd-1` 
python3 run.py --cfg=configs/config_video_det.py  --j5=m-swin_t_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_3,_eval_,vid_det,ipsc-16_53,len-6,vstrd-1,batch-4,save-vis-1,dbg-0,dyn-1
`strd-6` 
python3 run.py --cfg=configs/config_video_det.py  --j5=m-swin_t_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_3,_eval_,vid_det,ipsc-16_53,len-6,vstrd-6,batch-32,save-vis-1,dbg-0,dyn-1

<a id="ipsc_16_53_len_6_spd_2___swin_t_"></a>
## ipsc-16_53-len-6-spd-2       @ swin-t-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-16_53,len-6,vstrd-1,batch-3,dbg-0,dyn-1,dist-0,swin-t,spd-2

<a id="on_train___ipsc_16_53_len_6_spd_2_swin_t_"></a>
### on-train       @ ipsc-16_53-len-6-spd-2/swin-t-->p2s_vid
`strd-1` 
python3 run.py --cfg=configs/config_video_det.py  --j5=m-swin_t_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_3,_eval_,vid_det,ipsc-16_53,len-6,vstrd-1,batch-4,save-vis-1,dbg-0,dyn-1
`strd-6` 
python3 run.py --cfg=configs/config_video_det.py  --j5=m-swin_t_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_3,_eval_,vid_det,ipsc-16_53,len-6,vstrd-6,batch-32,save-vis-1,dbg-0,dyn-1

<a id="on_54_126___ipsc_16_53_len_6_spd_2_swin_t_"></a>
### on-54_126       @ ipsc-16_53-len-6-spd-2/swin-t-->p2s_vid
`strd-1` 
python3 run.py --cfg=configs/config_video_det.py  --j5=m-swin_t_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_3,_eval_,vid_det,ipsc-54_126,len-6,vstrd-1,batch-4,save-vis-1,dbg-0,dyn-1
`strd-6` 
python3 run.py --cfg=configs/config_video_det.py  --j5=m-swin_t_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_3,_eval_,vid_det,ipsc-54_126,len-6,vstrd-6,batch-4,save-vis-1,dbg-0,dyn-1
`batch-16`
python3 run.py --cfg=configs/config_video_det.py  --j5=m-swin_t_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_3,_eval_,vid_det,ipsc-54_126,len-6,vstrd-6,batch-16,save-vis-1,dbg-0,dyn-1

<a id="ipsc_16_53_len_6_jtr_res_1280___swin_t_"></a>
## ipsc-16_53-len-6-jtr-res-1280       @ swin-t-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-16_53,len-6,vstrd-1,batch-9,dbg-0,dyn-1,dist-0,swin-t,jtr,res-1280

<a id="gram_0_1___swin_t_"></a>
## gram-0_1       @ swin-t-->p2s_vid
<a id="len_2___gram_0_1_swin_t_"></a>
### len-2       @ gram-0_1/swin-t-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,gram-0_1,len-2,batch-3,dbg-1,dyn-1,dist-0,swin-t

<a id="resnet_640_lf_n_"></a>
# resnet-640-lfn 
<a id="gram_0_1___resnet_640_lfn_"></a>
## gram-0_1       @ resnet-640-lfn-->p2s_vid
<a id="len_9___gram_0_1_resnet_640_lf_n_"></a>
### len-9       @ gram-0_1/resnet-640-lfn-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,gram-0_1,len-9,batch-2,dbg-0,dyn-1,dist-0,lfn
<a id="on_train___len_9_gram_0_1_resnet_640_lf_n_"></a>
#### on-train       @ len-9/gram-0_1/resnet-640-lfn-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=vid_det,m-resnet_640_gram_only-length-9-stride-1-seq-0_1-batch_2-lfn,_eval_,gram-0_1,len-9,vstrd-9,batch-2,save-vis-1,dbg-0,dyn-1,dist-0

<a id="len_14_0_2000___gram_0_1_resnet_640_lf_n_"></a>
### len-14-0_2000       @ gram-0_1/resnet-640-lfn-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,gram-0_1-0_2000,len-14,batch-2,dbg-0,dyn-1,dist-1,lfn
<a id="on_3000_5000___len_14_0_2000_gram_0_1_resnet_640_lf_n_"></a>
#### on-3000_5000       @ len-14-0_2000/gram-0_1/resnet-640-lfn-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=vid_det,m-resnet_640_gram_only-0_2000-length-14-stride-1-seq-0_1-batch_2-lfn,_eval_,gram-0_1-3000_5000,len-14,vstrd-14,batch-1,save-vis-1,dbg-0,dyn-1,dist-0

<a id="len_16_0_2000___gram_0_1_resnet_640_lf_n_"></a>
### len-16-0_2000       @ gram-0_1/resnet-640-lfn-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,gram-0_1-0_2000,len-16,batch-2,dbg-0,dyn-1,dist-1,lfn

<a id="detrac_non_empty___resnet_640_lfn_"></a>
## detrac-non_empty       @ resnet-640-lfn-->p2s_vid
<a id="0_19_jtr___detrac_non_empty_resnet_640_lf_n_"></a>
### 0_19-jtr       @ detrac-non_empty/resnet-640-lfn-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,detrac-non_empty-0_19,len-6,batch-6,dbg-0,dyn-1,dist-1,lfn,jtr
<a id="on_train___0_19_jtr_detrac_non_empty_resnet_640_lfn_"></a>
#### on-train       @ 0_19-jtr/detrac-non_empty/resnet-640-lfn-->p2s_vid
<a id="strd_2___on_train_0_19_jtr_detrac_non_empty_resnet_640_lf_n_"></a>
##### strd-2       @ on-train/0_19-jtr/detrac-non_empty/resnet-640-lfn-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_19-batch_18,detrac-non_empty-0_19,vstrd-2,batch-12,save-vis-1,dbg-0,dyn-1,dist-0

<a id="0_19_len_9___detrac_non_empty_resnet_640_lf_n_"></a>
### 0_19-len-9       @ detrac-non_empty/resnet-640-lfn-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,detrac-non_empty-0_19,len-9,batch-2,dbg-0,dyn-1,dist-1,lfn

<a id="ipsc_16_53_len_2___resnet_640_lfn_"></a>
## ipsc-16_53-len-2       @ resnet-640-lfn-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-16_53,len-2,vstrd-1,batch-9,dbg-0,dyn-1,dist-0,lfn
<a id="on_train___ipsc_16_53_len_2_resnet_640_lf_n_"></a>
### on-train       @ ipsc-16_53-len-2/resnet-640-lfn-->p2s_vid
<a id="strd_1___on_train_ipsc_16_53_len_2_resnet_640_lfn_"></a>
#### strd-1       @ on-train/ipsc-16_53-len-2/resnet-640-lfn-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_9-lfn,_eval_,vid_det,ipsc-16_53,len-2,vstrd-1,batch-6,save-vis-1,dbg-0,dyn-1
<a id="strd_2___on_train_ipsc_16_53_len_2_resnet_640_lfn_"></a>
#### strd-2       @ on-train/ipsc-16_53-len-2/resnet-640-lfn-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_9-lfn,_eval_,vid_det,ipsc-16_53,len-2,vstrd-2,batch-6,save-vis-1,dbg-0,dyn-1
<a id="on_54_126___ipsc_16_53_len_2_resnet_640_lf_n_"></a>
### on-54_126       @ ipsc-16_53-len-2/resnet-640-lfn-->p2s_vid
<a id="strd_1___on_54_126_ipsc_16_53_len_2_resnet_640_lf_n_"></a>
#### strd-1       @ on-54_126/ipsc-16_53-len-2/resnet-640-lfn-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_9-lfn,_eval_,vid_det,ipsc-54_126,len-2,vstrd-1,batch-6,save-vis-1,dbg-0,dyn-1
<a id="strd_2___on_54_126_ipsc_16_53_len_2_resnet_640_lf_n_"></a>
#### strd-2       @ on-54_126/ipsc-16_53-len-2/resnet-640-lfn-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_9-lfn,_eval_,vid_det,ipsc-54_126,len-2,vstrd-2,batch-4,save-vis-1,dbg-0,dyn-1

<a id="ipsc_16_53_len_6___resnet_640_lfn_"></a>
## ipsc-16_53-len-6       @ resnet-640-lfn-->p2s_vid
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-16_53,len-6,vstrd-1,batch-3,dbg-0,dyn-1,dist-0,lfn
<a id="on_train___ipsc_16_53_len_6_resnet_640_lf_n_"></a>
### on-train       @ ipsc-16_53-len-6/resnet-640-lfn-->p2s_vid
<a id="strd_1___on_train_ipsc_16_53_len_6_resnet_640_lfn_"></a>
#### strd-1       @ on-train/ipsc-16_53-len-6/resnet-640-lfn-->p2s_vid
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_3-lfn,_eval_,vid_det,ipsc-16_53,len-6,vstrd-1,batch-2,save-vis-1,dbg-0,dyn-1
<a id="strd_6___on_train_ipsc_16_53_len_6_resnet_640_lfn_"></a>
#### strd-6       @ on-train/ipsc-16_53-len-6/resnet-640-lfn-->p2s_vid
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_3-lfn,_eval_,vid_det,ipsc-16_53,len-6,vstrd-6,batch-2,save-vis-1,dbg-0,dyn-1
<a id="on_54_126___ipsc_16_53_len_6_resnet_640_lf_n_"></a>
### on-54_126       @ ipsc-16_53-len-6/resnet-640-lfn-->p2s_vid
<a id="strd_1___on_54_126_ipsc_16_53_len_6_resnet_640_lf_n_"></a>
#### strd-1       @ on-54_126/ipsc-16_53-len-6/resnet-640-lfn-->p2s_vid
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_3-lfn,_eval_,vid_det,ipsc-54_126,len-6,vstrd-1,batch-2,save-vis-1,dbg-0,dyn-1
<a id="strd_6___on_54_126_ipsc_16_53_len_6_resnet_640_lf_n_"></a>
#### strd-6       @ on-54_126/ipsc-16_53-len-6/resnet-640-lfn-->p2s_vid
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_3-lfn,_eval_,vid_det,ipsc-54_126,len-6,vstrd-6,batch-2,save-vis-1,dbg-0,dyn-1

<a id="ipsc_16_53_len_3___resnet_640_lfn_"></a>
## ipsc-16_53-len-3       @ resnet-640-lfn-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-16_53,len-3,vstrd-1,batch-6,dbg-0,dyn-1,dist-0,lfn
<a id="on_train___ipsc_16_53_len_3_resnet_640_lf_n_"></a>
### on-train       @ ipsc-16_53-len-3/resnet-640-lfn-->p2s_vid
<a id="strd_1___on_train_ipsc_16_53_len_3_resnet_640_lfn_"></a>
#### strd-1       @ on-train/ipsc-16_53-len-3/resnet-640-lfn-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-3-stride-1-batch_6-lfn,_eval_,vid_det,ipsc-16_53,len-3,vstrd-1,batch-24,save-vis-1,dbg-0,dyn-1,lfn3d
<a id="strd_3___on_train_ipsc_16_53_len_3_resnet_640_lfn_"></a>
#### strd-3       @ on-train/ipsc-16_53-len-3/resnet-640-lfn-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-3-stride-1-batch_6-lfn,_eval_,vid_det,ipsc-16_53,len-3,vstrd-3,batch-24,save-vis-1,dbg-0,dyn-1,lfn3d
<a id="on_54_126___ipsc_16_53_len_3_resnet_640_lf_n_"></a>
### on-54_126       @ ipsc-16_53-len-3/resnet-640-lfn-->p2s_vid
<a id="strd_1___on_54_126_ipsc_16_53_len_3_resnet_640_lf_n_"></a>
#### strd-1       @ on-54_126/ipsc-16_53-len-3/resnet-640-lfn-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-3-stride-1-batch_6-lfn,_eval_,vid_det,ipsc-54_126,len-3,vstrd-1,batch-24,save-vis-1,dbg-0,dyn-1,lfn3d
<a id="strd_3___on_54_126_ipsc_16_53_len_3_resnet_640_lf_n_"></a>
#### strd-3       @ on-54_126/ipsc-16_53-len-3/resnet-640-lfn-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-3-stride-1-batch_6-lfn,_eval_,vid_det,ipsc-54_126,len-3,vstrd-3,batch-24,save-vis-1,dbg-0,dyn-1,lfn3d

<a id="ipsc_16_53_jtr_res_1280_len_2___resnet_640_lfn_"></a>
## ipsc-16_53-jtr-res-1280-len-2       @ resnet-640-lfn-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-16_53,len-2,vstrd-1,batch-9,dbg-0,dyn-1,dist-0,lfn,jtr,res-1280

<a id="ipsc_16_53_jtr_res_1280_len_6___resnet_640_lfn_"></a>
## ipsc-16_53-jtr-res-1280-len-6       @ resnet-640-lfn-->p2s_vid
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-16_53,len-6,vstrd-1,batch-3,dbg-0,dyn-1,dist-0,lfn,jtr,res-1280

<a id="resnet_64_0_"></a>
# resnet-640 

<a id="detrac_0_19___resnet_640_"></a>
## detrac-0_19       @ resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,detrac-non_empty-0_19,batch-3,dbg-0,dyn-1,dist-1
<a id="on_train___detrac_0_19_resnet_640_"></a>
### on-train       @ detrac-0_19/resnet-640-->p2s_vid
<a id="strd_1___on_train_detrac_0_19_resnet_64_0_"></a>
#### strd-1       @ on-train/detrac-0_19/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_19-batch_18,detrac-non_empty-0_19,batch-48,save-vis-1,dbg-0,dyn-1,dist-0
<a id="strd_2___on_train_detrac_0_19_resnet_64_0_"></a>
#### strd-2       @ on-train/detrac-0_19/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_19-batch_18,detrac-non_empty-0_19,vstrd-2,batch-12,save-vis-1,dbg-0,dyn-1,dist-0
<a id="on_49_68___detrac_0_19_resnet_640_"></a>
### on-49_68       @ detrac-0_19/resnet-640-->p2s_vid
<a id="strd_1___on_49_68_detrac_0_19_resnet_64_0_"></a>
#### strd-1       @ on-49_68/detrac-0_19/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_19-batch_18,detrac-non_empty-49_68,batch-48,save-vis-1,dbg-0,dyn-1,dist-0
<a id="strd_2___on_49_68_detrac_0_19_resnet_64_0_"></a>
#### strd-2       @ on-49_68/detrac-0_19/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_19-batch_18,eval,vid_det,detrac-non_empty-49_68,vstrd-2,batch-12,save-vis-1,dbg-0,dyn-1,dist-0


<a id="detrac_0_9___resnet_640_"></a>
## detrac-0_9       @ resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,detrac-non_empty-0_9,batch-18,dbg-0,dyn-1,dist-1
<a id="on_train___detrac_0_9_resnet_64_0_"></a>
### on-train       @ detrac-0_9/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_9-batch_18,detrac-non_empty-0_9,batch-48,save-vis-1,dbg-0,dyn-1,dist-0
<a id="strd_2___on_train_detrac_0_9_resnet_640_"></a>
#### strd-2       @ on-train/detrac-0_9/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_9-batch_18,detrac-non_empty-0_9,vstrd-2,batch-48,save-vis-1,dbg-0,dyn-1,dist-0
<a id="on_49_68___detrac_0_9_resnet_64_0_"></a>
### on-49_68       @ detrac-0_9/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_9-batch_18,detrac-non_empty-49_68,batch-1,save-vis-1,dbg-0,dyn-1,dist-0
<a id="strd_2___on_49_68_detrac_0_9_resnet_640_"></a>
#### strd-2       @ on-49_68/detrac-0_9/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_9-batch_18,eval,vid_det,detrac-non_empty-49_68,vstrd-2,batch-48,save-vis-1,dbg-0,dyn-1,dist-0

<a id="0_4___resnet_640_"></a>
## 0_4       @ resnet-640-->p2s_vid
<a id="batch_3___0_4_resnet_640_"></a>
### batch-3       @ 0_4/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-0,ipsc-g2_0_4,batch-3,dbg-1,dyn-1,ep-4000
<a id="on_g2_0_4___batch_3_0_4_resnet_640_"></a>
#### on-g2_0_4       @ batch-3/0_4/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_ext_reorg_roi_g2_0_4-length-2-stride-1-batch_3,ipsc-g2_0_4,batch-1,save-vis-1,dbg-1,dyn-1

<a id="batch_6___0_4_resnet_640_"></a>
### batch-6       @ 0_4/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-0,ipsc-g2_0_4,batch-6,dbg-0,dyn-1,ep-4000,dist-1
<a id="on_g2_5_9___batch_6_0_4_resnet_640_"></a>
#### on-g2_5_9       @ batch-6/0_4/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=eval,vid_det,m-resnet_640_ext_reorg_roi_g2_0_4-length-2-stride-1-batch_6,ipsc-g2_5_9,batch-3,save-vis-1,dbg-1,dyn-1
<a id="on_g2_0_4___batch_6_0_4_resnet_640_"></a>
#### on-g2_0_4       @ batch-6/0_4/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_ext_reorg_roi_g2_0_4-length-2-stride-1-batch_6,ipsc-g2_0_4,batch-1,save-vis-1,dbg-1,dyn-1

<a id="5_9___resnet_640_"></a>
## 5_9       @ resnet-640-->p2s_vid
<a id="batch_8___5_9_resnet_640_"></a>
### batch-8       @ 5_9/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-g2_5_9,batch-3,dbg-1,dyn-100,dist-0,ckpt_ep-20
<a id="on_g2_0_4___batch_8_5_9_resnet_640_"></a>
#### on-g2_0_4       @ batch-8/5_9/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py  --j5=eval,vid_det,m-resnet_640_ext_reorg_roi_g2_5_9-length-2-stride-1-batch_8,ipsc-g2_0_4,batch-4,save-vis-1,dbg-1,dyn-1
**12094**
python3 run.py --cfg=configs/config_video_det.py  --j5=eval,vid_det,m-resnet_640_ext_reorg_roi_g2_5_9-length-2-stride-1-batch_8,ipsc-12094_0_4,batch-4,save-vis-1,dbg-1,dyn-1
**12094_short**
python3 run.py --cfg=configs/config_video_det.py  --j5=eval,vid_det,m-resnet_640_ext_reorg_roi_g2_5_9-length-2-stride-1-batch_8,ipsc-12094_short_0_4,batch-4,save-vis-1,dbg-1,dyn-1
<a id="on_5_9___batch_8_5_9_resnet_640_"></a>
#### on-5_9       @ batch-8/5_9/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py  --j5=eval,vid_det,m-resnet_640_ext_reorg_roi_g2_5_9-length-2-stride-1-batch_8,ipsc-g2_5_9,batch-4,save-vis-1,dbg-1,dyn-1

<a id="fg_4___5_9_resnet_640_"></a>
### fg-4       @ 5_9/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-g2_5_9,batch-4,dbg-1,dyn-1,dist-1,ckpt_ep-20,fg-4

<a id="16_53_len_2___resnet_640_"></a>
## 16_53-len-2       @ resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-16_53,len-2,vstrd-1,batch-9,dbg-0,dyn-1,dist-0
<a id="on_train___16_53_len_2_resnet_640_"></a>
### on-train       @ 16_53-len-2/resnet-640-->p2s_vid
`strd-1` 
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_9,_eval_,vid_det,ipsc-16_53,len-2,vstrd-1,batch-4,save-vis-1,dbg-0,dyn-1
`strd-2` 
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_9,_eval_,vid_det,ipsc-16_53,len-2,vstrd-2,batch-4,save-vis-1,dbg-0,dyn-1
<a id="on_54_126___16_53_len_2_resnet_640_"></a>
### on-54_126       @ 16_53-len-2/resnet-640-->p2s_vid
`strd-1` 
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_9,_eval_,vid_det,ipsc-54_126,len-2,vstrd-1,batch-4,save-vis-1,dbg-0,dyn-1
`strd-2` 
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_9,_eval_,vid_det,ipsc-54_126,len-2,vstrd-2,batch-4,save-vis-1,dbg-0,dyn-1

<a id="16_53_len_6___resnet_640_"></a>
## 16_53-len-6       @ resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-16_53,len-6,vstrd-1,batch-3,dbg-0,dyn-1,dist-0
<a id="on_train___16_53_len_6_resnet_640_"></a>
### on-train       @ 16_53-len-6/resnet-640-->p2s_vid
<a id="strd_1___on_54_126_ipsc_16_53_resnet_64_0_"></a>
`strd-1` 
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_3,_eval_,vid_det,ipsc-16_53,len-6,vstrd-1,batch-4,save-vis-1,dbg-0,dyn-1
<a id="strd_2___on_54_126_ipsc_16_53_resnet_64_0_"></a>
`strd-6` 
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_3,_eval_,vid_det,ipsc-16_53,len-6,vstrd-6,batch-4,save-vis-1,dbg-0,dyn-1
<a id="on_54_126___16_53_len_6_resnet_640_"></a>
### on-54_126       @ 16_53-len-6/resnet-640-->p2s_vid
<a id="strd_1___on_54_126_ipsc_16_53_resnet_64_0_"></a>
`strd-1` 
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_3,_eval_,vid_det,ipsc-54_126,len-6,vstrd-1,batch-4,save-vis-1,dbg-0,dyn-1
<a id="strd_2___on_54_126_ipsc_16_53_resnet_64_0_"></a>
`strd-6` 
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_3,_eval_,vid_det,ipsc-54_126,len-6,vstrd-6,batch-4,save-vis-1,dbg-0,dyn-1

<a id="0_37___resnet_640_"></a>
## 0_37       @ resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-0_37,len-2,vstrd-1,batch-18,dbg-0,dyn-1,dist-1
<a id="on_54_126___0_37_resnet_64_0_"></a>
### on-54_126       @ 0_37/resnet-640-->p2s_vid
<a id="strd_1___on_54_126_0_37_resnet_64_0_"></a>
#### strd-1       @ on-54_126/0_37/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-0_37-length-2-stride-1-batch_18,_eval_,vid_det,ipsc-54_126,len-2,vstrd-1,batch-36,save-vis-1,dbg-0,dyn-1
<a id="strd_2___on_54_126_0_37_resnet_64_0_"></a>
#### strd-2       @ on-54_126/0_37/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-0_37-length-2-stride-1-batch_18,_eval_,vid_det,ipsc-54_126,len-2,vstrd-2,batch-36,save-vis-1,dbg-0,dyn-1

<a id="0_37_fg_4___resnet_640_"></a>
## 0_37-fg-4       @ resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-g2_0_37,len-2,vstrd-1,fg-4,batch-16,dbg-0,dyn-1,dist-1

<a id="16_53_jtr_res_1280___resnet_640_"></a>
## 16_53-jtr-res-1280       @ resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=_val_,ipsc-54_126,batch-2,len-2,vstrd-2,sample-8,_train_,resnet-640,vid_det,pt-1,ipsc-16_53,len-2,vstrd-1,batch-9,dbg-0,dyn-1,dist-00,jtr,res-1280
<a id="on_train___16_53_jtr_res_1280_resnet_64_0_"></a>
### on-train       @ 16_53-jtr-res-1280/resnet-640-->p2s_vid
<a id="strd_1___on_train_16_53_jtr_res_1280_resnet_640_"></a>
#### strd-1       @ on-train/16_53-jtr-res-1280/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_9-jtr-res_1280,_eval_,vid_det,ipsc-16_53,len-2,vstrd-1,batch-4,save-vis-1,dbg-0,dyn-1,sample-0
<a id="strd_2___on_train_16_53_jtr_res_1280_resnet_640_"></a>
#### strd-2       @ on-train/16_53-jtr-res-1280/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_9-jtr-res_1280,_eval_,vid_det,ipsc-16_53,len-2,vstrd-2,batch-4,save-vis-1,dbg-0,dyn-1,sample-0
<a id="on_54_126___16_53_jtr_res_1280_resnet_64_0_"></a>
### on-54_126       @ 16_53-jtr-res-1280/resnet-640-->p2s_vid
<a id="strd_1___on_54_126_16_53_jtr_res_1280_resnet_64_0_"></a>
#### strd-1       @ on-54_126/16_53-jtr-res-1280/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_9-jtr-res_1280,_eval_,vid_det,ipsc-54_126,len-2,vstrd-1,batch-4,save-vis-1,dbg-0,dyn-1,sample-0
<a id="strd_2___on_54_126_16_53_jtr_res_1280_resnet_64_0_"></a>
#### strd-2       @ on-54_126/16_53-jtr-res-1280/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_9-jtr-res_1280,_eval_,vid_det,ipsc-54_126,len-2,vstrd-2,batch-4,save-vis-1,dbg-0,dyn-1,sample-0

<a id="16_53_jtr_res_1280_len_6___resnet_640_"></a>
## 16_53-jtr-res-1280-len-6       @ resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-16_53,len-6,vstrd-1,batch-6,dbg-0,dyn-1,dist-1,jtr,res-1280

<a id="16_53_jtr_res_1280_len_6_val___resnet_640_"></a>
## 16_53-jtr-res-1280-len-6-val       @ resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=_val_,ipsc-54_126,batch-2,len-6,vstrd-6,sample-4,_train_,resnet-640,vid_det,pt-1,ipsc-16_53,len-6,vstrd-1,batch-6,dbg-0,dyn-1,dist-1,jtr,res-1280
<a id="on_train___16_53_jtr_res_1280_len_6_val_resnet_64_0_"></a>
### on-train       @ 16_53-jtr-res-1280-len-6-val/resnet-640-->p2s_vid
<a id="strd_1___on_train_16_53_jtr_res_1280_len_6_val_resnet_640_"></a>
#### strd-1       @ on-train/16_53-jtr-res-1280-len-6-val/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_6-jtr-res_1280,_eval_,vid_det,ipsc-16_53,len-6,vstrd-1,batch-4,save-vis-1,dbg-0,dyn-1
<a id="strd_6___on_train_16_53_jtr_res_1280_len_6_val_resnet_640_"></a>
#### strd-6       @ on-train/16_53-jtr-res-1280-len-6-val/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_6-jtr-res_1280,_eval_,vid_det,ipsc-16_53,len-6,vstrd-6,batch-4,save-vis-1,dbg-0,dyn-1
<a id="on_54_126___16_53_jtr_res_1280_len_6_val_resnet_64_0_"></a>
### on-54_126       @ 16_53-jtr-res-1280-len-6-val/resnet-640-->p2s_vid
<a id="strd_1___on_54_126_16_53_jtr_res_1280_len_6_val_resnet_64_0_"></a>
#### strd-1       @ on-54_126/16_53-jtr-res-1280-len-6-val/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_6-jtr-res_1280,_eval_,vid_det,ipsc-54_126,len-6,vstrd-1,batch-4,save-vis-1,dbg-0,dyn-1
<a id="strd_6___on_54_126_16_53_jtr_res_1280_len_6_val_resnet_64_0_"></a>
#### strd-6       @ on-54_126/16_53-jtr-res-1280-len-6-val/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_6-jtr-res_1280,_eval_,vid_det,ipsc-54_126,len-6,vstrd-6,batch-4,save-vis-1,dbg-0,dyn-1

<a id="acamp___resnet_640_"></a>
## acamp       @ resnet-640-->p2s_vid

<a id="1k8_vid_entire_seq___acamp_resnet_640_"></a>
### 1k8_vid_entire_seq       @ acamp/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,acamp-1k8_vid_entire_seq,batch-18,dbg-0,dyn-1,dist-1
<a id="on_train___acamp_resnet_640_"></a>
### on-train       @ acamp/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_1k8_vid_entire_seq-length-2-stride-1-batch_18,_eval_,vid_det,acamp-1k8_vid_entire_seq,vstrd-1,batch-2,save-vis-1,dbg-0,dyn-1
<a id="on_inv___acamp_resnet_640_"></a>
### on-inv       @ acamp/resnet-640-->p2s_vid
<a id="vstrd_1___on_inv_acamp_resnet_64_0_"></a>
#### vstrd-1       @ on-inv/acamp/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_1k8_vid_entire_seq-length-2-stride-1-batch_18,_eval_,vid_det,acamp-10k6_vid_entire_seq_inv,vstrd-1,batch-2,save-vis-1,dbg-0,dyn-1
<a id="vstrd_2___on_inv_acamp_resnet_64_0_"></a>
#### vstrd-2       @ on-inv/acamp/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_1k8_vid_entire_seq-length-2-stride-1-batch_18,_eval_,vid_det,acamp-10k6_vid_entire_seq_inv,vstrd-2,batch-2,save-vis-1,dbg-0,dyn-1

<a id="1k8_vid_entire_seq_jtr_res_1280___acamp_resnet_640_"></a>
### 1k8_vid_entire_seq-jtr-res-1280       @ acamp/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,acamp-1k8_vid_entire_seq,batch-18,dbg-0,dyn-1,dist-1,jtr,res-1280

<a id="10k6_vid_entire_seq___resnet_640_"></a>
## 10k6_vid_entire_seq       @ resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,acamp-10k6_vid_entire_seq,batch-18,dbg-0,dyn-1,dist-1

<a id="10k6_vid_entire_seq_jtr_res_1280___resnet_640_"></a>
## 10k6_vid_entire_seq-jtr-res-1280       @ resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,acamp-10k6_vid_entire_seq,batch-18,dbg-0,dyn-1,dist-1,jtr,res-1280



