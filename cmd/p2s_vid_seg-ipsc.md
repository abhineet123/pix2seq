<!-- MarkdownTOC -->

- [16_53-r-2560-p-640-sub-8](#16_53_r_2560_p_640_sub_8_)
    - [len-2       @ 16_53-r-2560-p-640-sub-8](#len_2___16_53_r_2560_p_640_sub_8_)
        - [ltac       @ len-2/16_53-r-2560-p-640-sub-8](#ltac___len_2_16_53_r_2560_p_640_sub_8_)
            - [on-train       @ ltac/len-2/16_53-r-2560-p-640-sub-8](#on_train___ltac_len_2_16_53_r_2560_p_640_sub_8_)
            - [on-54_126       @ ltac/len-2/16_53-r-2560-p-640-sub-8](#on_54_126___ltac_len_2_16_53_r_2560_p_640_sub_8_)
        - [mc       @ len-2/16_53-r-2560-p-640-sub-8](#mc___len_2_16_53_r_2560_p_640_sub_8_)
        - [mc-lac       @ len-2/16_53-r-2560-p-640-sub-8](#mc_lac___len_2_16_53_r_2560_p_640_sub_8_)
            - [on-train       @ mc-lac/len-2/16_53-r-2560-p-640-sub-8](#on_train___mc_lac_len_2_16_53_r_2560_p_640_sub_8_)
            - [on-54_126       @ mc-lac/len-2/16_53-r-2560-p-640-sub-8](#on_54_126___mc_lac_len_2_16_53_r_2560_p_640_sub_8_)
        - [mc-tac       @ len-2/16_53-r-2560-p-640-sub-8](#mc_tac___len_2_16_53_r_2560_p_640_sub_8_)
            - [on-train       @ mc-tac/len-2/16_53-r-2560-p-640-sub-8](#on_train___mc_tac_len_2_16_53_r_2560_p_640_sub_8_)
            - [on-54_126       @ mc-tac/len-2/16_53-r-2560-p-640-sub-8](#on_54_126___mc_tac_len_2_16_53_r_2560_p_640_sub_8_)
        - [mc-ltac       @ len-2/16_53-r-2560-p-640-sub-8](#mc_ltac___len_2_16_53_r_2560_p_640_sub_8_)
            - [on-train       @ mc-ltac/len-2/16_53-r-2560-p-640-sub-8](#on_train___mc_ltac_len_2_16_53_r_2560_p_640_sub_8_)
            - [on-54_126       @ mc-ltac/len-2/16_53-r-2560-p-640-sub-8](#on_54_126___mc_ltac_len_2_16_53_r_2560_p_640_sub_8_)
        - [mc-ltac-seq1k       @ len-2/16_53-r-2560-p-640-sub-8](#mc_ltac_seq1k___len_2_16_53_r_2560_p_640_sub_8_)
            - [on-train       @ mc-ltac-seq1k/len-2/16_53-r-2560-p-640-sub-8](#on_train___mc_ltac_seq1k_len_2_16_53_r_2560_p_640_sub_8_)
            - [on-54_126       @ mc-ltac-seq1k/len-2/16_53-r-2560-p-640-sub-8](#on_54_126___mc_ltac_seq1k_len_2_16_53_r_2560_p_640_sub_8_)
    - [len-4       @ 16_53-r-2560-p-640-sub-8](#len_4___16_53_r_2560_p_640_sub_8_)
        - [mc-ltac       @ len-4/16_53-r-2560-p-640-sub-8](#mc_ltac___len_4_16_53_r_2560_p_640_sub_8_)
            - [on-train       @ mc-ltac/len-4/16_53-r-2560-p-640-sub-8](#on_train___mc_ltac_len_4_16_53_r_2560_p_640_sub_8_)
            - [on-54_126       @ mc-ltac/len-4/16_53-r-2560-p-640-sub-8](#on_54_126___mc_ltac_len_4_16_53_r_2560_p_640_sub_8_)
    - [len-8       @ 16_53-r-2560-p-640-sub-8](#len_8___16_53_r_2560_p_640_sub_8_)
        - [mc-tac-buggy       @ len-8/16_53-r-2560-p-640-sub-8](#mc_tac_buggy___len_8_16_53_r_2560_p_640_sub_8_)
            - [on-train       @ mc-tac-buggy/len-8/16_53-r-2560-p-640-sub-8](#on_train___mc_tac_buggy_len_8_16_53_r_2560_p_640_sub_8_)
            - [on-54_126       @ mc-tac-buggy/len-8/16_53-r-2560-p-640-sub-8](#on_54_126___mc_tac_buggy_len_8_16_53_r_2560_p_640_sub_8_)
                - [vstrd-8       @ on-54_126/mc-tac-buggy/len-8/16_53-r-2560-p-640-sub-8](#vstrd_8___on_54_126_mc_tac_buggy_len_8_16_53_r_2560_p_640_sub_8_)
        - [mc-tac       @ len-8/16_53-r-2560-p-640-sub-8](#mc_tac___len_8_16_53_r_2560_p_640_sub_8_)
            - [on-train       @ mc-tac/len-8/16_53-r-2560-p-640-sub-8](#on_train___mc_tac_len_8_16_53_r_2560_p_640_sub_8_)
            - [on-train-vstrd-8       @ mc-tac/len-8/16_53-r-2560-p-640-sub-8](#on_train_vstrd_8___mc_tac_len_8_16_53_r_2560_p_640_sub_8_)
            - [on-54_126       @ mc-tac/len-8/16_53-r-2560-p-640-sub-8](#on_54_126___mc_tac_len_8_16_53_r_2560_p_640_sub_8_)
            - [on-54_126-vstrd-4       @ mc-tac/len-8/16_53-r-2560-p-640-sub-8](#on_54_126_vstrd_4___mc_tac_len_8_16_53_r_2560_p_640_sub_8_)
            - [on-54_126-vstrd-8       @ mc-tac/len-8/16_53-r-2560-p-640-sub-8](#on_54_126_vstrd_8___mc_tac_len_8_16_53_r_2560_p_640_sub_8_)

<!-- /MarkdownTOC -->
<a id="16_53_r_2560_p_640_sub_8_"></a>
# 16_53-r-2560-p-640-sub-8


<a id="len_2___16_53_r_2560_p_640_sub_8_"></a>
## len-2       @ 16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc


<a id="ltac___len_2_16_53_r_2560_p_640_sub_8_"></a>
### ltac       @ len-2/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:sub-8,batch-16,dbg-0,dyn-1,dist-1,pt-1,ltac
`dbg`
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:sub-8,batch-2,dbg-1,dyn-1,dist-0,pt-1,ltac
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg_160-0_1:p-640:r-2560:sub-8:seq-0,batch-3,dbg-1,dyn-1,dist-0,pt-1,ltac
<a id="on_train___ltac_len_2_16_53_r_2560_p_640_sub_8_"></a>
#### on-train       @ ltac/len-2/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-2-stride-1-sub_8-ltac-batch_16,_eval_,batch-4,save-vis-1,dbg-0,dyn-1,vid_seg-16_53:p-640:r-2560:sub-8,ltac
<a id="on_54_126___ltac_len_2_16_53_r_2560_p_640_sub_8_"></a>
#### on-54_126       @ ltac/len-2/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-2-stride-1-sub_8-ltac-batch_16,_eval_,batch-4,save-vis-1,dbg-0,dyn-1,vid_seg-54_126:p-640:r-2560:sub-8,ltac,asi


<a id="mc___len_2_16_53_r_2560_p_640_sub_8_"></a>
### mc       @ len-2/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:sub-8,batch-8,dbg-0,dyn-1,dist-1,pt-1,mc,voc15,seq1k
`dbg`
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:sub-8,batch-3,dbg-1,dyn-1,dist-0,pt-1,mc,voc15,seq1k


<a id="mc_lac___len_2_16_53_r_2560_p_640_sub_8_"></a>
### mc-lac       @ len-2/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:sub-8,batch-8,dbg-0,dyn-1,dist-0,pt-1,mc,lac,voc15,seq1k
<a id="on_train___mc_lac_len_2_16_53_r_2560_p_640_sub_8_"></a>
#### on-train       @ mc-lac/len-2/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-2-stride-1-sub_8-lac-mc-batch_8-seq1k,_eval_,vid_seg-16_53:p-640:r-2560:sub-8,batch-3,save-vis-1,dbg-0,dyn-1,mc,lac,voc15,seq1k
<a id="on_54_126___mc_lac_len_2_16_53_r_2560_p_640_sub_8_"></a>
#### on-54_126       @ mc-lac/len-2/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-2-stride-1-sub_8-lac-mc-batch_8-seq1k,_eval_,vid_seg-54_126:p-640:r-2560:sub-8,batch-4,save-vis-1,dbg-0,dyn-1,mc,lac,voc15,seq1k,asi


<a id="mc_tac___len_2_16_53_r_2560_p_640_sub_8_"></a>
### mc-tac       @ len-2/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:sub-8,batch-8,dbg-0,dyn-1,dist-0,pt-1,mc,tac,seq1k
<a id="on_train___mc_tac_len_2_16_53_r_2560_p_640_sub_8_"></a>
#### on-train       @ mc-tac/len-2/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-2-stride-1-sub_8-tac-mc-batch_8-seq1k,_eval_,vid_seg-16_53:p-640:r-2560:sub-8,batch-3,save-vis-1,dbg-0,dyn-1,mc,tac,seq1k
<a id="on_54_126___mc_tac_len_2_16_53_r_2560_p_640_sub_8_"></a>
#### on-54_126       @ mc-tac/len-2/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-2-stride-1-sub_8-tac-mc-batch_8-seq1k,_eval_,vid_seg-54_126:p-640:r-2560:sub-8,batch-3,save-vis-1,dbg-0,dyn-1,mc,tac,seq1k,asi


<a id="mc_ltac___len_2_16_53_r_2560_p_640_sub_8_"></a>
### mc-ltac       @ len-2/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:sub-8,batch-16,dbg-0,dyn-1,dist-1,pt-1,mc,ltac
`dbg`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-frame-0_2:seq-0_2:p-640:r-2560:sub-8,batch-8,dbg-0,dyn-1,dist-0,pt-1,mc,ltac
<a id="on_train___mc_ltac_len_2_16_53_r_2560_p_640_sub_8_"></a>
#### on-train       @ mc-ltac/len-2/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-2-stride-1-sub_8-ltac-mc-batch_16,_eval_,batch-32,save-vis-1,dbg-0,dyn-1,vid_seg-16_53:p-640:r-2560:sub-8
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-2-stride-1-sub_8-ltac-mc-batch_16,_eval_,batch-16,save-vis-1,dbg-0,dyn-1,vid_seg-16_53:p-640:r-2560:sub-8
`dbg`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-2-stride-1-sub_8-ltac-mc-batch_16,_eval_,batch-3,save-vis-1,dbg-0,dyn-1,vid_seg-0_1:p-640:r-2560:sub-8:seq-0_1,vis-0
python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-2-stride-1-sub_8-ltac-mc-batch_16,_eval_,batch-3,save-vis-1,dbg-0,dyn-1,vid_seg-frame-0_2:p-640:r-2560:sub-8:seq-0_2,vis-0
<a id="on_54_126___mc_ltac_len_2_16_53_r_2560_p_640_sub_8_"></a>
#### on-54_126       @ mc-ltac/len-2/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-2-stride-1-sub_8-ltac-mc-batch_16,_eval_,batch-3,save-vis-1,dbg-0,dyn-1,vid_seg-54_126:p-640:r-2560:sub-8,asi
`dbg`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-2-stride-1-sub_8-ltac-mc-batch_16,_eval_,batch-8,save-vis-1,dbg-1,dyn-1,vid_seg-54_126:p-640:r-2560:sub-8


<a id="mc_ltac_seq1k___len_2_16_53_r_2560_p_640_sub_8_"></a>
### mc-ltac-seq1k       @ len-2/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:sub-8,batch-8,dbg-0,dyn-1,dist-0,pt-1,mc,ltac,seq1k
<a id="on_train___mc_ltac_seq1k_len_2_16_53_r_2560_p_640_sub_8_"></a>
#### on-train       @ mc-ltac-seq1k/len-2/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-2-stride-1-sub_8-ltac-mc-batch_8-seq1k,_eval_,batch-4,save-vis-1,dbg-0,dyn-1,vid_seg-16_53:p-640:r-2560:sub-8,mc,ltac,seq1k
<a id="on_54_126___mc_ltac_seq1k_len_2_16_53_r_2560_p_640_sub_8_"></a>
#### on-54_126       @ mc-ltac-seq1k/len-2/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-2-stride-1-sub_8-ltac-mc-batch_8-seq1k,_eval_,batch-3,save-vis-1,dbg-0,dyn-1,vid_seg-54_126:p-640:r-2560:sub-8,mc,ltac,seq1k,asi


<a id="len_4___16_53_r_2560_p_640_sub_8_"></a>
## len-4       @ 16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc

<a id="mc_ltac___len_4_16_53_r_2560_p_640_sub_8_"></a>
### mc-ltac       @ len-4/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:sub-8,batch-8,dbg-0,dyn-1,dist-1,pt-1,mc,ltac,voc15,seq1k,len-4
<a id="on_train___mc_ltac_len_4_16_53_r_2560_p_640_sub_8_"></a>
#### on-train       @ mc-ltac/len-4/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-4-stride-1-sub_8-ltac-mc-batch_8-seq1k,_eval_,batch-12,save-vis-1,dbg-0,dyn-1,vid_seg-16_53:p-640:r-2560:sub-8,mc,ltac,voc15,seq1k,len-4
<a id="on_54_126___mc_ltac_len_4_16_53_r_2560_p_640_sub_8_"></a>
#### on-54_126       @ mc-ltac/len-4/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-4-stride-1-sub_8-ltac-mc-batch_8-seq1k,_eval_,batch-12,save-vis-1,dbg-0,dyn-1,vid_seg-54_126:p-640:r-2560:sub-8,mc,ltac,voc15,seq1k,len-4,asi


<a id="len_8___16_53_r_2560_p_640_sub_8_"></a>
## len-8       @ 16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc


<a id="mc_tac_buggy___len_8_16_53_r_2560_p_640_sub_8_"></a>
### mc-tac-buggy       @ len-8/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
<a id="on_train___mc_tac_buggy_len_8_16_53_r_2560_p_640_sub_8_"></a>
#### on-train       @ mc-tac-buggy/len-8/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-8-stride-1-sub_8-tac-mc-batch_4-seq3k,_eval_,batch-6,save-vis-1,dbg-0,dyn-1,vid_seg-16_53:p-640:r-2560:sub-8:ofj-0:ovl,mc,tac,voc15,seq3k,len-8,vis-0
`dbg`
python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-8-stride-1-sub_8-tac-mc-batch_4-seq3k,_eval_,batch-8,save-vis-1,dbg-0,dyn-1,vid_seg-frame-0_7:seq-0_1:p-640:r-2560:sub-8,mc,tac,voc15,seq3k,len-8
<a id="on_54_126___mc_tac_buggy_len_8_16_53_r_2560_p_640_sub_8_"></a>
#### on-54_126       @ mc-tac-buggy/len-8/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-8-stride-1-sub_8-tac-mc-batch_4-seq3k,_eval_,batch-8,save-vis-1,dbg-0,dyn-1,vid_seg-54_126:p-640:r-2560:sub-8:ofj-0:ovl,mc,tac,voc15,seq3k,len-8,vis-0,no_gt
`seq-0`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-8-stride-1-sub_8-tac-mc-batch_4-seq3k,_eval_,batch-8,save-vis-1,dbg-0,dyn-1,vid_seg-frame-54_126:seq-0:p-640:r-2560:sub-8:ofj-0:ovl,mc,tac,voc15,seq3k,len-8,vis-0,no_gt
<a id="vstrd_8___on_54_126_mc_tac_buggy_len_8_16_53_r_2560_p_640_sub_8_"></a>
##### vstrd-8       @ on-54_126/mc-tac-buggy/len-8/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-8-stride-1-sub_8-tac-mc-batch_4-seq3k,_eval_,batch-12,save-vis-1,dbg-0,dyn-1,vid_seg-frame-54_126:p-640:r-2560:sub-8:ofj-0:ovl,mc,tac,voc15,seq3k,len-8,vstrd-8,vis-0,no_gt
`seq-0`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-8-stride-1-sub_8-tac-mc-batch_4-seq3k,_eval_,batch-6,save-vis-1,dbg-0,dyn-1,vid_seg-frame-54_126:seq-0:p-640:r-2560:sub-8:ofj-0:ovl,mc,tac,voc15,seq3k,len-8,vstrd-8,vis-0,no_gt


<a id="mc_tac___len_8_16_53_r_2560_p_640_sub_8_"></a>
### mc-tac       @ len-8/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:sub-8,batch-4,dbg-0,dyn-1,dist-1,pt-1,mc,tac,voc15,seq3k,len-8
<a id="on_train___mc_tac_len_8_16_53_r_2560_p_640_sub_8_"></a>
#### on-train       @ mc-tac/len-8/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-8-stride-1-sub_8-tac-mc-batch_4-seq3k,_eval_,batch-6,save-vis-1,dbg-0,dyn-1,vid_seg-16_53:p-640:r-2560:sub-8,mc,tac,voc15,seq3k,len-8,vis-0
<a id="on_train_vstrd_8___mc_tac_len_8_16_53_r_2560_p_640_sub_8_"></a>
#### on-train-vstrd-8       @ mc-tac/len-8/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-8-stride-1-sub_8-tac-mc-batch_4-seq3k,_eval_,batch-8,save-vis-1,dbg-0,dyn-1,vid_seg-frame-16_53:p-640:r-2560:sub-8,mc,tac,voc15,seq3k,len-8,vstrd-8,vis-0,asi-0
<a id="on_54_126___mc_tac_len_8_16_53_r_2560_p_640_sub_8_"></a>
#### on-54_126       @ mc-tac/len-8/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-8-stride-1-sub_8-tac-mc-batch_4-seq3k,_eval_,batch-16,save-vis-1,dbg-0,dyn-1,vid_seg-54_126:p-640:r-2560:sub-8,mc,tac,voc15,seq3k,len-8,asi,vis-0
`54_126:seq-0`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-8-stride-1-sub_8-tac-mc-batch_4-seq3k,_eval_,batch-6,save-vis-1,dbg-0,dyn-1,vid_seg-54_126:seq-0:p-640:r-2560:sub-8,mc,tac,voc15,seq3k,len-8,vis-0
`54_61:seq-0`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-8-stride-1-sub_8-tac-mc-batch_4-seq3k,_eval_,batch-16,save-vis-1,dbg-0,dyn-1,vid_seg-54_61:seq-0:p-640:r-2560:sub-8,mc,tac,voc15,seq3k,len-8,vis-0
<a id="on_54_126_vstrd_4___mc_tac_len_8_16_53_r_2560_p_640_sub_8_"></a>
#### on-54_126-vstrd-4       @ mc-tac/len-8/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
`54_126:seq-0`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-8-stride-1-sub_8-tac-mc-batch_4-seq3k,_eval_,batch-6,save-vis-1,dbg-0,dyn-1,vid_seg-frame-54_126:seq-0:p-640:r-2560:sub-8,mc,tac,voc15,seq3k,len-8,vstrd-4,vis-0
<a id="on_54_126_vstrd_8___mc_tac_len_8_16_53_r_2560_p_640_sub_8_"></a>
#### on-54_126-vstrd-8       @ mc-tac/len-8/16_53-r-2560-p-640-sub-8-->p2s_vid_seg-ipsc
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-8-stride-1-sub_8-tac-mc-batch_4-seq3k,_eval_,batch-16,save-vis-1,dbg-0,dyn-1,vid_seg-frame-54_126:p-640:r-2560:sub-8,mc,tac,voc15,seq3k,len-8,vstrd-8,vis-0
`54_126:seq-0`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-8-stride-1-sub_8-tac-mc-batch_4-seq3k,_eval_,batch-3,save-vis-1,dbg-0,dyn-1,vid_seg-frame-54_126:seq-0:p-640:r-2560:sub-8,mc,tac,voc15,seq3k,len-8,vstrd-8,vis-0


