<!-- MarkdownTOC -->

- [resnet-640-16_53](#resnet_640_16_5_3_)
    - [r-2560-p-640       @ resnet-640-16_53](#r_2560_p_640___resnet_640_16_53_)
        - [sub-8       @ r-2560-p-640/resnet-640-16_53](#sub_8___r_2560_p_640_resnet_640_16_5_3_)
        - [sub-8-tac       @ r-2560-p-640/resnet-640-16_53](#sub_8_tac___r_2560_p_640_resnet_640_16_5_3_)
        - [sub-8-ltac       @ r-2560-p-640/resnet-640-16_53](#sub_8_ltac___r_2560_p_640_resnet_640_16_5_3_)
        - [sub-8-mc       @ r-2560-p-640/resnet-640-16_53](#sub_8_mc___r_2560_p_640_resnet_640_16_5_3_)
        - [sub-8-lac       @ r-2560-p-640/resnet-640-16_53](#sub_8_lac___r_2560_p_640_resnet_640_16_5_3_)
        - [sub-8-mc-tac       @ r-2560-p-640/resnet-640-16_53](#sub_8_mc_tac___r_2560_p_640_resnet_640_16_5_3_)
        - [sub-8-mc-ltac       @ r-2560-p-640/resnet-640-16_53](#sub_8_mc_ltac___r_2560_p_640_resnet_640_16_5_3_)
            - [on-train       @ sub-8-mc-ltac/r-2560-p-640/resnet-640-16_53](#on_train___sub_8_mc_ltac_r_2560_p_640_resnet_640_16_5_3_)
            - [on-54_126       @ sub-8-mc-ltac/r-2560-p-640/resnet-640-16_53](#on_54_126___sub_8_mc_ltac_r_2560_p_640_resnet_640_16_5_3_)
        - [sub-8-mc-ltac-seq1k       @ r-2560-p-640/resnet-640-16_53](#sub_8_mc_ltac_seq1k___r_2560_p_640_resnet_640_16_5_3_)
        - [sub-8-aug       @ r-2560-p-640/resnet-640-16_53](#sub_8_aug___r_2560_p_640_resnet_640_16_5_3_)
        - [sub-4-aug       @ r-2560-p-640/resnet-640-16_53](#sub_4_aug___r_2560_p_640_resnet_640_16_5_3_)
        - [sub-8-mc-len-8       @ r-2560-p-640/resnet-640-16_53](#sub_8_mc_len_8___r_2560_p_640_resnet_640_16_5_3_)

<!-- /MarkdownTOC -->
<a id="resnet_640_16_5_3_"></a>
# resnet-640-16_53
<a id="r_2560_p_640___resnet_640_16_53_"></a>
## r-2560-p-640       @ resnet-640-16_53-->p2s_vid_seg
<a id="sub_8___r_2560_p_640_resnet_640_16_5_3_"></a>
### sub-8       @ r-2560-p-640/resnet-640-16_53-->p2s_vid_seg
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:sub-8,batch-16,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,voc15
`dbg`
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-0_1:p-640:r-2560:sub-8:seq-0,batch-3,dbg-1,dyn-1,dist-0,ep-10000,gz,pt-1,voc15
<a id="sub_8_tac___r_2560_p_640_resnet_640_16_5_3_"></a>
### sub-8-tac       @ r-2560-p-640/resnet-640-16_53-->p2s_vid_seg
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:sub-8,batch-16,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,tac
`dbg`
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-0_1:p-640:r-2560:sub-8:seq-0,batch-3,dbg-1,dyn-1,dist-0,ep-10000,gz,pt-1,tac
<a id="sub_8_ltac___r_2560_p_640_resnet_640_16_5_3_"></a>
### sub-8-ltac       @ r-2560-p-640/resnet-640-16_53-->p2s_vid_seg
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:sub-8,batch-16,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,ltac
`dbg`
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:sub-8,batch-2,dbg-1,dyn-1,dist-0,ep-10000,gz,pt-1,ltac

python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg_160-0_1:p-640:r-2560:sub-8:seq-0,batch-3,dbg-1,dyn-1,dist-0,ep-10000,gz,pt-1,ltac

<a id="sub_8_mc___r_2560_p_640_resnet_640_16_5_3_"></a>
### sub-8-mc       @ r-2560-p-640/resnet-640-16_53-->p2s_vid_seg
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:sub-8,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,mc,voc15
`dbg`
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-0_1:p-640:r-2560:sub-8:seq-0,batch-2,dbg-1,dyn-1,dist-0,ep-10000,gz,pt-1,mc,voc15

<a id="sub_8_lac___r_2560_p_640_resnet_640_16_5_3_"></a>
### sub-8-lac       @ r-2560-p-640/resnet-640-16_53-->p2s_vid_seg
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:sub-8,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,mc,voc15,lac
`dbg`
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-0_1:p-640:r-2560:sub-8:seq-0,batch-3,dbg-1,dyn-1,dist-0,ep-10000,gz,pt-1,mc,lac,voc15
<a id="sub_8_mc_tac___r_2560_p_640_resnet_640_16_5_3_"></a>
### sub-8-mc-tac       @ r-2560-p-640/resnet-640-16_53-->p2s_vid_seg
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:sub-8,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,mc,tac
`dbg`
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-0_1:p-640:r-2560:sub-8:seq-0,batch-3,dbg-1,dyn-1,dist-0,ep-10000,gz,pt-1,mc,tac
<a id="sub_8_mc_ltac___r_2560_p_640_resnet_640_16_5_3_"></a>
### sub-8-mc-ltac       @ r-2560-p-640/resnet-640-16_53-->p2s_vid_seg
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:sub-8,batch-16,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,mc,ltac
`dbg`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:sub-8,batch-1,dbg-1,dyn-1,dist-0,ep-10000,gz,pt-1,mc,ltac
<a id="on_train___sub_8_mc_ltac_r_2560_p_640_resnet_640_16_5_3_"></a>
#### on-train       @ sub-8-mc-ltac/r-2560-p-640/resnet-640-16_53-->p2s_vid_seg
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-2-stride-1-sub_8-ltac-mc-batch_16,_eval_,batch-2,save-vis-1,dbg-2,dyn-1,vid_seg-16_53:p-640:r-2560:sub-8
<a id="on_54_126___sub_8_mc_ltac_r_2560_p_640_resnet_640_16_5_3_"></a>
#### on-54_126       @ sub-8-mc-ltac/r-2560-p-640/resnet-640-16_53-->p2s_vid_seg
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_2560-16_53-640_640-640_640-length-2-stride-1-sub_8-ltac-mc-batch_16,_eval_,batch-16,save-vis-1,dbg-0,dyn-1,vid_seg-54_126:p-640:r-2560:sub-8

<a id="sub_8_mc_ltac_seq1k___r_2560_p_640_resnet_640_16_5_3_"></a>
### sub-8-mc-ltac-seq1k       @ r-2560-p-640/resnet-640-16_53-->p2s_vid_seg
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:sub-8,batch-8,dbg-0,dyn-1,dist-0,ep-10000,gz,pt-1,mc,ltac,seq1k

<a id="sub_8_aug___r_2560_p_640_resnet_640_16_5_3_"></a>
### sub-8-aug       @ r-2560-p-640/resnet-640-16_53-->p2s_vid_seg
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:rot-15_345_4:sub-8,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1
`seq-0`
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:rot-15_345_4:seq-0:sub-8,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1

<a id="sub_4_aug___r_2560_p_640_resnet_640_16_5_3_"></a>
### sub-4-aug       @ r-2560-p-640/resnet-640-16_53-->p2s_vid_seg
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,seg_160-16_53:p-640:r-2560:rot-15_345_4:sub-4,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1

<a id="sub_8_mc_len_8___r_2560_p_640_resnet_640_16_5_3_"></a>
### sub-8-mc-len-8       @ r-2560-p-640/resnet-640-16_53-->p2s_vid_seg
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:sub-8,batch-4,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,mc,tac,voc15,seq3k,len-8


