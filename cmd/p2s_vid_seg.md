<!-- MarkdownTOC -->

- [resnet-640-16_53](#resnet_640_16_5_3_)
    - [r-2560       @ resnet-640-16_53](#r_2560___resnet_640_16_53_)
        - [p-640-sub-8       @ r-2560/resnet-640-16_53](#p_640_sub_8___r_2560_resnet_640_16_5_3_)
        - [p-640-sub-8-mc       @ r-2560/resnet-640-16_53](#p_640_sub_8_mc___r_2560_resnet_640_16_5_3_)
        - [p-640-aug-sub-8       @ r-2560/resnet-640-16_53](#p_640_aug_sub_8___r_2560_resnet_640_16_5_3_)
        - [p-640-aug-sub-4       @ r-2560/resnet-640-16_53](#p_640_aug_sub_4___r_2560_resnet_640_16_5_3_)

<!-- /MarkdownTOC -->
<a id="resnet_640_16_5_3_"></a>
# resnet-640-16_53
<a id="r_2560___resnet_640_16_53_"></a>
## r-2560       @ resnet-640-16_53-->p2s_vid_seg
<a id="p_640_sub_8___r_2560_resnet_640_16_5_3_"></a>
### p-640-sub-8       @ r-2560/resnet-640-16_53-->p2s_vid_seg
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:sub-8,batch-2,dbg-1,dyn-1,dist-0,ep-10000,gz,pt-1
`dbg`
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-0_1:p-640:r-2560:sub-8:seq-0,batch-3,dbg-1,dyn-1,dist-0,ep-10000,gz,pt-1,voc-15

python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-0_1:p-640:r-2560:sub-8:seq-0,batch-3,dbg-1,dyn-1,dist-0,ep-10000,gz,pt-1,tac,voc-8

<a id="p_640_sub_8_mc___r_2560_resnet_640_16_5_3_"></a>
### p-640-sub-8-mc       @ r-2560/resnet-640-16_53-->p2s_vid_seg
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:sub-8,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,mc,voc14
`dbg`
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-0_1:p-640:r-2560:sub-8:seq-0,batch-2,dbg-1,dyn-1,dist-0,ep-10000,gz,pt-1,mc,voc14
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-0_1:p-640:r-2560:sub-8:seq-0,batch-3,dbg-1,dyn-1,dist-0,ep-10000,gz,pt-1,mc,lac
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-0_1:p-640:r-2560:sub-8:seq-0,batch-3,dbg-1,dyn-1,dist-0,ep-10000,gz,pt-1,mc,tac
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-0_1:p-640:r-2560:sub-8:seq-0,batch-3,dbg-1,dyn-1,dist-0,ep-10000,gz,pt-1,mc,ltac

<a id="p_640_aug_sub_8___r_2560_resnet_640_16_5_3_"></a>
### p-640-aug-sub-8       @ r-2560/resnet-640-16_53-->p2s_vid_seg
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:rot-15_345_4:sub-8,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1
`seq-0`
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,vid_seg-16_53:p-640:r-2560:rot-15_345_4:seq-0:sub-8,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1

<a id="p_640_aug_sub_4___r_2560_resnet_640_16_5_3_"></a>
### p-640-aug-sub-4       @ r-2560/resnet-640-16_53-->p2s_vid_seg
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,seg_160-16_53:p-640:r-2560:rot-15_345_4:sub-4,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1


