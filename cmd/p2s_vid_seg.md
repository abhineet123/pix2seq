<!-- MarkdownTOC -->

- [resnet-640-16_53](#resnet_640_16_5_3_)
    - [res-2560       @ resnet-640-16_53](#res_2560___resnet_640_16_53_)
        - [sz-640-sub-8       @ res-2560/resnet-640-16_53](#sz_640_sub_8___res_2560_resnet_640_16_5_3_)
        - [sz-640-sub-8-mc       @ res-2560/resnet-640-16_53](#sz_640_sub_8_mc___res_2560_resnet_640_16_5_3_)
        - [sz-640-aug-sub-8       @ res-2560/resnet-640-16_53](#sz_640_aug_sub_8___res_2560_resnet_640_16_5_3_)
        - [sz-640-aug-sub-4       @ res-2560/resnet-640-16_53](#sz_640_aug_sub_4___res_2560_resnet_640_16_5_3_)
    - [res-640       @ resnet-640-16_53](#res_640___resnet_640_16_53_)
        - [sz-80       @ res-640/resnet-640-16_53](#sz_80___res_640_resnet_640_16_53_)
            - [on-train       @ sz-80/res-640/resnet-640-16_53](#on_train___sz_80_res_640_resnet_640_16_53_)
            - [on-54_126       @ sz-80/res-640/resnet-640-16_53](#on_54_126___sz_80_res_640_resnet_640_16_53_)
        - [sz-80-mc       @ res-640/resnet-640-16_53](#sz_80_mc___res_640_resnet_640_16_53_)

<!-- /MarkdownTOC -->
<a id="resnet_640_16_5_3_"></a>
# resnet-640-16_53
<a id="res_2560___resnet_640_16_53_"></a>
## res-2560       @ resnet-640-16_53-->p2s_vid_seg
<a id="sz_640_sub_8___res_2560_resnet_640_16_5_3_"></a>
### sz-640-sub-8       @ res-2560/resnet-640-16_53-->p2s_vid_seg
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,seg-16_53:psz-640:rsz-2560:sub-8,batch-24,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1
`seq-0`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,seg-16_53:psz-640:rsz-2560:sub-8:seq-0,batch-8,dbg-0,dyn-1,dist-0,ep-10000,gz,pt-1

<a id="sz_640_sub_8_mc___res_2560_resnet_640_16_5_3_"></a>
### sz-640-sub-8-mc       @ res-2560/resnet-640-16_53-->p2s_vid_seg
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,seg-16_53:psz-640:rsz-2560:sub-8,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,mc

<a id="sz_640_aug_sub_8___res_2560_resnet_640_16_5_3_"></a>
### sz-640-aug-sub-8       @ res-2560/resnet-640-16_53-->p2s_vid_seg
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,seg-16_53:psz-640:rsz-2560:rot-15_345_4:sub-8,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1
`seq-0`
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,seg-16_53:psz-640:rsz-2560:rot-15_345_4:seq-0:sub-8,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1

<a id="sz_640_aug_sub_4___res_2560_resnet_640_16_5_3_"></a>
### sz-640-aug-sub-4       @ res-2560/resnet-640-16_53-->p2s_vid_seg
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,seg_160-16_53:psz-640:rsz-2560:rot-15_345_4:sub-4,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1

<a id="res_640___resnet_640_16_53_"></a>
## res-640       @ resnet-640-16_53-->p2s_vid_seg
<a id="sz_80___res_640_resnet_640_16_53_"></a>
### sz-80       @ res-640/resnet-640-16_53-->p2s_vid_seg
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,seg-16_53:psz-80:rsz-640,batch-36,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1
<a id="on_train___sz_80_res_640_resnet_640_16_53_"></a>
#### on-train       @ sz-80/res-640/resnet-640-16_53-->p2s_vid_seg
python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_640-16_53-80_80-80_80-batch_36,_eval_,batch-18,save-vis-1,dbg-0,dyn-1,seg-16_53:psz-80:rsz-640
<a id="on_54_126___sz_80_res_640_resnet_640_16_53_"></a>
#### on-54_126       @ sz-80/res-640/resnet-640-16_53-->p2s_vid_seg
python3 run.py --cfg=configs/config_video_seg.py  --j5=m-resnet_640_resize_640-16_53-80_80-80_80-batch_36,_eval_,batch-18,save-vis-1,dbg-0,dyn-1,seg-54_126:psz-80:rsz-640

<a id="sz_80_mc___res_640_resnet_640_16_53_"></a>
### sz-80-mc       @ res-640/resnet-640-16_53-->p2s_vid_seg
python3 run.py --cfg=configs/config_video_seg.py  --j5=train,resnet-640,seg-16_53:psz-80:rsz-640,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,mc

