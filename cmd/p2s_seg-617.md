<!-- MarkdownTOC -->

- [0_31](#0_3_1_)
    - [orig-p-640-sub-8-aug-lac       @ 0_31](#orig_p_640_sub_8_aug_lac___0_31_)
            - [on-train       @ orig-p-640-sub-8-aug-lac/0_31](#on_train___orig_p_640_sub_8_aug_lac_0_3_1_)
            - [on-32_49       @ orig-p-640-sub-8-aug-lac/0_31](#on_32_49___orig_p_640_sub_8_aug_lac_0_3_1_)
    - [r-640       @ 0_31](#r_640___0_31_)
        - [p-640-sub-4-aug-lac       @ r-640/0_31](#p_640_sub_4_aug_lac___r_640_0_31_)
            - [on-train       @ p-640-sub-4-aug-lac/r-640/0_31](#on_train___p_640_sub_4_aug_lac_r_640_0_31_)
            - [on-32_49       @ p-640-sub-4-aug-lac/r-640/0_31](#on_32_49___p_640_sub_4_aug_lac_r_640_0_31_)

<!-- /MarkdownTOC -->
<a id="0_3_1_"></a>
# 0_31
<a id="orig_p_640_sub_8_aug_lac___0_31_"></a>
## orig-p-640-sub-8-aug-lac       @ 0_31-->p2s_seg-617
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,617,seg-0_31:p-640:sub-8:rot-15_345_8:strd-64_256:flip,batch-28,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,lac,seq1k
<a id="on_train___orig_p_640_sub_8_aug_lac_0_3_1_"></a>
#### on-train       @ orig-p-640-sub-8-aug-lac/0_31-->p2s_seg-617
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_0_31-640_640-64_256-rot_15_345_8-flip-sub_8-lac-617-batch_28-seq1k,_eval_,batch-16,save-vis-1,dbg-0,dyn-1,617,seg-0_31:p-640:sub-8,lac,seq1k
<a id="on_32_49___orig_p_640_sub_8_aug_lac_0_3_1_"></a>
#### on-32_49       @ orig-p-640-sub-8-aug-lac/0_31-->p2s_seg-617
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_0_31-640_640-64_256-rot_15_345_8-flip-sub_8-lac-617-batch_28-seq1k,_eval_,batch-16,save-vis-1,dbg-0,dyn-1,617,seg-32_49:p-640:sub-8,lac,seq1k


<a id="r_640___0_31_"></a>
## r-640       @ 0_31-->p2s_seg-617
<a id="p_640_sub_4_aug_lac___r_640_0_31_"></a>
### p-640-sub-4-aug-lac       @ r-640/0_31-->p2s_seg-617
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,617,seg-0_31:r-640:p-640:sub-4:rot-15_345_16:strd-64_256:flip,batch-6,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,lac,seq3k,voc28
<a id="on_train___p_640_sub_4_aug_lac_r_640_0_31_"></a>
#### on-train       @ p-640-sub-4-aug-lac/r-640/0_31-->p2s_seg-617
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_640-0_31-640_640-64_256-rot_15_345_16-flip-sub_4-lac-617-batch_6-seq3k,_eval_,batch-8,save-vis-1,dbg-0,dyn-1,617,seg-0_31:r-640:p-640:sub-4,lac,seq3k,voc28
<a id="on_32_49___p_640_sub_4_aug_lac_r_640_0_31_"></a>
#### on-32_49       @ p-640-sub-4-aug-lac/r-640/0_31-->p2s_seg-617
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_640-0_31-640_640-64_256-rot_15_345_16-flip-sub_4-lac-617-batch_6-seq3k,_eval_,batch-8,save-vis-1,dbg-0,dyn-1,617,seg-32_49:r-640:p-640:sub-4,lac,seq3k,voc28
