<!-- MarkdownTOC -->

- [0_31](#0_3_1_)
    - [orig-p-640-sub-8-aug-lac       @ 0_31](#orig_p_640_sub_8_aug_lac___0_31_)
    - [r-640       @ 0_31](#r_640___0_31_)
        - [p-640-sub-4-aug-lac       @ r-640/0_31](#p_640_sub_4_aug_lac___r_640_0_31_)

<!-- /MarkdownTOC -->
<a id="0_3_1_"></a>
# 0_31
<a id="orig_p_640_sub_8_aug_lac___0_31_"></a>
## orig-p-640-sub-8-aug-lac       @ 0_31-->p2s_seg-617
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,617,seg-0_31:p-640:sub-8:rot-15_345_8:strd-64_256:flip,batch-28,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,lac,seq1k

<a id="r_640___0_31_"></a>
## r-640       @ 0_31-->p2s_seg-617
<a id="p_640_sub_4_aug_lac___r_640_0_31_"></a>
### p-640-sub-4-aug-lac       @ r-640/0_31-->p2s_seg-617
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,617,seg-0_31:r-640:p-640:sub-4:rot-15_345_16:strd-64_256:flip,batch-12,dbg-0,dyn-1,dist-0,ep-10000,gz,pt-1,lac,seq3k,voc28

