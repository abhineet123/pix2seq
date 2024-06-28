<!-- MarkdownTOC -->

- [0_31       @ ipsc](#0_31___ipsc_)
    - [no-resize       @ 0_31](#no_resize___0_31_)
        - [p-640-sub-8       @ no-resize/0_31](#p_640_sub_8___no_resize_0_31_)
    - [r-2560       @ 0_31](#r_2560___0_31_)
        - [p-640-sub-8       @ r-2560/0_31](#p_640_sub_8___r_2560_0_3_1_)
        - [p-640-sub-8-mc       @ r-2560/0_31](#p_640_sub_8_mc___r_2560_0_3_1_)
- [0_49       @ ipsc](#0_49___ipsc_)
    - [r-2560-p-640-sub-8       @ 0_49](#r_2560_p_640_sub_8___0_49_)
        - [bin       @ r-2560-p-640-sub-8/0_49](#bin___r_2560_p_640_sub_8_0_4_9_)
        - [mc       @ r-2560-p-640-sub-8/0_49](#mc___r_2560_p_640_sub_8_0_4_9_)
        - [lac       @ r-2560-p-640-sub-8/0_49](#lac___r_2560_p_640_sub_8_0_4_9_)
    - [r-2560-p-640-sub-4       @ 0_49](#r_2560_p_640_sub_4___0_49_)
        - [bin       @ r-2560-p-640-sub-4/0_49](#bin___r_2560_p_640_sub_4_0_4_9_)
        - [mc       @ r-2560-p-640-sub-4/0_49](#mc___r_2560_p_640_sub_4_0_4_9_)
        - [lac       @ r-2560-p-640-sub-4/0_49](#lac___r_2560_p_640_sub_4_0_4_9_)
    - [r-640-p-640-sub-8       @ 0_49](#r_640_p_640_sub_8___0_49_)
        - [bin       @ r-640-p-640-sub-8/0_49](#bin___r_640_p_640_sub_8_0_49_)
        - [mc       @ r-640-p-640-sub-8/0_49](#mc___r_640_p_640_sub_8_0_49_)
        - [lac       @ r-640-p-640-sub-8/0_49](#lac___r_640_p_640_sub_8_0_49_)

<!-- /MarkdownTOC -->

<a id="0_31___ipsc_"></a>
# 0_31       @ ipsc-->p2s_seg_tfrecord
<a id="no_resize___0_31_"></a>
## no-resize       @ 0_31-->p2s_seg_tf-617
<a id="p_640_sub_8___no_resize_0_31_"></a>
### p-640-sub-8       @ no-resize/0_31-->p2s_seg_tf-617
python3 data/scripts/create_seg_tfrecord.py cfg=617:0_31:p-640:sub-8

<a id="r_2560___0_31_"></a>
## r-2560       @ 0_31-->p2s_seg_tf-617
<a id="p_640_sub_8___r_2560_0_3_1_"></a>
### p-640-sub-8       @ r-2560/0_31-->p2s_seg_tf-617
python3 data/scripts/create_seg_tfrecord.py cfg=617:0_31:p-640:sub-8:r-2560:json
<a id="p_640_sub_8_mc___r_2560_0_3_1_"></a>
### p-640-sub-8-mc       @ r-2560/0_31-->p2s_seg_tf-617
python3 data/scripts/create_seg_tfrecord.py cfg=617:0_31:p-640:sub-8:r-2560:mc:json

<a id="0_49___ipsc_"></a>
# 0_49       @ ipsc-->p2s_seg_tfrecord
<a id="r_2560_p_640_sub_8___0_49_"></a>
## r-2560-p-640-sub-8       @ 0_49-->p2s_seg_tf-617
<a id="bin___r_2560_p_640_sub_8_0_4_9_"></a>
### bin       @ r-2560-p-640-sub-8/0_49-->p2s_seg_tf-617
python3 data/scripts/create_seg_tfrecord.py cfg=617:0_49:p-640:sub-8:r-2560:json
<a id="mc___r_2560_p_640_sub_8_0_4_9_"></a>
### mc       @ r-2560-p-640-sub-8/0_49-->p2s_seg_tf-617
python3 data/scripts/create_seg_tfrecord.py cfg=617:0_49:p-640:sub-8:r-2560:mc:json
<a id="lac___r_2560_p_640_sub_8_0_4_9_"></a>
### lac       @ r-2560-p-640-sub-8/0_49-->p2s_seg_tf-617
python3 data/scripts/create_seg_tfrecord.py cfg=617:0_49:p-640:sub-8:r-2560:lac:json

<a id="r_2560_p_640_sub_4___0_49_"></a>
## r-2560-p-640-sub-4       @ 0_49-->p2s_seg_tf-617
<a id="bin___r_2560_p_640_sub_4_0_4_9_"></a>
### bin       @ r-2560-p-640-sub-4/0_49-->p2s_seg_tf-617
python3 data/scripts/create_seg_tfrecord.py cfg=617:0_49:p-640:sub-4:r-2560:json
<a id="mc___r_2560_p_640_sub_4_0_4_9_"></a>
### mc       @ r-2560-p-640-sub-4/0_49-->p2s_seg_tf-617
python3 data/scripts/create_seg_tfrecord.py cfg=617:0_49:p-640:sub-4:r-2560:mc:json
<a id="lac___r_2560_p_640_sub_4_0_4_9_"></a>
### lac       @ r-2560-p-640-sub-4/0_49-->p2s_seg_tf-617
python3 data/scripts/create_seg_tfrecord.py cfg=617:0_49:p-640:sub-4:r-2560:lac:json

<a id="r_640_p_640_sub_8___0_49_"></a>
## r-640-p-640-sub-8       @ 0_49-->p2s_seg_tf-617
<a id="bin___r_640_p_640_sub_8_0_49_"></a>
### bin       @ r-640-p-640-sub-8/0_49-->p2s_seg_tf-617
python3 data/scripts/create_seg_tfrecord.py cfg=617:0_49:p-640:sub-8:r-640:json
<a id="mc___r_640_p_640_sub_8_0_49_"></a>
### mc       @ r-640-p-640-sub-8/0_49-->p2s_seg_tf-617
python3 data/scripts/create_seg_tfrecord.py cfg=617:0_49:p-640:sub-8:r-640:mc:json
<a id="lac___r_640_p_640_sub_8_0_49_"></a>
### lac       @ r-640-p-640-sub-8/0_49-->p2s_seg_tf-617
python3 data/scripts/create_seg_tfrecord.py cfg=617:0_49:p-640:sub-8:r-640:lac:json

