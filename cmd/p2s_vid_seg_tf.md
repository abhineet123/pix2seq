<!-- MarkdownTOC -->

- [16_53       @ ipsc](#16_53___ipsc_)
    - [r-2560       @ 16_53](#r_2560___16_5_3_)
        - [p-1280-sub-8       @ r-2560/16_53](#p_1280_sub_8___r_2560_16_53_)
            - [vis       @ p-1280-sub-8/r-2560/16_53](#vis___p_1280_sub_8_r_2560_16_5_3_)
        - [p-640-sub-8       @ r-2560/16_53](#p_640_sub_8___r_2560_16_53_)
            - [tac       @ p-640-sub-8/r-2560/16_53](#tac___p_640_sub_8_r_2560_16_53_)
            - [ltac       @ p-640-sub-8/r-2560/16_53](#ltac___p_640_sub_8_r_2560_16_53_)
            - [ord       @ p-640-sub-8/r-2560/16_53](#ord___p_640_sub_8_r_2560_16_53_)
        - [p-640-sub-8-mc       @ r-2560/16_53](#p_640_sub_8_mc___r_2560_16_53_)
            - [lac       @ p-640-sub-8-mc/r-2560/16_53](#lac___p_640_sub_8_mc_r_2560_16_5_3_)
            - [tac       @ p-640-sub-8-mc/r-2560/16_53](#tac___p_640_sub_8_mc_r_2560_16_5_3_)
            - [ltac       @ p-640-sub-8-mc/r-2560/16_53](#ltac___p_640_sub_8_mc_r_2560_16_5_3_)
            - [ord       @ p-640-sub-8-mc/r-2560/16_53](#ord___p_640_sub_8_mc_r_2560_16_5_3_)
        - [p-640-sub-8-len-3       @ r-2560/16_53](#p_640_sub_8_len_3___r_2560_16_53_)
        - [p-640-sub-8-mc-len-3       @ r-2560/16_53](#p_640_sub_8_mc_len_3___r_2560_16_53_)
        - [p-640-sub-8-len-4       @ r-2560/16_53](#p_640_sub_8_len_4___r_2560_16_53_)
        - [p-640-sub-8-mc-len-4       @ r-2560/16_53](#p_640_sub_8_mc_len_4___r_2560_16_53_)
        - [p-640-sub-4       @ r-2560/16_53](#p_640_sub_4___r_2560_16_53_)
        - [p-640-sub-4-mc       @ r-2560/16_53](#p_640_sub_4_mc___r_2560_16_53_)
            - [len-3       @ p-640-sub-4-mc/r-2560/16_53](#len_3___p_640_sub_4_mc_r_2560_16_5_3_)
    - [res-640       @ 16_53](#res_640___16_5_3_)
        - [sz-80       @ res-640/16_53](#sz_80___res_640_16_5_3_)
            - [seq-0       @ sz-80/res-640/16_53](#seq_0___sz_80_res_640_16_5_3_)
            - [seq-1       @ sz-80/res-640/16_53](#seq_1___sz_80_res_640_16_5_3_)
        - [sz-80-mc       @ res-640/16_53](#sz_80_mc___res_640_16_5_3_)
        - [sz-160       @ res-640/16_53](#sz_160___res_640_16_5_3_)
        - [sz-640-sub-8       @ res-640/16_53](#sz_640_sub_8___res_640_16_5_3_)
        - [sz-640-sub-4       @ res-640/16_53](#sz_640_sub_4___res_640_16_5_3_)
    - [res-320       @ 16_53](#res_320___16_5_3_)
        - [sz-80       @ res-320/16_53](#sz_80___res_320_16_5_3_)
        - [sz-80-aug       @ res-320/16_53](#sz_80_aug___res_320_16_5_3_)
        - [sz-160       @ res-320/16_53](#sz_160___res_320_16_5_3_)
        - [sz-160-aug       @ res-320/16_53](#sz_160_aug___res_320_16_5_3_)
- [54_126](#54_12_6_)
    - [res-640       @ 54_126](#res_640___54_126_)
        - [sz-80       @ res-640/54_126](#sz_80___res_640_54_126_)
    - [res-320       @ 54_126](#res_320___54_126_)
        - [sz-80       @ res-320/54_126](#sz_80___res_320_54_126_)
        - [sz-160       @ res-320/54_126](#sz_160___res_320_54_126_)

<!-- /MarkdownTOC -->
<a id="16_53___ipsc_"></a>
# 16_53       @ ipsc-->p2s_seg_tfrecord
<a id="r_2560___16_5_3_"></a>
## r-2560       @ 16_53-->p2s_vid_seg_tf

<a id="p_1280_sub_8___r_2560_16_53_"></a>
### p-1280-sub-8       @ r-2560/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-1280:sub-8:r-2560:gz:proc-1
<a id="vis___p_1280_sub_8_r_2560_16_5_3_"></a>
#### vis       @ p-1280-sub-8/r-2560/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:88_100:p-1280:sub-8:r-2560:gz:proc-1:ltac:seq-0:vis
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:88_100:p-1280:sub-8:r-2560:gz:proc-1:ltac:mc:seq-0:vis

<a id="p_640_sub_8___r_2560_16_53_"></a>
### p-640-sub-8       @ r-2560/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-8:r-2560:gz:proc-1
`dbg`
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:p-640:sub-8:r-2560:gz:proc-1:seq-0:frame-0_1
<a id="tac___p_640_sub_8_r_2560_16_53_"></a>
#### tac       @ p-640-sub-8/r-2560/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-8:r-2560:gz:proc-1:tac
`dbg`
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:p-640:sub-8:r-2560:gz:proc-1:seq-0:frame-0_1:tac
<a id="ltac___p_640_sub_8_r_2560_16_53_"></a>
#### ltac       @ p-640-sub-8/r-2560/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-8:r-2560:gz:proc-1:ltac
`dbg`
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:p-640:sub-8:r-2560:gz:proc-1:seq-0:frame-0_1:ltac

<a id="ord___p_640_sub_8_r_2560_16_53_"></a>
#### ord       @ p-640-sub-8/r-2560/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-8:r-2560:gz:proc-1:ord

<a id="p_640_sub_8_mc___r_2560_16_53_"></a>
### p-640-sub-8-mc       @ r-2560/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-8:r-2560:gz:proc-1:mc
`dbg`
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:p-640:sub-8:r-2560:gz:proc-1:seq-0:frame-0_1:mc
<a id="lac___p_640_sub_8_mc_r_2560_16_5_3_"></a>
#### lac       @ p-640-sub-8-mc/r-2560/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-8:r-2560:gz:proc-1:mc:lac
`dbg`
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:p-640:sub-8:r-2560:gz:proc-1:seq-0:frame-0_1:mc:lac
<a id="tac___p_640_sub_8_mc_r_2560_16_5_3_"></a>
#### tac       @ p-640-sub-8-mc/r-2560/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-8:r-2560:gz:proc-1:mc:tac
`dbg`
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:p-640:sub-8:r-2560:gz:proc-1:seq-0:frame-0_1:mc:tac
<a id="ltac___p_640_sub_8_mc_r_2560_16_5_3_"></a>
#### ltac       @ p-640-sub-8-mc/r-2560/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-8:r-2560:gz:proc-1:mc:ltac
`dbg`
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:p-640:sub-8:r-2560:gz:proc-1:seq-0:frame-0_1:mc:ltac
<a id="ord___p_640_sub_8_mc_r_2560_16_5_3_"></a>
#### ord       @ p-640-sub-8-mc/r-2560/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-8:r-2560:gz:proc-1:mc:ord

<a id="p_640_sub_8_len_3___r_2560_16_53_"></a>
### p-640-sub-8-len-3       @ r-2560/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-8:r-2560:gz:proc-1:len-3:stats 
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-8:r-2560:gz:proc-1:len-3:tac:stats
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-8:r-2560:gz:proc-1:len-3:ltac:stats

<a id="p_640_sub_8_mc_len_3___r_2560_16_53_"></a>
### p-640-sub-8-mc-len-3       @ r-2560/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-8:r-2560:gz:proc-1:mc:len-3:stats 
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-8:r-2560:gz:proc-1:mc:len-3:lac:stats 
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-8:r-2560:gz:proc-1:mc:len-3:tac:stats
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-8:r-2560:gz:proc-1:mc:len-3:ltac:stats

<a id="p_640_sub_8_len_4___r_2560_16_53_"></a>
### p-640-sub-8-len-4       @ r-2560/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-8:r-2560:gz:proc-1:len-4:stats
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-8:r-2560:gz:proc-1:len-4:tac:stats
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-8:r-2560:gz:proc-1:len-4:ltac:stats

<a id="p_640_sub_8_mc_len_4___r_2560_16_53_"></a>
### p-640-sub-8-mc-len-4       @ r-2560/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-8:r-2560:gz:proc-1:mc:len-4:stats
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-8:r-2560:gz:proc-1:mc:len-4:lac:stats 
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-8:r-2560:gz:proc-1:mc:len-4:tac:stats
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-8:r-2560:gz:proc-1:mc:len-4:ltac:stats

<a id="p_640_sub_4___r_2560_16_53_"></a>
### p-640-sub-4       @ r-2560/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-4:r-2560:gz:proc-1

python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-4:r-2560:gz:proc-1:ord

python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-4:r-2560:gz:proc-1:tac

python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-4:r-2560:gz:proc-1:ltac:stats

<a id="p_640_sub_4_mc___r_2560_16_53_"></a>
### p-640-sub-4-mc       @ r-2560/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-4:r-2560:gz:proc-1:mc

python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-4:r-2560:gz:proc-1:mc:lac:stats

python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-4:r-2560:gz:proc-1:mc:ord

python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-4:r-2560:gz:proc-1:mc:tac

python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-4:r-2560:gz:proc-1:mc:ltac:stats

`vis`
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-4:r-2560:gz:proc-1:mc:vis:seq-0_0

<a id="len_3___p_640_sub_4_mc_r_2560_16_5_3_"></a>
#### len-3       @ p-640-sub-4-mc/r-2560/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-4:r-2560:gz:proc-1:mc:vis:len-3
`vis`
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-4:r-2560:gz:proc-1:mc:vis:len-3:seq-0_0
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-640:sub-4:r-2560:gz:proc-1:mc:vis:tac-0:seq-0_0

<a id="res_640___16_5_3_"></a>
## res-640       @ 16_53-->p2s_vid_seg_tf
<a id="sz_80___res_640_16_5_3_"></a>
### sz-80       @ res-640/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-80:r-640:gz
<a id="seq_0___sz_80_res_640_16_5_3_"></a>
#### seq-0       @ sz-80/res-640/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-80:r-640:gz:seq-0
<a id="seq_1___sz_80_res_640_16_5_3_"></a>
#### seq-1       @ sz-80/res-640/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-80:r-640:gz:seq-1

<a id="sz_80_mc___res_640_16_5_3_"></a>
### sz-80-mc       @ res-640/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:p-80:r-640:gz:mc

<a id="sz_160___res_640_16_5_3_"></a>
### sz-160       @ res-640/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:gz:p-160:r-640

<a id="sz_640_sub_8___res_640_16_5_3_"></a>
### sz-640-sub-8       @ res-640/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:gz:p-640:r-640:sub-8

<a id="sz_640_sub_4___res_640_16_5_3_"></a>
### sz-640-sub-4       @ res-640/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:gz:p-640:r-640:sub-4

python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:gz:p-640:r-640:sub-4:mc


<a id="res_320___16_5_3_"></a>
## res-320       @ 16_53-->p2s_vid_seg_tf
<a id="sz_80___res_320_16_5_3_"></a>
### sz-80       @ res-320/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:gz:p-80:r-320

<a id="sz_80_aug___res_320_16_5_3_"></a>
### sz-80-aug       @ res-320/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:gz:p-80:r-320:strd-40_80:rot-15_345_4:flip-1
`seq-0`
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:gz:p-80:r-320:strd-40_80:rot-15_345_4:flip-1:seq-0

<a id="sz_160___res_320_16_5_3_"></a>
### sz-160       @ res-320/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:gz:p-160:r-320

<a id="sz_160_aug___res_320_16_5_3_"></a>
### sz-160-aug       @ res-320/16_53-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:16_53:gz:p-160:r-320:strd-40_160:rot-15_345_4:flip-1

<a id="54_12_6_"></a>
# 54_126
<a id="res_640___54_126_"></a>
## res-640       @ 54_126-->p2s_vid_seg_tf
<a id="sz_80___res_640_54_126_"></a>
### sz-80       @ res-640/54_126-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:54_126:p-80:r-640:gz
<a id="res_320___54_126_"></a>
## res-320       @ 54_126-->p2s_vid_seg_tf
<a id="sz_80___res_320_54_126_"></a>
### sz-80       @ res-320/54_126-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:54_126:p-80:r-320:gz
<a id="sz_160___res_320_54_126_"></a>
### sz-160       @ res-320/54_126-->p2s_vid_seg_tf
python3 data/scripts/create_video_seg_tfrecord.py cfg=ipsc:54_126:gz:p-160:r-320