<!-- MarkdownTOC -->

- [0_4        @ ipsc](#0_4___ipsc_)
- [5_9       @ ipsc](#5_9___ipsc_)
- [16_53       @ ipsc](#16_53___ipsc_)
    - [len-2       @ 16_53](#len_2___16_5_3_)
    - [len-3       @ 16_53](#len_3___16_5_3_)
    - [len-6       @ 16_53](#len_6___16_5_3_)
- [0_37       @ ipsc](#0_37___ipsc_)
- [54_126       @ ipsc](#54_126___ipsc_)
    - [len-2       @ 54_126](#len_2___54_126_)
        - [sample-8       @ len-2/54_126](#sample_8___len_2_54_126_)
    - [len-3       @ 54_126](#len_3___54_126_)
    - [len-6       @ 54_126](#len_6___54_126_)
        - [strd-6       @ len-6/54_126](#strd_6___len_6_54_126_)
        - [sample-8       @ len-6/54_126](#sample_8___len_6_54_126_)
        - [sample-4       @ len-6/54_126](#sample_4___len_6_54_126_)

<!-- /MarkdownTOC -->

<a id="0_4___ipsc_"></a>
# 0_4        @ ipsc-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=ipsc:shards-2:len-2 ann_file=ext_reorg_roi_g2_0_4
**12094**
python data/scripts/create_video_tfrecord.py cfg=ipsc:shards-2:len-2 ann_file=ext_reorg_roi_12094_0_4
**12094_short**
python data/scripts/create_video_tfrecord.py cfg=ipsc:shards-2:len-2 ann_file=ext_reorg_roi_12094_short_0_4

<a id="5_9___ipsc_"></a>
# 5_9       @ ipsc-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=ipsc:shards-2:len-2 ann_file=ext_reorg_roi_g2_5_9
**fgs-4**
python data/scripts/create_video_tfrecord.py cfg=ipsc:shards-2:len-2:fgs-4 ann_file=ext_reorg_roi_g2_5_9

<a id="16_53___ipsc_"></a>
# 16_53       @ ipsc-->p2s_vid_tf
<a id="len_2___16_5_3_"></a>
## len-2       @ 16_53-->tf_vid
python data/scripts/create_video_tfrecord.py cfg=ipsc:16_53:len-2
python data/scripts/create_video_tfrecord.py cfg=ipsc:16_53:len-2:strd-2
<a id="len_3___16_5_3_"></a>
## len-3       @ 16_53-->tf_vid
python data/scripts/create_video_tfrecord.py cfg=ipsc:16_53:len-3
python data/scripts/create_video_tfrecord.py cfg=ipsc:16_53:len-3:strd-3
<a id="len_6___16_5_3_"></a>
## len-6       @ 16_53-->tf_vid
python data/scripts/create_video_tfrecord.py cfg=ipsc:16_53:len-6
python data/scripts/create_video_tfrecord.py cfg=ipsc:16_53:len-6:strd-6

<a id="0_37___ipsc_"></a>
# 0_37       @ ipsc-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=ipsc:0_37:len-2
**fgs-4**
python data/scripts/create_video_tfrecord.py cfg=ipsc:0_37:len-2:fgs-4 

<a id="54_126___ipsc_"></a>
# 54_126       @ ipsc-->p2s_vid_tf
<a id="len_2___54_126_"></a>
## len-2       @ 54_126-->tf_vid
python data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:len-2
python data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:len-2:strd-2
<a id="sample_8___len_2_54_126_"></a>
### sample-8       @ len-2/54_126-->tf_vid
python data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:len-2:strd-2:sample-8

<a id="len_3___54_126_"></a>
## len-3       @ 54_126-->tf_vid
python data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:len-3
python data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:len-3:strd-3
<a id="len_6___54_126_"></a>
## len-6       @ 54_126-->tf_vid
python data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:len-6:asi-2
`dbg`
python data/scripts/create_video_tfrecord.py cfg=ipsc:frame-54_65:seq-0:len-6:asi-2
<a id="strd_6___len_6_54_126_"></a>
### strd-6       @ len-6/54_126-->tf_vid
python data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:len-6:strd-6
<a id="sample_8___len_6_54_126_"></a>
### sample-8       @ len-6/54_126-->tf_vid
python data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:len-6:strd-6:sample-8
<a id="sample_4___len_6_54_126_"></a>
### sample-4       @ len-6/54_126-->tf_vid
python data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:len-6:strd-6:sample-4
