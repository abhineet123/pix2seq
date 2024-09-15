<!-- MarkdownTOC -->

- [1k8_vid_entire_seq       @ acamp](#1k8_vid_entire_seq___acam_p_)
- [1k8_vid_entire_seq_inv       @ acamp](#1k8_vid_entire_seq_inv___acam_p_)
- [1k8_vid_entire_seq_inv_2_per_seq       @ acamp](#1k8_vid_entire_seq_inv_2_per_seq___acam_p_)
- [10k6_vid_entire_seq       @ acamp](#10k6_vid_entire_seq___acam_p_)
- [10k6_vid_entire_seq_inv       @ acamp](#10k6_vid_entire_seq_inv___acam_p_)
- [20k6_5_video       @ acamp](#20k6_5_video___acam_p_)
- [20k6_5_video_inv       @ acamp](#20k6_5_video_inv___acam_p_)
- [2_per_seq_dbg_bear       @ acamp](#2_per_seq_dbg_bear___acam_p_)

<!-- /MarkdownTOC -->

<a id="1k8_vid_entire_seq___acam_p_"></a>
# 1k8_vid_entire_seq       @ acamp-->tf_vid-acamp
python data/scripts/create_video_tfrecord.py cfg=acamp:1k8_vid_entire_seq:len-2:strd-1:asi-2
python data/scripts/create_video_tfrecord.py cfg=acamp:1k8_vid_entire_seq:len-2:strd-2
<a id="1k8_vid_entire_seq_inv___acam_p_"></a>
# 1k8_vid_entire_seq_inv       @ acamp-->tf_vid-acamp
python data/scripts/create_video_tfrecord.py cfg=acamp:1k8_vid_entire_seq_inv:len-2:strd-1:asi-2
<a id="1k8_vid_entire_seq_inv_2_per_seq___acam_p_"></a>
# 1k8_vid_entire_seq_inv_2_per_seq       @ acamp-->tf_vid-acamp
python data/scripts/create_video_tfrecord.py cfg=acamp:1k8_vid_entire_seq_inv_2_per_seq:len-2:strd-1:asi-2

<a id="10k6_vid_entire_seq___acam_p_"></a>
# 10k6_vid_entire_seq       @ acamp-->tf_vid-acamp
python data/scripts/create_video_tfrecord.py cfg=acamp:10k6_vid_entire_seq:len-2:strd-1:asi-2
python data/scripts/create_video_tfrecord.py cfg=acamp:10k6_vid_entire_seq:len-2:strd-2
<a id="10k6_vid_entire_seq_inv___acam_p_"></a>
# 10k6_vid_entire_seq_inv       @ acamp-->tf_vid-acamp
python data/scripts/create_video_tfrecord.py cfg=acamp:10k6_vid_entire_seq_inv:len-2:strd-1:asi-2
python data/scripts/create_video_tfrecord.py cfg=acamp:10k6_vid_entire_seq_inv:len-2:strd-2

<a id="20k6_5_video___acam_p_"></a>
# 20k6_5_video       @ acamp-->tf_vid-acamp
python data/scripts/create_video_tfrecord.py cfg=acamp:20k6_5_video:len-2:strd-1:asi-2
python data/scripts/create_video_tfrecord.py cfg=acamp:20k6_5_video:len-2:strd-2
<a id="20k6_5_video_inv___acam_p_"></a>
# 20k6_5_video_inv       @ acamp-->tf_vid-acamp
python data/scripts/create_video_tfrecord.py cfg=acamp:20k6_5_video_inv:len-2:strd-1:asi-2
python data/scripts/create_video_tfrecord.py cfg=acamp:20k6_5_video_inv:len-2:strd-2

<a id="2_per_seq_dbg_bear___acam_p_"></a>
# 2_per_seq_dbg_bear       @ acamp-->tf_vid-acamp
python data/scripts/create_video_tfrecord.py cfg=acamp:2_per_seq_dbg_bear:len-2:strd-1
