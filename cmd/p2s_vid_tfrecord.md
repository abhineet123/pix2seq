<!-- MarkdownTOC -->

- [mnist       @ tfrecord](#mnist___tfrecord_)
    - [640-1       @ mnist](#640_1___mnis_t_)
        - [len-2       @ 640-1/mnist](#len_2___640_1_mnis_t_)
        - [len-3       @ 640-1/mnist](#len_3___640_1_mnis_t_)
        - [len-9       @ 640-1/mnist](#len_9___640_1_mnis_t_)
    - [640-3       @ mnist](#640_3___mnis_t_)
    - [640-5       @ mnist](#640_5___mnis_t_)
        - [len-2       @ 640-5/mnist](#len_2___640_5_mnis_t_)
        - [len-3       @ 640-5/mnist](#len_3___640_5_mnis_t_)
        - [len-4       @ 640-5/mnist](#len_4___640_5_mnis_t_)
        - [len-6       @ 640-5/mnist](#len_6___640_5_mnis_t_)
- [ipsc](#ips_c_)
    - [ext_reorg_roi_g2_0_4       @ ipsc](#ext_reorg_roi_g2_0_4___ipsc_)
    - [ext_reorg_roi_g2_5_9       @ ipsc](#ext_reorg_roi_g2_5_9___ipsc_)
    - [ext_reorg_roi_g2_0_37       @ ipsc](#ext_reorg_roi_g2_0_37___ipsc_)

<!-- /MarkdownTOC -->
<a id="mnist___tfrecord_"></a>
# mnist       @ tfrecord-->p2s_setup
<a id="640_1___mnis_t_"></a>
## 640-1       @ mnist-->p2s_vid_tfrecord
<a id="len_2___640_1_mnis_t_"></a>
### len-2       @ 640-1/mnist-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-1:12_1000:gz:len-2:strd-1
<a id="len_3___640_1_mnis_t_"></a>
### len-3       @ 640-1/mnist-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-1:12_1000:gz:len-3:strd-1:proc-12
<a id="len_9___640_1_mnis_t_"></a>
### len-9       @ 640-1/mnist-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-1:12_1000:gz:len-9:strd-1:proc-12:suffix-train

<a id="640_3___mnis_t_"></a>
## 640-3       @ mnist-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-3:12_1000:gz:len-2:strd-1:proc-6

<a id="640_5___mnis_t_"></a>
## 640-5       @ mnist-->p2s_vid_tfrecord
<a id="len_2___640_5_mnis_t_"></a>
### len-2       @ 640-5/mnist-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-2:strd-1:proc-12:suffix-train
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-2:strd-1:proc-12:suffix-test
<a id="len_3___640_5_mnis_t_"></a>
### len-3       @ 640-5/mnist-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-3:strd-1:proc-12:suffix-train
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-3:strd-1:proc-12:suffix-test
<a id="len_4___640_5_mnis_t_"></a>
### len-4       @ 640-5/mnist-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-4:strd-1:proc-12:suffix-train
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-4:strd-1:proc-12:suffix-test
<a id="len_6___640_5_mnis_t_"></a>
### len-6       @ 640-5/mnist-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-6:strd-1:proc-12:suffix-train
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-6:strd-1:proc-12:suffix-test

<a id="ips_c_"></a>
# ipsc
<a id="ext_reorg_roi_g2_0_4___ipsc_"></a>
## ext_reorg_roi_g2_0_4       @ ipsc-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:gz:shards-2:len-2:strd-1 ann_file=ext_reorg_roi_g2_0_4
**12094**
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:gz:shards-2:len-2:strd-1 ann_file=ext_reorg_roi_12094_0_4
**12094_short**
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:gz:shards-2:len-2:strd-1 ann_file=ext_reorg_roi_12094_short_0_4

<a id="ext_reorg_roi_g2_5_9___ipsc_"></a>
## ext_reorg_roi_g2_5_9       @ ipsc-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:gz:shards-2:len-2:strd-1 ann_file=ext_reorg_roi_g2_5_9
**fg-4**
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:gz:shards-2:len-2:strd-1:fg-4 ann_file=ext_reorg_roi_g2_5_9
<a id="ext_reorg_roi_g2_0_37___ipsc_"></a>
## ext_reorg_roi_g2_0_37       @ ipsc-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:gz:shards-2:len-2:strd-1 ann_file=ext_reorg_roi_g2_0_37
**fg-4**
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:gz:shards-2:len-2:strd-1:fg-4 ann_file=ext_reorg_roi_g2_0_37
