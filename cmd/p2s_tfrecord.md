<!-- MarkdownTOC -->

- [mnist       @ tfrecord](#mnist___tfrecord_)
    - [640-1       @ mnist](#640_1___mnis_t_)
        - [len-2       @ 640-1/mnist](#len_2___640_1_mnis_t_)
        - [len-3       @ 640-1/mnist](#len_3___640_1_mnis_t_)
    - [640-3       @ mnist](#640_3___mnis_t_)
    - [640-5       @ mnist](#640_5___mnis_t_)
- [ipsc       @ tfrecord](#ipsc___tfrecord_)
    - [ext_reorg_roi_g2_16_53       @ ipsc](#ext_reorg_roi_g2_16_53___ipsc_)
    - [ext_reorg_roi_g2_0_1       @ ipsc](#ext_reorg_roi_g2_0_1___ipsc_)
    - [ext_reorg_roi_g2_0_15       @ ipsc](#ext_reorg_roi_g2_0_15___ipsc_)
    - [ext_reorg_roi_g2_0_37       @ ipsc](#ext_reorg_roi_g2_0_37___ipsc_)
    - [ext_reorg_roi_g2_38_53       @ ipsc](#ext_reorg_roi_g2_38_53___ipsc_)
- [ipsc_video](#ipsc_vide_o_)
    - [ext_reorg_roi_g2_0_4       @ ipsc_video](#ext_reorg_roi_g2_0_4___ipsc_video_)
    - [ext_reorg_roi_g2_5_9       @ ipsc_video](#ext_reorg_roi_g2_5_9___ipsc_video_)
    - [ext_reorg_roi_g2_0_37       @ ipsc_video](#ext_reorg_roi_g2_0_37___ipsc_video_)

<!-- /MarkdownTOC -->
<a id="mnist___tfrecord_"></a>
# mnist       @ tfrecord-->p2s_setup
<a id="640_1___mnis_t_"></a>
## 640-1       @ mnist-->p2s_tfrecord
<a id="len_2___640_1_mnis_t_"></a>
### len-2       @ 640-1/mnist-->p2s_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-1:12_1000:gz:len-2:strd-1
<a id="len_3___640_1_mnis_t_"></a>
### len-3       @ 640-1/mnist-->p2s_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-1:12_1000:gz:len-3:strd-1:proc-12

<a id="640_3___mnis_t_"></a>
## 640-3       @ mnist-->p2s_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-3:12_1000:gz:len-2:strd-1:proc-6

<a id="640_5___mnis_t_"></a>
## 640-5       @ mnist-->p2s_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-2:strd-1:proc-6

<a id="ipsc___tfrecord_"></a>
# ipsc       @ tfrecord-->p2s_setup
python3 data/scripts/create_ipsc_tfrecord.py
<a id="ext_reorg_roi_g2_16_53___ipsc_"></a>
## ext_reorg_roi_g2_16_53       @ ipsc-->p2s_tfrecord
python3 data/scripts/create_ipsc_tfrecord.py --ann_file=ext_reorg_roi_g2_16_53.json --n_proc=0
<a id="ext_reorg_roi_g2_0_1___ipsc_"></a>
## ext_reorg_roi_g2_0_1       @ ipsc-->p2s_tfrecord
python data/scripts/create_ipsc_tfrecord.py --ann_file=ext_reorg_roi_g2_0_1.json --n_proc=0
<a id="ext_reorg_roi_g2_0_15___ipsc_"></a>
## ext_reorg_roi_g2_0_15       @ ipsc-->p2s_tfrecord
python data/scripts/create_ipsc_tfrecord.py --ann_file=ext_reorg_roi_g2_0_15.json --n_proc=0
<a id="ext_reorg_roi_g2_0_37___ipsc_"></a>
## ext_reorg_roi_g2_0_37       @ ipsc-->p2s_tfrecord
python data/scripts/create_ipsc_tfrecord.py --ann_file=ext_reorg_roi_g2_0_37.json --n_proc=0
<a id="ext_reorg_roi_g2_38_53___ipsc_"></a>
## ext_reorg_roi_g2_38_53       @ ipsc-->p2s_tfrecord
python3 data/scripts/create_ipsc_tfrecord.py --ann_file=ext_reorg_roi_g2_38_53.json --n_proc=0

<a id="ipsc_vide_o_"></a>
# ipsc_video
<a id="ext_reorg_roi_g2_0_4___ipsc_video_"></a>
## ext_reorg_roi_g2_0_4       @ ipsc_video-->p2s_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:gz:shards-2:len-2:strd-1 ann_file=ext_reorg_roi_g2_0_4
**12094**
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:gz:shards-2:len-2:strd-1 ann_file=ext_reorg_roi_12094_0_4
**12094_short**
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:gz:shards-2:len-2:strd-1 ann_file=ext_reorg_roi_12094_short_0_4

<a id="ext_reorg_roi_g2_5_9___ipsc_video_"></a>
## ext_reorg_roi_g2_5_9       @ ipsc_video-->p2s_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:gz:shards-2:len-2:strd-1 ann_file=ext_reorg_roi_g2_5_9
**fg-4**
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:gz:shards-2:len-2:strd-1:fg-4 ann_file=ext_reorg_roi_g2_5_9
<a id="ext_reorg_roi_g2_0_37___ipsc_video_"></a>
## ext_reorg_roi_g2_0_37       @ ipsc_video-->p2s_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:gz:shards-2:len-2:strd-1 ann_file=ext_reorg_roi_g2_0_37
**fg-4**
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:gz:shards-2:len-2:strd-1:fg-4 ann_file=ext_reorg_roi_g2_0_37
