<!-- MarkdownTOC -->

- [gram       @ tfrecord](#gram___tfrecord_)
    - [0_1       @ gram](#0_1___gram_)
        - [len-9       @ 0_1/gram](#len_9___0_1_gram_)
        - [len-14       @ 0_1/gram](#len_14___0_1_gram_)
            - [0_2000       @ len-14/0_1/gram](#0_2000___len_14_0_1_gra_m_)
            - [3000_5000       @ len-14/0_1/gram](#3000_5000___len_14_0_1_gra_m_)
        - [len-16       @ 0_1/gram](#len_16___0_1_gram_)
- [idot       @ gram](#idot___gram_)
    - [8_8       @ idot](#8_8___idot_)
- [detrac](#detra_c_)
    - [0_0       @ detrac](#0_0___detrac_)
    - [0_19       @ detrac](#0_19___detrac_)
        - [strd-2       @ 0_19/detrac](#strd_2___0_19_detra_c_)
        - [len-3       @ 0_19/detrac](#len_3___0_19_detra_c_)
        - [len-4       @ 0_19/detrac](#len_4___0_19_detra_c_)
        - [len-6       @ 0_19/detrac](#len_6___0_19_detra_c_)
        - [len-8       @ 0_19/detrac](#len_8___0_19_detra_c_)
        - [len-9       @ 0_19/detrac](#len_9___0_19_detra_c_)
    - [0_9       @ detrac](#0_9___detrac_)
        - [strd-2       @ 0_9/detrac](#strd_2___0_9_detrac_)
    - [49_68       @ detrac](#49_68___detrac_)
        - [strd-2       @ 49_68/detrac](#strd_2___49_68_detrac_)
- [ipsc](#ips_c_)
    - [0_4        @ ipsc](#0_4___ipsc_)
    - [5_9       @ ipsc](#5_9___ipsc_)
    - [16_53       @ ipsc](#16_53___ipsc_)
        - [len-2       @ 16_53/ipsc](#len_2___16_53_ipsc_)
        - [len-3       @ 16_53/ipsc](#len_3___16_53_ipsc_)
        - [len-6       @ 16_53/ipsc](#len_6___16_53_ipsc_)
    - [0_37       @ ipsc](#0_37___ipsc_)
    - [54_126       @ ipsc](#54_126___ipsc_)
        - [len-2       @ 54_126/ipsc](#len_2___54_126_ips_c_)
            - [sample-8       @ len-2/54_126/ipsc](#sample_8___len_2_54_126_ips_c_)
        - [len-3       @ 54_126/ipsc](#len_3___54_126_ips_c_)
        - [len-6       @ 54_126/ipsc](#len_6___54_126_ips_c_)
            - [sample-8       @ len-6/54_126/ipsc](#sample_8___len_6_54_126_ips_c_)
            - [sample-4       @ len-6/54_126/ipsc](#sample_4___len_6_54_126_ips_c_)
- [acamp](#acamp_)
    - [1k8_vid_entire_seq       @ acamp](#1k8_vid_entire_seq___acam_p_)
    - [1k8_vid_entire_seq_inv       @ acamp](#1k8_vid_entire_seq_inv___acam_p_)
    - [10k6_vid_entire_seq       @ acamp](#10k6_vid_entire_seq___acam_p_)
    - [10k6_vid_entire_seq_inv       @ acamp](#10k6_vid_entire_seq_inv___acam_p_)

<!-- /MarkdownTOC -->

<a id="gram___tfrecord_"></a>
# gram       @ tfrecord-->p2s_setup
<a id="0_1___gram_"></a>
## 0_1       @ gram-->p2s_vid_tf
<a id="len_9___0_1_gram_"></a>
### len-9       @ 0_1/gram-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=gram:0_1:len-9:strd-1

python data/scripts/create_video_tfrecord.py cfg=gram:0_1:len-9:strd-9

<a id="len_14___0_1_gram_"></a>
### len-14       @ 0_1/gram-->p2s_vid_tf
<a id="0_2000___len_14_0_1_gra_m_"></a>
#### 0_2000       @ len-14/0_1/gram-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=gram:0_1:frame-0_2000:len-14:strd-1
<a id="3000_5000___len_14_0_1_gra_m_"></a>
#### 3000_5000       @ len-14/0_1/gram-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=gram:0_1:frame-3000_5000:len-14:strd-14

<a id="len_16___0_1_gram_"></a>
### len-16       @ 0_1/gram-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=gram:0_1:frame-0_2000:len-16:strd-1

<a id="idot___gram_"></a>
# idot       @ gram-->p2s_vid_tfrecord
python data/scripts/create_video_tfrecord.py cfg=idot:len-9:strd-1
<a id="8_8___idot_"></a>
## 8_8       @ idot-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=idot:8_8:len-2:strd-10:fg-10

python data/scripts/create_video_tfrecord.py cfg=idot:8_8:len-3:strd-20:fg-10

python data/scripts/create_video_tfrecord.py cfg=idot:8_8:len-6:strd-50:fg-10

<a id="detra_c_"></a>
# detrac
<a id="0_0___detrac_"></a>
## 0_0       @ detrac-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_0:len-2:strd-1

<a id="0_19___detrac_"></a>
## 0_19       @ detrac-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:len-2:strd-1

<a id="strd_2___0_19_detra_c_"></a>
### strd-2       @ 0_19/detrac-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:len-2:strd-2
<a id="len_3___0_19_detra_c_"></a>
### len-3       @ 0_19/detrac-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:len-3:strd-1
<a id="len_4___0_19_detra_c_"></a>
### len-4       @ 0_19/detrac-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:len-4:strd-1
<a id="len_6___0_19_detra_c_"></a>
### len-6       @ 0_19/detrac-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:len-6:strd-1
<a id="len_8___0_19_detra_c_"></a>
### len-8       @ 0_19/detrac-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:len-8:strd-1
<a id="len_9___0_19_detra_c_"></a>
### len-9       @ 0_19/detrac-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:len-9:strd-1

<a id="0_9___detrac_"></a>
## 0_9       @ detrac-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_9:len-2:strd-1
<a id="strd_2___0_9_detrac_"></a>
### strd-2       @ 0_9/detrac-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_9:len-2:strd-2

<a id="49_68___detrac_"></a>
## 49_68       @ detrac-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_68:len-2:strd-1
<a id="strd_2___49_68_detrac_"></a>
### strd-2       @ 49_68/detrac-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_68:len-2:strd-2

<a id="ips_c_"></a>
# ipsc
<a id="0_4___ipsc_"></a>
##0_4        @ ipsc-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=ipsc:shards-2:len-2:strd-1 ann_file=ext_reorg_roi_g2_0_4
**12094**
python data/scripts/create_video_tfrecord.py cfg=ipsc:shards-2:len-2:strd-1 ann_file=ext_reorg_roi_12094_0_4
**12094_short**
python data/scripts/create_video_tfrecord.py cfg=ipsc:shards-2:len-2:strd-1 ann_file=ext_reorg_roi_12094_short_0_4

<a id="5_9___ipsc_"></a>
## 5_9       @ ipsc-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=ipsc:shards-2:len-2:strd-1 ann_file=ext_reorg_roi_g2_5_9
**fgs-4**
python data/scripts/create_video_tfrecord.py cfg=ipsc:shards-2:len-2:strd-1:fgs-4 ann_file=ext_reorg_roi_g2_5_9

<a id="16_53___ipsc_"></a>
## 16_53       @ ipsc-->p2s_vid_tf
<a id="len_2___16_53_ipsc_"></a>
### len-2       @ 16_53/ipsc-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=ipsc:16_53:len-2:strd-1
python data/scripts/create_video_tfrecord.py cfg=ipsc:16_53:len-2:strd-2
<a id="len_3___16_53_ipsc_"></a>
### len-3       @ 16_53/ipsc-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=ipsc:16_53:len-3:strd-1
python data/scripts/create_video_tfrecord.py cfg=ipsc:16_53:len-3:strd-3
<a id="len_6___16_53_ipsc_"></a>
### len-6       @ 16_53/ipsc-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=ipsc:16_53:len-6:strd-1
python data/scripts/create_video_tfrecord.py cfg=ipsc:16_53:len-6:strd-6

<a id="0_37___ipsc_"></a>
## 0_37       @ ipsc-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=ipsc:0_37:len-2:strd-1
**fgs-4**
python data/scripts/create_video_tfrecord.py cfg=ipsc:0_37:len-2:strd-1:fgs-4 

<a id="54_126___ipsc_"></a>
## 54_126       @ ipsc-->p2s_vid_tf
<a id="len_2___54_126_ips_c_"></a>
### len-2       @ 54_126/ipsc-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:len-2:strd-1
python data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:len-2:strd-2
<a id="sample_8___len_2_54_126_ips_c_"></a>
#### sample-8       @ len-2/54_126/ipsc-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:len-2:strd-2:sample-8

<a id="len_3___54_126_ips_c_"></a>
### len-3       @ 54_126/ipsc-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:len-3:strd-1
python data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:len-3:strd-3
<a id="len_6___54_126_ips_c_"></a>
### len-6       @ 54_126/ipsc-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:len-6:strd-1
python data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:len-6:strd-6
<a id="sample_8___len_6_54_126_ips_c_"></a>
#### sample-8       @ len-6/54_126/ipsc-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:len-6:strd-6:sample-8
<a id="sample_4___len_6_54_126_ips_c_"></a>
#### sample-4       @ len-6/54_126/ipsc-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:len-6:strd-6:sample-4

<a id="acamp_"></a>
# acamp
<a id="1k8_vid_entire_seq___acam_p_"></a>
## 1k8_vid_entire_seq       @ acamp-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=acamp:1k8_vid_entire_seq:len-2:strd-1
python data/scripts/create_video_tfrecord.py cfg=acamp:1k8_vid_entire_seq:len-2:strd-2
<a id="1k8_vid_entire_seq_inv___acam_p_"></a>
## 1k8_vid_entire_seq_inv       @ acamp-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=acamp:1k8_vid_entire_seq_inv:len-2:strd-1
python data/scripts/create_video_tfrecord.py cfg=acamp:1k8_vid_entire_seq_inv:len-2:strd-2
<a id="10k6_vid_entire_seq___acam_p_"></a>
## 10k6_vid_entire_seq       @ acamp-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=acamp:10k6_vid_entire_seq:len-2:strd-1
python data/scripts/create_video_tfrecord.py cfg=acamp:10k6_vid_entire_seq:len-2:strd-2
<a id="10k6_vid_entire_seq_inv___acam_p_"></a>
## 10k6_vid_entire_seq_inv       @ acamp-->p2s_vid_tf
python data/scripts/create_video_tfrecord.py cfg=acamp:10k6_vid_entire_seq_inv:len-2:strd-1
python data/scripts/create_video_tfrecord.py cfg=acamp:10k6_vid_entire_seq_inv:len-2:strd-2
