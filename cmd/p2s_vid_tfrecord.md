<!-- MarkdownTOC -->

- [gram       @ tfrecord](#gram___tfrecord_)
    - [0_1       @ gram](#0_1___gram_)
        - [len-9       @ 0_1/gram](#len_9___0_1_gram_)
        - [len-14       @ 0_1/gram](#len_14___0_1_gram_)
            - [0_2000       @ len-14/0_1/gram](#0_2000___len_14_0_1_gra_m_)
            - [3000_5000       @ len-14/0_1/gram](#3000_5000___len_14_0_1_gra_m_)
        - [len-16       @ 0_1/gram](#len_16___0_1_gram_)
- [idot       @ gram](#idot___gram_)
- [detrac       @ tfrecord](#detrac___tfrecord_)
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
- [mnist       @ tfrecord](#mnist___tfrecord_)
    - [640-1       @ mnist](#640_1___mnis_t_)
        - [len-2       @ 640-1/mnist](#len_2___640_1_mnis_t_)
        - [test       @ 640-1/mnist](#test___640_1_mnis_t_)
            - [strd-2       @ test/640-1/mnist](#strd_2___test_640_1_mnist_)
        - [len-3       @ 640-1/mnist](#len_3___640_1_mnis_t_)
        - [test       @ 640-1/mnist](#test___640_1_mnis_t__1)
            - [strd-3       @ test/640-1/mnist](#strd_3___test_640_1_mnist_)
        - [len-9       @ 640-1/mnist](#len_9___640_1_mnis_t_)
    - [640-3       @ mnist](#640_3___mnis_t_)
    - [640-5       @ mnist](#640_5___mnis_t_)
        - [len-2       @ 640-5/mnist](#len_2___640_5_mnis_t_)
        - [len-3       @ 640-5/mnist](#len_3___640_5_mnis_t_)
        - [len-4       @ 640-5/mnist](#len_4___640_5_mnis_t_)
        - [len-6       @ 640-5/mnist](#len_6___640_5_mnis_t_)
        - [len-9       @ 640-5/mnist](#len_9___640_5_mnis_t_)

<!-- /MarkdownTOC -->

<a id="gram___tfrecord_"></a>
# gram       @ tfrecord-->p2s_setup
<a id="0_1___gram_"></a>
## 0_1       @ gram-->p2s_vid_tfrecord
<a id="len_9___0_1_gram_"></a>
### len-9       @ 0_1/gram-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=gram:0_1:gz:len-9:strd-1

python3 data/scripts/create_video_tfrecord.py cfg=gram:0_1:gz:len-9:strd-9

<a id="len_14___0_1_gram_"></a>
### len-14       @ 0_1/gram-->p2s_vid_tfrecord
<a id="0_2000___len_14_0_1_gra_m_"></a>
#### 0_2000       @ len-14/0_1/gram-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=gram:0_1:frame-0_2000:gz:len-14:strd-1
<a id="3000_5000___len_14_0_1_gra_m_"></a>
#### 3000_5000       @ len-14/0_1/gram-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=gram:0_1:frame-3000_5000:gz:len-14:strd-14

<a id="len_16___0_1_gram_"></a>
### len-16       @ 0_1/gram-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=gram:0_1:frame-0_2000:gz:len-16:strd-1

<a id="idot___gram_"></a>
# idot       @ gram-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=idot:gz:len-9:strd-1

<a id="detrac___tfrecord_"></a>
# detrac       @ tfrecord-->p2s_setup
<a id="0_19___detrac_"></a>
## 0_19       @ detrac-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:gz:len-2:strd-1
<a id="strd_2___0_19_detra_c_"></a>
### strd-2       @ 0_19/detrac-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:gz:len-2:strd-2
<a id="len_3___0_19_detra_c_"></a>
### len-3       @ 0_19/detrac-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:gz:len-3:strd-1
<a id="len_4___0_19_detra_c_"></a>
### len-4       @ 0_19/detrac-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:gz:len-4:strd-1
<a id="len_6___0_19_detra_c_"></a>
### len-6       @ 0_19/detrac-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:gz:len-6:strd-1
<a id="len_8___0_19_detra_c_"></a>
### len-8       @ 0_19/detrac-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:gz:len-8:strd-1
<a id="len_9___0_19_detra_c_"></a>
### len-9       @ 0_19/detrac-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:gz:len-9:strd-1

<a id="0_9___detrac_"></a>
## 0_9       @ detrac-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_9:gz:len-2:strd-1
<a id="strd_2___0_9_detrac_"></a>
### strd-2       @ 0_9/detrac-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_9:gz:len-2:strd-2

<a id="49_68___detrac_"></a>
## 49_68       @ detrac-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_68:gz:len-2:strd-1
<a id="strd_2___49_68_detrac_"></a>
### strd-2       @ 49_68/detrac-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_68:gz:len-2:strd-2

<a id="ips_c_"></a>
# ipsc
<a id="0_4___ipsc_"></a>
##0_4        @ ipsc-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:gz:shards-2:len-2:strd-1 ann_file=ext_reorg_roi_g2_0_4
**12094**
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:gz:shards-2:len-2:strd-1 ann_file=ext_reorg_roi_12094_0_4
**12094_short**
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:gz:shards-2:len-2:strd-1 ann_file=ext_reorg_roi_12094_short_0_4

<a id="5_9___ipsc_"></a>
## 5_9       @ ipsc-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:gz:shards-2:len-2:strd-1 ann_file=ext_reorg_roi_g2_5_9
**fg-4**
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:gz:shards-2:len-2:strd-1:fg-4 ann_file=ext_reorg_roi_g2_5_9

<a id="16_53___ipsc_"></a>
## 16_53       @ ipsc-->p2s_vid_tfrecord
<a id="len_2___16_53_ipsc_"></a>
### len-2       @ 16_53/ipsc-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:16_53:gz:shards-32:len-2:strd-1
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:16_53:gz:shards-32:len-2:strd-2
<a id="len_3___16_53_ipsc_"></a>
### len-3       @ 16_53/ipsc-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:16_53:gz:shards-32:len-3:strd-1
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:16_53:gz:shards-32:len-3:strd-3
<a id="len_6___16_53_ipsc_"></a>
### len-6       @ 16_53/ipsc-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:16_53:gz:shards-32:len-6:strd-1
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:16_53:gz:shards-32:len-6:strd-6

<a id="0_37___ipsc_"></a>
## 0_37       @ ipsc-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:0_37:gz:shards-32:len-2:strd-1
**fg-4**
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:0_37:gz:shards-32:len-2:strd-1:fg-4 

<a id="54_126___ipsc_"></a>
## 54_126       @ ipsc-->p2s_vid_tfrecord
<a id="len_2___54_126_ips_c_"></a>
### len-2       @ 54_126/ipsc-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:gz:shards-32:len-2:strd-1
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:gz:shards-32:len-2:strd-2
<a id="sample_8___len_2_54_126_ips_c_"></a>
#### sample-8       @ len-2/54_126/ipsc-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:gz:shards-32:len-2:strd-2:sample-8

<a id="len_3___54_126_ips_c_"></a>
### len-3       @ 54_126/ipsc-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:gz:shards-32:len-3:strd-1
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:gz:shards-32:len-3:strd-3
<a id="len_6___54_126_ips_c_"></a>
### len-6       @ 54_126/ipsc-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:gz:shards-32:len-6:strd-1
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:gz:shards-32:len-6:strd-6
<a id="sample_8___len_6_54_126_ips_c_"></a>
#### sample-8       @ len-6/54_126/ipsc-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:gz:shards-32:len-6:strd-6:sample-8
<a id="sample_4___len_6_54_126_ips_c_"></a>
#### sample-4       @ len-6/54_126/ipsc-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=ipsc:54_126:gz:shards-32:len-6:strd-6:sample-4


<a id="mnist___tfrecord_"></a>
# mnist       @ tfrecord-->p2s_setup
<a id="640_1___mnis_t_"></a>
## 640-1       @ mnist-->p2s_vid_tfrecord
<a id="len_2___640_1_mnis_t_"></a>
### len-2       @ 640-1/mnist-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-1:12_1000:gz:len-2:strd-1:train
<a id="test___640_1_mnis_t_"></a>
### test       @ 640-1/mnist-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-1:12_1000:gz:len-2:strd-1:test
<a id="strd_2___test_640_1_mnist_"></a>
#### strd-2       @ test/640-1/mnist-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-1:12_1000:gz:len-2:strd-2:test

<a id="len_3___640_1_mnis_t_"></a>
### len-3       @ 640-1/mnist-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-1:12_1000:gz:len-3:strd-1:proc-12
<a id="test___640_1_mnis_t__1"></a>
### test       @ 640-1/mnist-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-1:12_1000:gz:len-3:strd-1:test
<a id="strd_3___test_640_1_mnist_"></a>
#### strd-3       @ test/640-1/mnist-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-1:12_1000:gz:len-3:strd-3:test

<a id="len_9___640_1_mnis_t_"></a>
### len-9       @ 640-1/mnist-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-1:12_1000:gz:len-9:strd-1:proc-12:train

<a id="640_3___mnis_t_"></a>
## 640-3       @ mnist-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-3:12_1000:gz:len-2:strd-1:proc-6

<a id="640_5___mnis_t_"></a>
## 640-5       @ mnist-->p2s_vid_tfrecord
<a id="len_2___640_5_mnis_t_"></a>
### len-2       @ 640-5/mnist-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-2:strd-1:proc-12:train

python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-2:strd-1:proc-12:test
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-2:strd-2:proc-12:test

<a id="len_3___640_5_mnis_t_"></a>
### len-3       @ 640-5/mnist-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-3:strd-1:proc-12:train

python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-3:strd-1:proc-12:test
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-3:strd-3:proc-12:test

<a id="len_4___640_5_mnis_t_"></a>
### len-4       @ 640-5/mnist-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-4:strd-1:proc-12:train

python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-4:strd-1:proc-12:test
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-4:strd-4:proc-12:test
<a id="len_6___640_5_mnis_t_"></a>
### len-6       @ 640-5/mnist-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-6:strd-1:proc-12:train

python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-6:strd-1:proc-12:test
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-6:strd-6:proc-12:test

<a id="len_9___640_5_mnis_t_"></a>
### len-9       @ 640-5/mnist-->p2s_vid_tfrecord
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-9:strd-1:proc-12:train

python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-9:strd-1:proc-12:test
python3 data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-9:strd-9:proc-12:test
