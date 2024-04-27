<!-- MarkdownTOC -->

- [detrac       @ tfrecord](#detrac___tfrecord_)
    - [0_19       @ detrac](#0_19___detrac_)
    - [0_9       @ detrac](#0_9___detrac_)
    - [49_68       @ detrac](#49_68___detrac_)
- [mnist       @ tfrecord](#mnist___tfrecord_)
    - [640-1       @ mnist](#640_1___mnis_t_)
        - [frame-0_1       @ 640-1/mnist](#frame_0_1___640_1_mnis_t_)
        - [frame-2_3       @ 640-1/mnist](#frame_2_3___640_1_mnis_t_)
    - [640-3       @ mnist](#640_3___mnis_t_)
    - [640-5       @ mnist](#640_5___mnis_t_)
- [ipsc       @ tfrecord](#ipsc___tfrecord_)
    - [0_1       @ ipsc](#0_1___ipsc_)
    - [2_3       @ ipsc](#2_3___ipsc_)
    - [16_53       @ ipsc](#16_53___ipsc_)
    - [0_37       @ ipsc](#0_37___ipsc_)
    - [54_126       @ ipsc](#54_126___ipsc_)
    - [0_1       @ ipsc](#0_1___ipsc__1)
    - [0_15       @ ipsc](#0_15___ipsc_)
    - [38_53       @ ipsc](#38_53___ipsc_)

<!-- /MarkdownTOC -->
<a id="detrac___tfrecord_"></a>
# detrac       @ tfrecord-->p2s_setup
<a id="0_19___detrac_"></a>
## 0_19       @ detrac-->p2s_tfrecord
python3 data/scripts/create_ipsc_tfrecord.py cfg=detrac:non_empty-0_19:gz
<a id="0_9___detrac_"></a>
## 0_9       @ detrac-->p2s_tfrecord
python3 data/scripts/create_ipsc_tfrecord.py cfg=detrac:non_empty-0_9:gz
<a id="49_68___detrac_"></a>
## 49_68       @ detrac-->p2s_tfrecord
python3 data/scripts/create_ipsc_tfrecord.py cfg=detrac:non_empty-49_68:gz

<a id="mnist___tfrecord_"></a>
# mnist       @ tfrecord-->p2s_setup
<a id="640_1___mnis_t_"></a>
## 640-1       @ mnist-->p2s_tfrecord
python3 data/scripts/create_ipsc_tfrecord.py cfg=mnist:640-1:12_1000:train:gz
<a id="frame_0_1___640_1_mnis_t_"></a>
### frame-0_1       @ 640-1/mnist-->p2s_tfrecord
python3 data/scripts/create_ipsc_tfrecord.py cfg=mnist:640-1:12_1000:train:frame-0_1:gz
<a id="frame_2_3___640_1_mnis_t_"></a>
### frame-2_3       @ 640-1/mnist-->p2s_tfrecord
python3 data/scripts/create_ipsc_tfrecord.py cfg=mnist:640-1:12_1000:train:frame-2_3:gz
python3 data/scripts/create_ipsc_tfrecord.py cfg=mnist:640-1:12_1000:test:frame-2_3:gz
<a id="640_3___mnis_t_"></a>
## 640-3       @ mnist-->p2s_tfrecord
python3 data/scripts/create_ipsc_tfrecord.py cfg=mnist:640-3:12_1000:train:gz
<a id="640_5___mnis_t_"></a>
## 640-5       @ mnist-->p2s_tfrecord
python3 data/scripts/create_ipsc_tfrecord.py cfg=mnist:640-5:12_1000:train:gz
python3 data/scripts/create_ipsc_tfrecord.py cfg=mnist:640-5:12_1000:test:gz

python3 data/scripts/create_ipsc_tfrecord.py cfg=mnist:640-5:12_1000:test:seq-0_5:frame-0_5:gz

<a id="ipsc___tfrecord_"></a>
# ipsc       @ tfrecord-->p2s_setup
<a id="0_1___ipsc_"></a>
## 0_1       @ ipsc-->p2s_tfrecord
python3 data/scripts/create_ipsc_tfrecord.py cfg=ipsc:0_1:gz:mask
<a id="2_3___ipsc_"></a>
## 2_3       @ ipsc-->p2s_tfrecord
python3 data/scripts/create_ipsc_tfrecord.py cfg=ipsc:2_3:gz
<a id="16_53___ipsc_"></a>
## 16_53       @ ipsc-->p2s_tfrecord
python3 data/scripts/create_ipsc_tfrecord.py cfg=ipsc:16_53:gz
<a id="0_37___ipsc_"></a>
## 0_37       @ ipsc-->p2s_tfrecord
python3 data/scripts/create_ipsc_tfrecord.py cfg=ipsc:0_37:gz
<a id="38_53___ipsc_"></a>
<a id="54_126___ipsc_"></a>
## 54_126       @ ipsc-->p2s_tfrecord
python3 data/scripts/create_ipsc_tfrecord.py cfg=ipsc:54_126:gz
python3 data/scripts/create_ipsc_tfrecord.py cfg=ipsc:54_126:gz:strd-5
python3 data/scripts/create_ipsc_tfrecord.py cfg=ipsc:54_126:gz:strd-8

<a id="0_1___ipsc__1"></a>
## 0_1       @ ipsc-->p2s_tfrecord
python data/scripts/create_ipsc_tfrecord.py --ann_file=ext_reorg_roi_g2_0_1.json --n_proc=0
<a id="0_15___ipsc_"></a>
## 0_15       @ ipsc-->p2s_tfrecord
python data/scripts/create_ipsc_tfrecord.py --ann_file=ext_reorg_roi_g2_0_15.json --n_proc=0
<a id="38_53___ipsc_"></a>
## 38_53       @ ipsc-->p2s_tfrecord
python3 data/scripts/create_ipsc_tfrecord.py --ann_file=ext_reorg_roi_g2_38_53.json --n_proc=0
