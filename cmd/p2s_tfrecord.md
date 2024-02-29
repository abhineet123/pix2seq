<!-- MarkdownTOC -->

- [detrac       @ tfrecord](#detrac___tfrecord_)
    - [0_19       @ detrac](#0_19___detrac_)
    - [0_9       @ detrac](#0_9___detrac_)
    - [49_68       @ detrac](#49_68___detrac_)
- [mnist       @ tfrecord](#mnist___tfrecord_)
    - [640-1       @ mnist](#640_1___mnis_t_)
    - [640-1       @ mnist](#640_1___mnis_t__1)
    - [640-3       @ mnist](#640_3___mnis_t_)
    - [640-5       @ mnist](#640_5___mnis_t_)
- [ipsc       @ tfrecord](#ipsc___tfrecord_)
    - [16_53       @ ipsc](#16_53___ipsc_)
    - [g2_0_1       @ ipsc](#g2_0_1___ipsc_)
    - [g2_0_15       @ ipsc](#g2_0_15___ipsc_)
    - [g2_0_37       @ ipsc](#g2_0_37___ipsc_)
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
<a id="640_1___mnis_t__1"></a>
## 640-1       @ mnist-->p2s_tfrecord
python3 data/scripts/create_ipsc_tfrecord.py cfg=mnist:640-1:12_1000:train:gz
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

<a id="16_53___ipsc_"></a>
## 16_53       @ ipsc-->p2s_tfrecord
python3 data/scripts/create_ipsc_tfrecord.py --ann_file=ext_reorg_roi_g2_16_53.json --n_proc=0

python3 data/scripts/create_ipsc_tfrecord.py cfg=ipsc:frame-16_53:gz

<a id="g2_0_1___ipsc_"></a>
## g2_0_1       @ ipsc-->p2s_tfrecord
python data/scripts/create_ipsc_tfrecord.py --ann_file=ext_reorg_roi_g2_0_1.json --n_proc=0
<a id="g2_0_15___ipsc_"></a>
## g2_0_15       @ ipsc-->p2s_tfrecord
python data/scripts/create_ipsc_tfrecord.py --ann_file=ext_reorg_roi_g2_0_15.json --n_proc=0
<a id="g2_0_37___ipsc_"></a>
## g2_0_37       @ ipsc-->p2s_tfrecord
python data/scripts/create_ipsc_tfrecord.py --ann_file=ext_reorg_roi_g2_0_37.json --n_proc=0
<a id="38_53___ipsc_"></a>
## 38_53       @ ipsc-->p2s_tfrecord
python3 data/scripts/create_ipsc_tfrecord.py --ann_file=ext_reorg_roi_g2_38_53.json --n_proc=0
