<!-- MarkdownTOC -->

- [16_53       @ ipsc](#16_53___ipsc_)
    - [res-2560       @ 16_53](#res_2560___16_5_3_)
        - [sz-640-sub-8       @ res-2560/16_53](#sz_640_sub_8___res_2560_16_53_)
        - [sz-640-sub-2       @ res-2560/16_53](#sz_640_sub_2___res_2560_16_53_)
    - [res-640       @ 16_53](#res_640___16_5_3_)
        - [sz-80       @ res-640/16_53](#sz_80___res_640_16_5_3_)
            - [seq-0       @ sz-80/res-640/16_53](#seq_0___sz_80_res_640_16_5_3_)
            - [seq-1       @ sz-80/res-640/16_53](#seq_1___sz_80_res_640_16_5_3_)
        - [sz-160       @ res-640/16_53](#sz_160___res_640_16_5_3_)
    - [res-320       @ 16_53](#res_320___16_5_3_)
        - [sz-80       @ res-320/16_53](#sz_80___res_320_16_5_3_)
        - [sz-80-aug       @ res-320/16_53](#sz_80_aug___res_320_16_5_3_)
        - [sz-160       @ res-320/16_53](#sz_160___res_320_16_5_3_)
        - [sz-160-aug       @ res-320/16_53](#sz_160_aug___res_320_16_5_3_)
- [54_126](#54_12_6_)
    - [res-320       @ 54_126](#res_320___54_126_)
        - [sz-80       @ res-320/54_126](#sz_80___res_320_54_126_)
        - [sz-160       @ res-320/54_126](#sz_160___res_320_54_126_)

<!-- /MarkdownTOC -->
<a id="16_53___ipsc_"></a>
# 16_53       @ ipsc-->p2s_seg_tfrecord
<a id="res_2560___16_5_3_"></a>
## res-2560       @ 16_53-->p2s_seg_tfrecord
<a id="sz_640_sub_8___res_2560_16_53_"></a>
### sz-640-sub-8       @ res-2560/16_53-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:16_53:sz-640:sub-8:res-2560:gz
`seq-0`
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:16_53:sz-640:sub-8:res-2560:gz:seq-0
<a id="sz_640_sub_2___res_2560_16_53_"></a>
### sz-640-sub-2       @ res-2560/16_53-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:16_53:sz-640:sub-2:res-2560:gz
<a id="res_640___16_5_3_"></a>
## res-640       @ 16_53-->p2s_seg_tfrecord
<a id="sz_80___res_640_16_5_3_"></a>
### sz-80       @ res-640/16_53-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:16_53:sz-80:res-640:gz
<a id="seq_0___sz_80_res_640_16_5_3_"></a>
#### seq-0       @ sz-80/res-640/16_53-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:16_53:sz-80:res-640:gz:seq-0
<a id="seq_1___sz_80_res_640_16_5_3_"></a>
#### seq-1       @ sz-80/res-640/16_53-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:16_53:sz-80:res-640:gz:seq-1

<a id="sz_160___res_640_16_5_3_"></a>
### sz-160       @ res-640/16_53-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:16_53:gz:sz-160:res-640

<a id="res_320___16_5_3_"></a>
## res-320       @ 16_53-->p2s_seg_tfrecord
<a id="sz_80___res_320_16_5_3_"></a>
### sz-80       @ res-320/16_53-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:16_53:gz:sz-80:res-320

<a id="sz_80_aug___res_320_16_5_3_"></a>
### sz-80-aug       @ res-320/16_53-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:16_53:gz:sz-80:res-320:strd-40_80:rot-15_345_3:flip-1

<a id="sz_160___res_320_16_5_3_"></a>
### sz-160       @ res-320/16_53-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:16_53:gz:sz-160:res-320

<a id="sz_160_aug___res_320_16_5_3_"></a>
### sz-160-aug       @ res-320/16_53-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:16_53:gz:sz-160:res-320:strd-40_160:rot-15_345_4:flip-1

<a id="54_12_6_"></a>
# 54_126
<a id="res_320___54_126_"></a>
## res-320       @ 54_126-->p2s_seg_tfrecord
<a id="sz_80___res_320_54_126_"></a>
### sz-80       @ res-320/54_126-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:54_126:sz-80:res-320:gz
<a id="sz_160___res_320_54_126_"></a>
### sz-160       @ res-320/54_126-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:54_126:gz:sz-160:res-320