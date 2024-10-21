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
        - [len-64       @ 0_19/detrac](#len_64___0_19_detra_c_)
    - [0_9       @ detrac](#0_9___detrac_)
        - [strd-2       @ 0_9/detrac](#strd_2___0_9_detrac_)
    - [49_68       @ detrac](#49_68___detrac_)
        - [len-2       @ 49_68/detrac](#len_2___49_68_detrac_)
        - [len-6       @ 49_68/detrac](#len_6___49_68_detrac_)
        - [len-9       @ 49_68/detrac](#len_9___49_68_detrac_)
        - [len-64       @ 49_68/detrac](#len_64___49_68_detrac_)
    - [0_59       @ detrac](#0_59___detrac_)
        - [len-32       @ 0_59/detrac](#len_32___0_59_detra_c_)
        - [len-40       @ 0_59/detrac](#len_40___0_59_detra_c_)
        - [len-48       @ 0_59/detrac](#len_48___0_59_detra_c_)
        - [len-64       @ 0_59/detrac](#len_64___0_59_detra_c_)

<!-- /MarkdownTOC -->

<a id="gram___tfrecord_"></a>
# gram       @ tfrecord-->p2s_setup
<a id="0_1___gram_"></a>
## 0_1       @ gram-->tf_vid-isl
<a id="len_9___0_1_gram_"></a>
### len-9       @ 0_1/gram-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=gram:0_1:len-9:strd-1
python data/scripts/create_video_tfrecord.py cfg=gram:0_1:len-9:strd-9

<a id="len_14___0_1_gram_"></a>
### len-14       @ 0_1/gram-->tf_vid-isl
<a id="0_2000___len_14_0_1_gra_m_"></a>
#### 0_2000       @ len-14/0_1/gram-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=gram:0_1:frame-0_2000:len-14:strd-1
<a id="3000_5000___len_14_0_1_gra_m_"></a>
#### 3000_5000       @ len-14/0_1/gram-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=gram:0_1:frame-3000_5000:len-14:strd-14

<a id="len_16___0_1_gram_"></a>
### len-16       @ 0_1/gram-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=gram:0_1:frame-0_2000:len-16:strd-1

<a id="idot___gram_"></a>
# idot       @ gram-->p2s_vid_tfrecord
python data/scripts/create_video_tfrecord.py cfg=idot:len-9:strd-1
<a id="8_8___idot_"></a>
## 8_8       @ idot-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=idot:8_8:len-2:strd-10:fg-10

python data/scripts/create_video_tfrecord.py cfg=idot:8_8:len-3:strd-20:fg-10

python data/scripts/create_video_tfrecord.py cfg=idot:8_8:len-6:strd-50:fg-10

<a id="detra_c_"></a>
# detrac
<a id="0_0___detrac_"></a>
## 0_0       @ detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_0:len-2:strd-1

<a id="0_19___detrac_"></a>
## 0_19       @ detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:len-2:strd-1

<a id="strd_2___0_19_detra_c_"></a>
### strd-2       @ 0_19/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:len-2:strd-2
<a id="len_3___0_19_detra_c_"></a>
### len-3       @ 0_19/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:len-3:strd-1
<a id="len_4___0_19_detra_c_"></a>
### len-4       @ 0_19/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:len-4:strd-1
<a id="len_6___0_19_detra_c_"></a>
### len-6       @ 0_19/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:len-6:strd-1
<a id="len_8___0_19_detra_c_"></a>
### len-8       @ 0_19/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:len-8:strd-1
<a id="len_9___0_19_detra_c_"></a>
### len-9       @ 0_19/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:len-9:strd-1
<a id="len_32___0_19_detra_c_"></a>
<a id="len_64___0_19_detra_c_"></a>
### len-64       @ 0_19/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:len-64:strd-1:asi-0

<a id="0_9___detrac_"></a>
## 0_9       @ detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_9:len-2:strd-1
<a id="strd_2___0_9_detrac_"></a>
### strd-2       @ 0_9/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_9:len-2:strd-2

<a id="49_68___detrac_"></a>
## 49_68       @ detrac-->tf_vid-isl
<a id="len_2___49_68_detrac_"></a>
### len-2       @ 49_68/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_68:len-2:strd-1:asi-2
`strd-2` 
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_68:len-2:strd-2

<a id="len_6___49_68_detrac_"></a>
### len-6       @ 49_68/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_68:len-6:strd-1:asi

<a id="len_9___49_68_detrac_"></a>
### len-9       @ 49_68/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_68:len-9:strd-1:asi-2

<a id="len_64___49_68_detrac_"></a>
### len-64       @ 49_68/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_68:len-64:strd-1:asi-2

<a id="0_59___detrac_"></a>
## 0_59       @ detrac-->tf_vid-isl
<a id="len_32___0_59_detra_c_"></a>
### len-32       @ 0_59/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_59:len-32:strd-1:asi-0
<a id="len_40___0_59_detra_c_"></a>
### len-40       @ 0_59/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_59:len-40:strd-1:asi-0
<a id="len_48___0_59_detra_c_"></a>
### len-48       @ 0_59/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_59:len-48:strd-1:asi-0
<a id="len_64___0_59_detra_c_"></a>
### len-64       @ 0_59/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_59:len-64:strd-1:asi-0
