<!-- MarkdownTOC -->

- [16_53-resize-640       @ ipsc](#16_53_resize_640___ipsc_)
    - [size-80       @ 16_53-resize-640](#size_80___16_53_resize_640_)
        - [seq-0       @ size-80/16_53-resize-640](#seq_0___size_80_16_53_resize_640_)
        - [seq-1       @ size-80/16_53-resize-640](#seq_1___size_80_16_53_resize_640_)
    - [size-160       @ 16_53-resize-640](#size_160___16_53_resize_640_)
- [16_53-resize-320       @ ipsc](#16_53_resize_320___ipsc_)
        - [size-80       @ 16_53-resize-320/](#size_80___16_53_resize_320__)
        - [size-160       @ 16_53-resize-320/](#size_160___16_53_resize_320__)
- [54_126-resize-320       @ ipsc](#54_126_resize_320___ipsc_)
        - [size-80       @ 54_126-resize-320/](#size_80___54_126_resize_320_)
        - [size-160       @ 54_126-resize-320/](#size_160___54_126_resize_320_)

<!-- /MarkdownTOC -->

<a id="16_53_resize_640___ipsc_"></a>
# 16_53-resize-640       @ ipsc-->p2s_seg_tfrecord
<a id="size_80___16_53_resize_640_"></a>
## size-80       @ 16_53-resize-640-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:16_53:size-80:smin-0:smax-0:rmin-15:rmax-345:rnum-0:flip-0:resize-640:gz
<a id="seq_0___size_80_16_53_resize_640_"></a>
### seq-0       @ size-80/16_53-resize-640-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:16_53:size-80:smin-0:smax-0:rmin-15:rmax-345:rnum-0:flip-0:resize-640:gz:seq-0
<a id="seq_1___size_80_16_53_resize_640_"></a>
### seq-1       @ size-80/16_53-resize-640-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:16_53:size-80:smin-0:smax-0:rmin-15:rmax-345:rnum-0:flip-0:resize-640:gz:seq-1

<a id="size_160___16_53_resize_640_"></a>
## size-160       @ 16_53-resize-640-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:16_53:gz:size-160:smin-0:smax-0:rmin-15:rmax-345:rnum-0:flip-0:resize-640

<a id="16_53_resize_320___ipsc_"></a>
# 16_53-resize-320       @ ipsc-->p2s_seg_tfrecord
<a id="size_80___16_53_resize_320__"></a>
### size-80       @ 16_53-resize-320/-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:16_53:size-80:smin-0:smax-0:rmin-15:rmax-345:rnum-0:flip-0:resize-320:gz
<a id="size_160___16_53_resize_320__"></a>
### size-160       @ 16_53-resize-320/-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:16_53:gz:size-160:smin-0:smax-0:rmin-15:rmax-345:rnum-0:flip-0:resize-320

<a id="54_126_resize_320___ipsc_"></a>
# 54_126-resize-320       @ ipsc-->p2s_seg_tfrecord
<a id="size_80___54_126_resize_320_"></a>
### size-80       @ 54_126-resize-320/-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:54_126:size-80:smin-0:smax-0:rmin-15:rmax-345:rnum-0:flip-0:resize-320:gz
<a id="size_160___54_126_resize_320_"></a>
### size-160       @ 54_126-resize-320/-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:54_126:gz:size-160:smin-0:smax-0:rmin-15:rmax-345:rnum-0:flip-0:resize-320