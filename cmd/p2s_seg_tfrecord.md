<!-- MarkdownTOC -->

- [ipsc       @ tfrecord](#ipsc___tfrecord_)
    - [16_53-resize-640       @ ipsc](#16_53_resize_640___ipsc_)
        - [size-80       @ 16_53-resize-640/ipsc](#size_80___16_53_resize_640_ips_c_)
            - [seq-0       @ size-80/16_53-resize-640/ipsc](#seq_0___size_80_16_53_resize_640_ips_c_)
            - [seq-1       @ size-80/16_53-resize-640/ipsc](#seq_1___size_80_16_53_resize_640_ips_c_)
        - [size-160       @ 16_53-resize-640/ipsc](#size_160___16_53_resize_640_ips_c_)
    - [16_53-resize-320       @ ipsc](#16_53_resize_320___ipsc_)
        - [size-80       @ 16_53-resize-320/ipsc](#size_80___16_53_resize_320_ips_c_)
        - [size-160       @ 16_53-resize-320/ipsc](#size_160___16_53_resize_320_ips_c_)

<!-- /MarkdownTOC -->

<a id="ipsc___tfrecord_"></a>
# ipsc       @ tfrecord-->p2s_setup
<a id="16_53_resize_640___ipsc_"></a>
## 16_53-resize-640       @ ipsc-->p2s_seg_tfrecord
<a id="size_80___16_53_resize_640_ips_c_"></a>
### size-80       @ 16_53-resize-640/ipsc-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:16_53:size-80:smin-0:smax-0:rmin-15:rmax-345:rnum-0:flip-0:resize-640:gz
<a id="seq_0___size_80_16_53_resize_640_ips_c_"></a>
#### seq-0       @ size-80/16_53-resize-640/ipsc-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:16_53:size-80:smin-0:smax-0:rmin-15:rmax-345:rnum-0:flip-0:resize-640:gz:seq-0
<a id="seq_1___size_80_16_53_resize_640_ips_c_"></a>
#### seq-1       @ size-80/16_53-resize-640/ipsc-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:16_53:size-80:smin-0:smax-0:rmin-15:rmax-345:rnum-0:flip-0:resize-640:gz:seq-1

<a id="size_160___16_53_resize_640_ips_c_"></a>
### size-160       @ 16_53-resize-640/ipsc-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:16_53:gz:size-160:smin-0:smax-0:rmin-15:rmax-345:rnum-0:flip-0:resize-640

<a id="16_53_resize_320___ipsc_"></a>
## 16_53-resize-320       @ ipsc-->p2s_seg_tfrecord
<a id="size_80___16_53_resize_320_ips_c_"></a>
### size-80       @ 16_53-resize-320/ipsc-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:16_53:size-80:smin-0:smax-0:rmin-15:rmax-345:rnum-0:flip-0:resize-320:gz
<a id="size_160___16_53_resize_320_ips_c_"></a>
### size-160       @ 16_53-resize-320/ipsc-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:16_53:gz:size-160:smin-0:smax-0:rmin-15:rmax-345:rnum-0:flip-0:resize-320
