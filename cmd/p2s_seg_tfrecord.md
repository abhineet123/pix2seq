<!-- MarkdownTOC -->

- [ipsc       @ tfrecord](#ipsc___tfrecord_)
    - [16_53       @ ipsc](#16_53___ipsc_)
        - [size-80       @ 16_53/ipsc](#size_80___16_53_ipsc_)
        - [size-160       @ 16_53/ipsc](#size_160___16_53_ipsc_)

<!-- /MarkdownTOC -->

<a id="ipsc___tfrecord_"></a>
# ipsc       @ tfrecord-->p2s_setup
<a id="16_53___ipsc_"></a>
## 16_53       @ ipsc-->p2s_seg_tfrecord
<a id="size_80___16_53_ipsc_"></a>
### size-80       @ 16_53/ipsc-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:16_53:size-80:smin-0:smax-0:rmin-15:rmax-345:rnum-0:flip-0:resize-640:gz
<a id="size_160___16_53_ipsc_"></a>
### size-160       @ 16_53/ipsc-->p2s_seg_tfrecord
python3 data/scripts/create_seg_tfrecord.py cfg=ipsc:16_53:gz:size-160:smin-0:smax-0:rmin-15:rmax-345:rnum-0:flip-0:resize-640
