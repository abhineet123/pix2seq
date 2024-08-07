<!-- MarkdownTOC -->

- [resnet-640](#resnet_64_0_)
    - [detrac-non_empty       @ resnet-640](#detrac_non_empty___resnet_640_)
        - [0_19       @ detrac-non_empty/resnet-640](#0_19___detrac_non_empty_resnet_64_0_)
            - [on-train       @ 0_19/detrac-non_empty/resnet-640](#on_train___0_19_detrac_non_empty_resnet_640_)
            - [on-test       @ 0_19/detrac-non_empty/resnet-640](#on_test___0_19_detrac_non_empty_resnet_640_)
        - [0_9       @ detrac-non_empty/resnet-640](#0_9___detrac_non_empty_resnet_64_0_)

<!-- /MarkdownTOC -->
<a id="resnet_64_0_"></a>
# resnet-640 
<a id="detrac_non_empty___resnet_640_"></a>
## detrac-non_empty       @ resnet-640-->p2s-isl
<a id="0_19___detrac_non_empty_resnet_64_0_"></a>
### 0_19       @ detrac-non_empty/resnet-640-->p2s-isl
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,detrac-non_empty-0_19,batch-3,dbg-0,dyn-1,dist-0
<a id="on_train___0_19_detrac_non_empty_resnet_640_"></a>
#### on-train       @ 0_19/detrac-non_empty/resnet-640-->p2s-isl
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=eval,m-resnet_640_detrac-non_empty-seq-0_19-batch_18,detrac-non_empty-0_19,batch-48,save-vis-1,dbg-0,dyn-1,dist-0
<a id="on_test___0_19_detrac_non_empty_resnet_640_"></a>
#### on-test       @ 0_19/detrac-non_empty/resnet-640-->p2s-isl
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=eval,m-resnet_640_detrac-non_empty-seq-0_19-batch_18,detrac-non_empty-49_68,batch-48,save-vis-1,dbg-0,dyn-1,dist-0
<a id="0_9___detrac_non_empty_resnet_64_0_"></a>
### 0_9       @ detrac-non_empty/resnet-640-->p2s-isl
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,detrac-non_empty-0_9,batch-18,dbg-0,dyn-1,dist-0

