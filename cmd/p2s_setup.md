<!-- MarkdownTOC -->

- [wsl](#wsl_)
- [x99 / 18.04](#x99___18_0_4_)
    - [upgrade       @ x99_/_18.04](#upgrade___x99___18_0_4_)
- [virtualenv](#virtualen_v_)
    - [windows       @ virtualenv](#windows___virtualenv_)
- [install](#install_)
    - [tensorflow       @ install](#tensorflow___instal_l_)
        - [wsl-gpu       @ tensorflow/install](#wsl_gpu___tensorflow_install_)
        - [ubuntu22.04       @ tensorflow/install](#ubuntu22_04___tensorflow_install_)
        - [all       @ tensorflow/install](#all___tensorflow_install_)
    - [netifaces       @ install](#netifaces___instal_l_)
- [soft-links](#soft_link_s_)
- [pretrained](#pretraine_d_)
    - [install_gcloud       @ pretrained](#install_gcloud___pretrained_)
        - [ubuntu       @ install_gcloud/pretrained](#ubuntu___install_gcloud_pretraine_d_)
    - [resnet_640       @ pretrained](#resnet_640___pretrained_)
    - [vit_b       @ pretrained](#vit_b___pretrained_)
    - [vit_l       @ pretrained](#vit_l___pretrained_)
- [secondary ethernet](#secondary_ethernet_)
- [bugs](#bug_s_)

<!-- /MarkdownTOC -->

<a id="wsl_"></a>
# wsl
https://github.com/microsoft/WSL/issues/4585
cat /etc/resolv.conf
New-NetFirewallRule -DisplayName "WSL" -Direction Inbound  -LocalAddress 172.23.0.1 -Action Allow

<a id="x99___18_0_4_"></a>
# x99 / 18.04
__deadsnakes no longer supports ubuntu 18.04 which has reached end-of-life__
https://github.com/deadsnakes/issues/issues/251
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3\.10

<a id="upgrade___x99___18_0_4_"></a>
## upgrade       @ x99_/_18.04-->p2s_setup
sudo apt upgrade
sudo apt dist-upgrade
sudo do-release-upgrade

python3.8 -m pip install virtualenv virtualenvwrapper

sudo apt install python3.10

<a id="virtualen_v_"></a>
# virtualenv
python3.10 -m pip install virtualenv virtualenvwrapper
mkvirtualenv -p python3.10  pix2seq
workon pix2seq

nano ~/.bashrc
alias p2s='workon pix2seq'
source ~/.bashrc

<a id="windows___virtualenv_"></a>
## windows       @ virtualenv-->p2s_setup
python310 -m virtualenv pix2seq
pix2seq\Scripts\activate

<a id="install_"></a>
# install
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

<a id="tensorflow___instal_l_"></a>
## tensorflow       @ install-->p2s_setup
python -m pip install tensorflow==2.14
python -m pip install tensorflow-text
<a id="wsl_gpu___tensorflow_install_"></a>
### wsl-gpu       @ tensorflow/install-->p2s_setup
install cuda:
https://docs.nvidia.com/cuda/wsl-user-guide/index.html
latest:
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
cuda 12.2 for tf 2.15:
https://developer.nvidia.com/cuda-12-2-2-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
12.x for tensorflow 2.15 and 11.x for tensorflow 2.10

<a id="ubuntu22_04___tensorflow_install_"></a>
### ubuntu22.04       @ tensorflow/install-->p2s_setup
update driver:
```
ubuntu-drivers devices
sudo apt install nvidia-driver-535
```

CUDA 12.2:
https://developer.nvidia.com/cuda-12-2-2-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.2-535.104.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.2-535.104.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-12-2

```

<a id="all___tensorflow_install_"></a>
### all       @ tensorflow/install-->p2s_setup
download cudnn tar from
https://developer.nvidia.com/rdp/cudnn-download
and extract into cuda-12.2/targets/ folder
```
tar xvf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
sudo mv cudnn-linux-x86_64-8.9.7.29_cuda12-archive/lib/* /usr/local/cuda-12.2/targets/x86_64-linux/lib/
sudo mv cudnn-linux-x86_64-8.9.7.29_cuda12-archive/include/* /usr/local/cuda-12.2/targets/x86_64-linux/include/
```

add following environment variables to bashrc as well as the pycharm debug configuration window
```
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=$PATH:/usr/local/cuda-12.2/bin
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-12.2
```

check tf:
```
python3 -c "import tensorflow as tf; print(tf.__version__)"
python3 -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

<a id="netifaces___instal_l_"></a>
## netifaces       @ install-->p2s_setup
sudo apt-get install python3.10-dev
pip install netifaces

<a id="soft_link_s_"></a>
# soft-links
ln -s /data datasets

mkdir /data/p2s_log
ln -s /data/p2s_log log

mkdir /data/p2s_log/pretrained
ln -s /data/p2s_log/pretrained pretrained

<a id="pretraine_d_"></a>
# pretrained

<a id="install_gcloud___pretrained_"></a>
## install_gcloud       @ pretrained-->p2s_setup
<a id="ubuntu___install_gcloud_pretraine_d_"></a>
### ubuntu       @ install_gcloud/pretrained-->p2s_setup
https://cloud.google.com/sdk/docs/install#deb

sudo apt-get install apt-transport-https ca-certificates gnupg curl sudo
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update && sudo apt-get install google-cloud-cli
gcloud init

<a id="resnet_640___pretrained_"></a>
## resnet_640       @ pretrained-->p2s_setup
gsutil -m cp "gs://pix2seq/coco_det_finetune/resnet_640x640/checkpoint"  "gs://pix2seq/coco_det_finetune/resnet_640x640/ckpt-74844.data-00000-of-00001"  "gs://pix2seq/coco_det_finetune/resnet_640x640/ckpt-74844.index" "gs://pix2seq/coco_det_finetune/resnet_640x640/config.json" "gs://pix2seq/coco_det_finetune/resnet_640x640/ev_object_detection_validation_p0.4_result.json" "gs://pix2seq/coco_det_finetune/resnet_640x640/ev_object_detection_validation_p0.4cocoeval.pkl"

gsutil -m cp -r "gs://pix2seq/coco_det_finetune/resnet_640x640"

<a id="vit_b___pretrained_"></a>
## vit_b       @ pretrained-->p2s_setup
gsutil -m cp -r "gs://pix2seq/coco_det_finetune/vit_b_1024x1024"  "gs://pix2seq/coco_det_finetune/vit_b_1333x1333"  "gs://pix2seq/coco_det_finetune/vit_b_640x640" .

gsutil -m cp   "gs://pix2seq/coco_det_finetune/vit_b_640x640/checkpoint"   "gs://pix2seq/coco_det_finetune/vit_b_640x640/ckpt-112728.data-00000-of-00001"   "gs://pix2seq/coco_det_finetune/vit_b_640x640/ckpt-112728.index"   "gs://pix2seq/coco_det_finetune/vit_b_640x640/config.json"   "gs://pix2seq/coco_det_finetune/vit_b_640x640/ev_object_detection_validation_p0.4_result.json"   "gs://pix2seq/coco_det_finetune/vit_b_640x640/ev_object_detection_validation_p0.4cocoeval.pkl"   .

<a id="vit_l___pretrained_"></a>
## vit_l       @ pretrained-->p2s_setup
gsutil -m cp -r "gs://pix2seq/coco_det_finetune/vit_l_1024x1024"  "gs://pix2seq/coco_det_finetune/vit_l_1333x1333"  "gs://pix2seq/coco_det_finetune/vit_l_640x640"  .

gsutil -m cp -r "gs://pix2seq/coco_det_finetune/vit_l_1024x1024" .

<a id="secondary_ethernet_"></a>
# secondary ethernet
```
sudo ip route del 192.168.177.0/24
GRS:
sudo ip route add 192.168.177.0/24 dev enp6s0 metric 1
X99:
sudo ip route add 192.168.177.0/24 dev enp8s0 metric 1
E5G
sudo ip route add 192.168.177.0/24 dev enp0s25 metric 1

sudo apt install ethtool
sudo apt install bmon
```

<a id="bug_s_"></a>
# bugs
``Collective ops is aborted by: failed to connect to all addresses``
probably something to do with the dataset loader
https://github.com/tensorflow/tensorflow/issues/39122
https://github.com/tensorflow/tensorflow/issues/39099




