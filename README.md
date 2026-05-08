# SLIP-RS: Structured-Attribute Language-Image Pre-Training for Remote Sensing Object Detection (ICML 2026)

<p align="center">
  <img src="./figures/motivation.png" width="800"/>
</p>

---

## 🔍 Abstract

Existing language-image pre-training methods for remote sensing object detection are constrained by Monolithic Label Learning, which relies on exhaustively enumerating open-set categories via black-box data to acquire fine-grained representations. This paradigm introduces a strong dependency on large-scale labeled data, which is fundamentally incompatible with the inherent data scarcity in remote sensing scenarios.

To transcend this bottleneck, we propose **SLIP-RS**, a **Structured-Attribute Decoupling Paradigm** that maps the open-ended category space into a finite and physically meaningful attribute space. This formulation enables fine-grained discriminability through explicit structural reasoning.

Our approach is built upon two key technical pillars:

- **Structured-Attribute Contrastive Learning (SACL)**  
  Enforces the learning of disentangled intrinsic visual representations via combinatorial attribute augmentation.

- **Conformal Attribute Reliability Engine (CARE)**  
  Leverages conformal prediction theory to distill high-fidelity supervision from noisy data sources, resulting in **RS-Attribute-15M**, the largest remote sensing attribute dataset with over **15 million annotations**.

Extensive experiments demonstrate that **SLIP-RS** achieves state-of-the-art performance in both **fine-grained object detection** and **cross-domain generalization**.

---

## Method Overview

<p align="center">
  <img src="./figures/pipeline.png" width="800"/>
</p>

---


## SACL for Classification

### Environment

```bash
conda create -n remoteclip_ft python=3.10 -y
conda activate remoteclip_ft

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### Data
Coming Soon

### Train
```bash
bash ./scripts/train_dist.sh
```


## SACL for Detection

### Enviroment
```bash
conda create -n sliprs python==3.10 -y
conda activate sliprs
cd mmdetection_sliprs
ip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -U openmim
mim install mmcv-full==1.7.1
cd mmdetection_sliprs
pip install -v -e .
pip install ftfy regex numpy==1.26.1 yapf==0.40.1
```

### Data
Coming Soon

### Train
```bash
bash ./tools/dist_train.sh ./sliprs_configs/slip-rs_convnext-t_lora-clip_fpn_1x_rs-attri.py 8
```

### Test
Our pretrained model weights:
| Model     | Weight                                                                                                  |
|-----------|---------------------------------------------------------------------------------------------------------|
| SLIP-RS-T | [Pretrained Weight](https://drive.google.com/file/d/1enHOD4X827pgObkUG45hU9H6bPtiixeL/view?usp=sharing) |
| SLIP-RS-L | [Pretrained Weight](https://drive.google.com/file/d/1_upTH-zclhUcB_CrS3iYuUAiuN5Y153x/view?usp=sharing) |


```bash
bash ./tools/dist_test.sh ./sliprs_configs/slip-rs_convnext-t_lora-clip_fpn_1x_rs-attri.py ./path/to/SLIP_RS_T.pth 8 --eval bbox
```

### Visualization
```bash
python ./tools/sliprs_infer_visualize.py ./sliprs_configs/slip-rs_convnext-t_lora-clip_fpn_1x.py ./path/to/SLIP_RS_T.pth ./tools/plane.png --prompt ['plane+twin-engines', 'plane+four-engines'] --out-dir ./
```