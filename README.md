# PSW++: Improved Position-Sensitive Windowing strategy for Light Field Image Super-Resolution

👌 The repository already contains all training, validation and test codes as well as pre-trained models for NTIRE2024 challenge.

**Abstract:** 

**Author:** Angulia Yang, Kai Jin, Zeqiang Wei, Sha Guo, Mingzhi Gao, Xiuzhuang Zhou

If you find this work helpful, please consider citing the following papers:

```bibtex
@InProceedings{DistgEPIT,
    author    = {Jin, Kai and Yang, Angulia and Wei, Zeqiang and Guo, Sha and Gao, Mingzhi and Zhou, Xiuzhuang},
    title     = {DistgEPIT: Enhanced Disparity Learning for Light Field Image Super-Resolution},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
    year      = {2023},
}
```

## PSW++ Explaination

## News

✅ **Mar 22, 2024:** This repository contains official pytorch implementation of "PSW++: Improved Position-Sensitive Windowing strategy for Light Field Image Super-Resolution" in **? solutions 👑** in [NTIRE2024 Light-Field Super Resolution: Track 2 Fidelity & Efficiency](https://codalab.lisn.upsaclay.fr/competitions/17266) .

✅ **Mar 22, 2024:** This repository contains official pytorch implementation of "PSW++: Improved Position-Sensitive Windowing strategy for Light Field Image Super-Resolution" in **? solutions 👑** in [NTIRE2024 Light-Field Super Resolution: Track 1 Fidelity Only](https://codalab.lisn.upsaclay.fr/competitions/17265) .

## Code

### Dependencies

It is recommended to use a **Python 3.8** or above version.

```
pip install opencv-python numpy scikit-image h5py imageio mat73 scipy
pip install torch torchvision einops
```

### Prepare Dataset

```bash
# default use patch size 32 * 4 and stride 32
python Generate_Data_for_Training.py --angRes 5 --scale_factor 4
python Generate_Data_for_Test.py --angRes 5 --scale_factor 4
python Generate_Data_for_inference.py --angRes 5 --scale_factor 4
```

### 🔥 FOR NTIRE2024 LFSR Challenge Track 2: Fidelity & Efficiency

**🌟 Training Model**
```bash
CUDA_VISIBLE_DEVICES=0,1,2 python lfsr.py \
    --name Exp329.LFSR.S32.LF-DET-0312.EMA0.999.B3.LR2e-4.E90.S3 \
    --multiprocess \
    --device cuda:0 \
    --task train \
    --scale 4 \
    --patch-size 32 \
    --stride 16 \
    --processor vanilla \
    --angular 5 \
    --model LF_DET_0312 \
    --dataset LFSR.ALL \
    --log 10 \
    --log-val 1 \
    --log-save 1 \
    --train-lr-scheduler 3 \
    --train-batchsize 3 \
    --train-epoch 90 \
    --train-lr 2e-4 \
    --ema-decay 0.999
```

**🌟 Validation Model**
Expected Result should be [**mean_psnr: 33.1377, mean_ssim: 0.9494**]

```bash
python lfsr.py \
    --name Exp329.LFSR.S32.LF-DET-0312-SSAAA-SEQ.EMA0.999.B3.LR2e-4.E90.S3.VAL \
    --device cuda:0 \
    --task val_all \
    --scale 4 \
    --patch-size 32 \
    --stride 16 \
    --processor psw++ \
    --angular 5 \
    --model LF_DET_0312 \
    --dataset LFSR.ALL \
    --model-source vanilla \
    --model-path checkpoints/Exp329.LFSR.S32.LF-DET-0312.EMA0.999.B3.LR2e-4.E90.S3.FT-ALL2.E2.S46172.pth
```

**🌟 Submission for NTIRE24:VALIDATION Stage**
```bash
python lfsr.py \
    --name Exp329.LFSR.S32.LF-DET-0312-SSAAA-SEQ.EMA0.999.B3.LR2e-4.E90.S3.NTIRE.VAL \
    --device cuda:0 \
    --task test \
    --scale 4 \
    --patch-size 32 \
    --stride 16 \
    --processor psw++ \
    --angular 5 \
    --model LF_DET_0312 \
    --dataset LFSR.NTIRE.VAL \
    --model-source vanilla \
    --model-path checkpoints/Exp329.LFSR.S32.LF-DET-0312.EMA0.999.B3.LR2e-4.E90.S3.FT-ALL2.E2.S46172.pth
```

**🌟 Submission for NTIRE24:TEST Stage**
```bash
python lfsr.py \
    --name Exp329.LFSR.S32.LF-DET-0312-SSAAA-SEQ.EMA0.999.B3.LR2e-4.E90.S3.NTIRE.TEST \
    --device cuda:0 \
    --task test \
    --scale 4 \
    --patch-size 32 \
    --stride 16 \
    --processor psw++ \
    --angular 5 \
    --model LF_DET_0312 \
    --dataset LFSR.NTIRE.TEST \
    --model-source vanilla \
    --model-path checkpoints/Exp329.LFSR.S32.LF-DET-0312.EMA0.999.B3.LR2e-4.E90.S3.FT-ALL2.E2.S46172.pth
```

### 🔥 FOR NTIRE2024 LFSR Challenge Track 1: Fidelity Only

**🌟 Training Model**
```bash
CUDA_VISIBLE_DEVICES=0,1,2 python lfsr.py \
    --name Exp329.LFSR.S32.LF-DET-0310.EMA0.999.B3.LR2e-4.E90.S3 \
    --multiprocess \
    --device cuda:0 \
    --task train \
    --scale 4 \
    --patch-size 32 \
    --stride 16 \
    --processor vanilla \
    --angular 5 \
    --model LF_DET_0310 \
    --dataset LFSR.ALL \
    --log 10 \
    --log-val 1 \
    --log-save 1 \
    --train-lr-scheduler 3 \
    --train-batchsize 3 \
    --train-epoch 90 \
    --train-lr 2e-4 \
    --ema-decay 0.999
```

**🌟 Validation Model**
Expected Result should be [**mean_psnr: 33.1377, mean_ssim: 0.9494**]

```bash
python lfsr.py \
    --name Exp329.LFSR.S32.LF-DET-0310-SSAAA-SEQ.EMA0.999.B3.LR2e-4.E90.S3.VAL \
    --device cuda:0 \
    --task val_all \
    --scale 4 \
    --patch-size 32 \
    --stride 16 \
    --processor psw++ \
    --angular 5 \
    --model LF_DET_0310 \
    --dataset LFSR.ALL \
    --model-source vanilla \
    --model-path checkpoints/Exp329.LFSR.S32.LF-DET-0310.EMA0.999.B3.LR2e-4.E90.S3.FT-ALL2.E2.S46172.pth
```

**🌟 Submission for NTIRE24:VALIDATION Stage**
```bash
python lfsr.py \
    --name Exp329.LFSR.S32.LF-DET-0310-SSAAA-SEQ.EMA0.999.B3.LR2e-4.E90.S3.NTIRE.VAL \
    --device cuda:0 \
    --task test \
    --scale 4 \
    --patch-size 32 \
    --stride 16 \
    --processor psw++ \
    --angular 5 \
    --model LF_DET_0310 \
    --dataset LFSR.NTIRE.VAL \
    --model-source vanilla \
    --model-path checkpoints/Exp329.LFSR.S32.LF-DET-0310.EMA0.999.B3.LR2e-4.E90.S3.FT-ALL2.E2.S46172.pth
```

**🌟 Submission for NTIRE24:TEST Stage**
```bash
python lfsr.py \
    --name Exp329.LFSR.S32.LF-DET-0310-SSAAA-SEQ.EMA0.999.B3.LR2e-4.E90.S3.NTIRE.TEST \
    --device cuda:0 \
    --task test \
    --scale 4 \
    --patch-size 32 \
    --stride 16 \
    --processor psw++ \
    --angular 5 \
    --model LF_DET_0310 \
    --dataset LFSR.NTIRE.TEST \
    --model-source vanilla \
    --model-path checkpoints/Exp329.LFSR.S32.LF-DET-0310.EMA0.999.B3.LR2e-4.E90.S3.FT-ALL2.E2.S46172.pth
```

## Citation

If you find this work helpful, please consider citing the following papers:

```bibtex
@InProceedings{BasicLFSR,
  author    = {Wang, Yingqian and Wang, Longguang and Liang, Zhengyu and Yang, Jungang and Timofte, Radu and Guo, Yulan and Jin, Kai and Wei, Zeqiang and Yang, Angulia and Guo, Sha and Gao, Mingzhi and Zhou, Xiuzhuang and Duong, Vinh Van and Huu, Thuc Nguyen and Yim, Jonghoon and Jeon, Byeungwoo and Liu, Yutong and Cheng, Zhen and Xiao, Zeyu and Xu, Ruikang and Xiong, Zhiwei and Liu, Gaosheng and Jin, Manchang and Yue, Huanjing and Yang, Jingyu and Gao, Chen and Zhang, Shuo and Chang, Song and Lin, Youfang and Chao, Wentao and Wang, Xuechun and Wang, Guanghui and Duan, Fuqing and Xia, Wang and Wang, Yan and Xia, Peiqi and Wang, Shunzhou and Lu, Yao and Cong, Ruixuan and Sheng, Hao and Yang, Da and Chen, Rongshan and Wang, Sizhe and Cui, Zhenglong and Chen, Yilei and Lu, Yongjie and Cai, Dongjun and An, Ping and Salem, Ahmed and Ibrahem, Hatem and Yagoub, Bilel and Kang, Hyun-Soo and Zeng, Zekai and Wu, Heng},
  title     = {NTIRE 2023 Challenge on Light Field Image Super-Resolution: Dataset, Methods and Results},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year      = {2023},
}
```

## Contact

Welcome to send email to jinkai@bigo.sg, if you have any questions about this repository or other issues.
