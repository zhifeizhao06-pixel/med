# SegResNetVAE → MedSegDiff

这是一个不修改原 baseline 的两阶段 BraTS-WT（二值全肿瘤）研究实现：

1. 三维 `SegResNetVAE` 从四模态 MRI 生成粗分割，并通过 VAE 分支约束潜在表示。
2. 将三维粗分割和不确定性缓存为体数据。
3. 二维 MedSegDiff 接收 `T1/T1ce/T2/FLAIR + coarse + uncertainty + noisy mask`，细化轴向切片。

第一版的目的，是先验证两阶段设计能否稳定超过 Stage 1 和原 MedSegDiff。当前实现不替换原目录中的任何文件。

## 数据结构

支持常见 BraTS 病例目录：

```text
BraTS/
  BraTS20_Training_001/
    BraTS20_Training_001_t1.nii.gz
    BraTS20_Training_001_t1ce.nii.gz
    BraTS20_Training_001_t2.nii.gz
    BraTS20_Training_001_flair.nii.gz
    BraTS20_Training_001_seg.nii.gz
```

所有命令都从仓库根目录执行。数据按患者划分，禁止按切片随机划分。

## 安装

建议单独建立 Python 3.10/3.11 环境，并先按服务器 CUDA 版本安装 PyTorch：

```bash
python -m pip install -r requirement.txt
python -m pip install -r world_model_pipeline/requirements.txt
```

## 1. 创建患者级划分

```bash
python -m world_model_pipeline.prepare_manifest \
  --data_dir /data/BraTS2020/TrainingData \
  --output world_model_pipeline/brats_manifest.json
```

默认比例为 train/val/test = 70/10/20。同一份 manifest 必须贯穿两个阶段。

## 2. 训练 Stage 1

```bash
python -m world_model_pipeline.train_stage1 \
  --manifest world_model_pipeline/brats_manifest.json \
  --output results/world_model/stage1_best.pt \
  --roi_size 96 96 96 \
  --batch_size 1 \
  --epochs 300
```

显存不足时先将 ROI 改成 `64 96 96`。Stage 1 checkpoint 根据验证集粗分割 Dice 选择，而不是根据重建损失选择。

## 3. 导出 Stage 1 条件

```bash
python -m world_model_pipeline.export_stage1_conditions \
  --manifest world_model_pipeline/brats_manifest.json \
  --checkpoint results/world_model/stage1_best.pt \
  --output_dir results/world_model/conditions
```

每位患者生成一个压缩 NPZ：

```text
image       [4,H,W,D]
mask        [1,H,W,D]
coarse      [1,H,W,D]
uncertainty [1,H,W,D]
```

当前 uncertainty 是粗分割 Bernoulli entropy。后续若要研究 VAE posterior uncertainty，需要让 latent 显式进入 segmentation decoder，并对 latent 重复采样。

## 4. 训练 Stage 2

```bash
python -m world_model_pipeline.train_stage2 \
  --cache_dir results/world_model/conditions \
  --output results/world_model/stage2_best.pt \
  --image_size 128 \
  --num_channels 64 \
  --batch_size 4 \
  --epochs 100
```

训练时会随机腐蚀或膨胀 coarse mask，降低 Diffusion 直接复制第一阶段结果的风险。正式实验建议增加 out-of-fold Stage 1 预测。

## 5. 采样和评估

先用较少步数检查流程；正式对比再统一采样步数：

```bash
python -m world_model_pipeline.sample_stage2 \
  --cache_dir results/world_model/conditions \
  --checkpoint results/world_model/stage2_best.pt \
  --output_dir results/world_model/predictions \
  --split test \
  --num_ensemble 5 \
  --sampling_steps 1000

python -m world_model_pipeline.evaluate_stage2 \
  --cache_dir results/world_model/conditions \
  --prediction_dir results/world_model/predictions \
  --split test
```

## 公平实验最低要求

- 使用完全相同的患者划分比较 Stage 1、MedSegDiff baseline 和完整模型。
- 报告 Stage 1 coarse Dice，证明 Stage 2 的实际增益。
- Stage 2 只能使用 Stage 1 预测，不能使用 GT mask 作为 coarse condition。
- 最终实验加入 HD95/ASSD；当前评估脚本只提供体级 Dice/IoU smoke evaluation。
- 当前为 WT 二值版本。升级 WT/TC/ET 时需要把 Stage 1、扩散 mask 和损失统一改为三通道。

## 已知边界

- Stage 1 是三维，Stage 2 是二维，因此最终结果还没有显式的跨切片一致性约束。
- 原 MedSegDiff V2 内部仍有 highway/calibration 分支；在这里它接收图像、粗分割和不确定性。后续消融应比较“保留 highway”与“移除 highway”。
- 原仓库的扩散采样较慢，`sampling_steps` 必须小于或等于训练时的 diffusion steps。
- NPZ 缓存侧重可复现和解耦，不是最高吞吐实现；大规模实验可以换成 Zarr/HDF5。

