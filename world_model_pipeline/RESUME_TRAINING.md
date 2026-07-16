# 断点续训

两个阶段都会在每个完整 epoch 结束后保存滚动 checkpoint：

- `stage1_last.pt` / `stage2_last.pt`：最近完成的 epoch，推荐用于恢复训练。
- `stage1_best.pt` / `stage2_best.pt`：验证集指标最好的 epoch，用于导出、采样和最终评估。

checkpoint 包含模型、优化器、已完成 epoch、模型配置和历史最佳指标。写入使用临时文件替换，训练在保存期间意外中断时不会破坏上一个有效 checkpoint。

## Stage 1

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u -m world_model_pipeline.train_stage1 \
  --manifest world_model_pipeline/brats_manifest.json \
  --output results/world_model/stage1_best.pt \
  --resume results/world_model/stage1_last.pt \
  --roi_size 96 96 96 \
  --batch_size 1 \
  --epochs 300 \
  --lr 1e-4 \
  --vae_weight 0.1 \
  --device cuda
```

## Stage 2

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u -m world_model_pipeline.train_stage2 \
  --cache_dir results/world_model/conditions \
  --output results/world_model/stage2_best.pt \
  --resume results/world_model/stage2_last.pt \
  --image_size 128 \
  --num_channels 128 \
  --num_res_blocks 2 \
  --diffusion_steps 1000 \
  --batch_size 2 \
  --epochs 100 \
  --lr 5e-5 \
  --num_workers 8 \
  --device cuda
```

`--epochs` 是目标总轮数。例如 checkpoint 已完成第 17 轮，指定 `--epochs 100` 会继续训练第 18–100 轮。

如果只有旧版 `stage1_best.pt` 或 `stage2_best.pt`，也可以先把它传给 `--resume`。脚本会恢复模型和 epoch，但由于旧文件没有优化器状态，会显示 `model only (fresh optimizer)`；完成下一个 epoch 后生成的新 `*_last.pt` 就能完整恢复。

默认滚动 checkpoint 根据 `--output` 自动命名。也可以用 `--last_output` 指定其他位置。恢复以完整 epoch 为粒度；如果在一个 epoch 中间中断，会从上一个已经完成并保存的 epoch 继续。
