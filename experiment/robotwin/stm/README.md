# STM 训练源码快照与启动脚本

下载原件：`/root/autodl-tmp/projects/icm`，HF `zhuojianli/icm` revision `6cae6fc54d004c34b573cdb486c1beb9762f1c2c`。377 个文件全部校验，原件不修改。`src/` 是实际训练所用的独立源码快照（337 个文件），保留上游 LICENSE。它不替换仓库主目录的 PICA/Z4 实现。`source_snapshot_sha256.json` 可验证源码与正在运行的副本逐文件一致；`local_cuda_training.patch` 记录相对 HF 原件的 CUDA/保存逻辑修改。

启动或恢复（后台运行需自行用 nohup；输出锁防止重复启动）：

```bash
bash experiment/robotwin/train_stm_from20k.sh
```

入口在现有训练机优先使用 `/root/autodl-tmp/projects/lingbot-vla-stm`，以沿用已有运行和输出锁；新 checkout 默认使用本目录的完整快照。显式使用仓库内快照：

```bash
STM_WORK_DIR="$PWD/experiment/robotwin/stm" bash experiment/robotwin/train_stm_from20k.sh
```

`STM_ASSETS_ROOT` 默认是当前仓库根目录，数据、Qwen processor、辅助教师模型需放在该目录的既有路径；`PYTHON_BIN` 可指定已安装依赖的 Python，`INIT_MODEL` 可覆盖初始模型路径。公开仓库不包含训练数据或模型二进制文件。`CHECK_ONLY=1` 仅生成配置和检查资源路径，不启动训练。

初始化模型：`/root/autodl-tmp/models/hggforget-lingbotvla-robotwin-clean-sft/global_step_20000`。数据为既有官方 `RoboTwin_lerobot_v30`，2500 episodes、15 fps，使用配套 clean normalization。

## 参数及证据范围

- 文档明确的正式规模：global batch 512，4000 步，每 1000 步保存。
- 文档没有提供正式 H20 完整 CLI 配置；其余参数沿用公开 initial_candidate，不能声称完全复现正式实验。
- Muon，基础 LR `1e-4`，cosine 到 `5e-5`，无 warmup，weight decay 0，梯度裁剪 1；momentum 0.95、Nesterov、NS 5、match_rms_adamw。
- 保留候选配置的专家 LR 缩放：32/4 的平方根，专家初始 LR 约 `2.828e-4`。
- 2 卡，每卡 16，梯度累积 16。可设置 `MICRO_BATCH_SIZE`，必须使 `512 / (GPU数 × micro)` 为整数。CUDA FSDP2、混合精度、非重入梯度检查点、不 compile。
- STM 256 维、4 heads，当前帧及前 0.5/1 秒历史；15 fps 实际偏移为 -8/-15 帧，episode 边界屏蔽，未来监督帧不进入记忆。
- 冻结 ViT，历史编码 no_grad，STM 和其他原本可训练参数继续更新。
- 使用公开候选默认颜色增强：亮度 0.7–1.3、对比度 0.6–1.4、饱和度 0.5–1.5，共享当前/历史/未来和视角参数；`LINGBOT_AUG_LEVEL=0`、状态噪声 0。正式实验名字中的 z3a 环境变量没有公开，不猜测替换为本地 Z4。
- 保留 depth/future-depth/future-video 辅助损失和已有教师权重。

## 保存与恢复

现有运行输出：`/root/autodl-tmp/projects/lingbot-vla-stm/outputs/stm-from20k-gbs512`。仓库内快照默认输出到仓库根目录的 `outputs/stm-from20k-gbs512`，也可用 `OUTPUT_DIR` 覆盖。日志 `train.log`；实际配置 `launch/config.yaml`；来源记录 `launch/provenance.json`。

第 3 步提前保存验证；之后每 1000 步或距上次保存约 7200 秒触发保存，在完整 optimizer step 后执行。最后一步也保存。可通过 `SAVE_INTERVAL_SECONDS` 调整时间间隔。

`checkpoints/global_step_N/` 包含 `model`、`optimizer`、每卡 `extra_state` 和可部署的 `hf_ckpt`。DCP 保留训练精度状态，HF 导出沿用原实现 BF16。保存定时由 rank 0 统一广播，避免双卡走入不同分支；限制写盘线程，降低 CPU 内存峰值。

重复运行同一脚本会恢复最新可加载的完整训练 checkpoint；不会将已有目录静默当成新的实验。断电或磁盘故障仍可能中断训练，应保留完整 checkpoint 目录用于续训。

## 验证

提供的 18 项 STM/缓存测试已通过。实际训练、梯度、显存和 checkpoint 验证记录见 `outputs/stm-from20k-gbs512/validation.json`；仓库内保存初次验证副本 `../records/stm_from20k_validation_20261024.json`。

源码测试：

```bash
/root/miniconda3/envs/lingbotvla/bin/python -m unittest discover -s experiment/robotwin/stm/tests -v
```

配置证据：正式 H20 的 global batch 512、4000 步和每 1000 步保存见 HF 文档；其余超参数来自 `training/initial_candidate/train.temporal.yaml`。时间保存为本机按用户要求新增，日志中的 157/284/412/530 等步数由约 2 小时触发，并非固定步数间隔。
