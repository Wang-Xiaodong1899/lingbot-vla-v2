# Z6b 增广：从 PICA1300 权重继续训练

本实现迁入本机 `lingbot-vla` 仓库 revision `746c647713cc2ed0dcfc7581b80f3a0e84b4088f` 的 `Z6b_20k_aug5_500`。增广函数和必要辅助函数来自 `patches/aug5_whitelist.py` / `patches/FULL_vs_be969b8.diff`，训练参数来自 `tools/train_z6b.sh` 与已保存配置。文件 SHA-256 记录在 [z6b_source.json](records/z6b_source.json)。

原实验从 hgg step20000 初始化；本入口改为**显式指定 PICA1300 HF 权重**。这是 PICA → Z6b 的顺序微调，不是原始 Z6b 结果复现：加载学生权重，创建新 Muon 状态，训练步数从 0 开始，关闭 PICA 蒸馏损失，保留基线的 depth/video 辅助目标。不要把 `robotwin_z4_from_pica1300_lr1e-5/global_step_400` 当作 PICA1300 基线。训练不会下载或猜测初始 checkpoint。

## Level 5 图像增广

每个样本以 10% 概率保持图像原样；其余 90% 的样本，各项操作独立触发。**不叠加默认 brightness/contrast/saturation jitter**，否则会破坏 10% 原图分支。几何参数由样本种子和相机键决定，在同一视角的当前/未来视频帧间共享；传感器噪声沿用原实现的全局随机数流。

| 操作 | 条件概率与幅度 |
|---|---|
| 颜色与色温 | 80%；亮度 0.25–1.10、对比度 0.5–1.5、饱和度 0.4–1.6、红蓝色温偏移 ±10% |
| 缩放、平移 | 50%；缩放 0.90–1.10、平移 ≤6%，露出边界填黑 |
| 矩形遮挡 | 95%；面积 2–10%，纯色/马赛克/像素噪声；至多尝试 12 次，仅混合到桌面掩码覆盖区域 |
| 高斯噪声、椒盐噪声 | 分别独立 50% / 15%；sigma 0.01–0.05、密度 0.2–1% |
| 高斯模糊、运动模糊 | 互斥 25% / 10%；sigma 0.5–1.5、3–9 像素线核 |
| JPEG | 30%；质量 35–90；预检要求 OpenCV 可用 |

桌面掩码严格沿用原实现：根据**第一帧**下方区域的颜色中位数估计桌面并扩张非桌面区域。它是颜色启发式，不保证所有真实物体/跨帧移动物体都被正确识别。Z6b 与本仓库 Z4 的前景保护算法、概率关联和幅度不同，不能互换。

额外使用归一化状态高斯噪声 0.01（动作目标不加噪），以及原始 v1 **seen** 指令池。仓库包含压缩池 `assets/training_data/official_instruction_pool_v1.json.gz`，覆盖官方 2500 个 episode，解压后 MD5 为 `d67aa0f314ff7b2ce18834eacf91d7f9`。只在该入口通过 `LINGBOT_INSTRUCTION_POOL` 显式启用，不改写原数据目录；每次启动在输出 `launch/` 中保存池副本和校验记录。

## 准备与启动

沿用主仓库的训练环境、官方 `data/RoboTwin_lerobot_v30`、Qwen3-VL processor、depth/video 教师权重。数据必须为官方 v3.0：2500 episodes、548893 frames、15 fps、相机 240×320。

```bash
# 在本仓库根目录，先填写真实 PICA1300 checkpoint、配套 normalization 和 processor 路径。
export INIT_MODEL=/path/to/PICA/global_step_1300/hf_ckpt
export NORM_FILE=/path/to/robotwin_clean_norm.json
export QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct
export OUTPUT_DIR="$PWD/outputs/robotwin-z6b-from-pica1300-gbs2048"

# 生成配置并检查数据、模型、教师和依赖；不会启动训练。
CHECK_ONLY=1 bash experiment/robotwin/train_z6b_from_pica1300.sh

# 确认资源空闲后正式启动；入口本身不负责排队或后台托管。
bash experiment/robotwin/train_z6b_from_pica1300.sh
```

默认参数与可覆盖项：

| 参数 | 默认 |
|---|---|
| Muon / expert LR scaling | Muon，关闭额外 expert LR scaling |
| GLOBAL_BATCH_SIZE / MICRO_BATCH_SIZE | 2048 / 8；两卡时累积 128 次，四卡时 64 次；要求整除 |
| LR / LR_MIN / LR_DECAY_RATIO | 1e-4 / 5e-5 / 2.0，cosine，无 warmup |
| MAX_STEPS / SAVE_STEPS | 500 / 500；余弦按 1000 步周期，在 500 步停止时 LR 约 7.5e-5 |
| STATE_NOISE_STD / DATA_NUM_WORKERS | 0.01 / 2 |
| ENABLE_FP32 | 1，沿用参考配置；设为 0 会改变计算精度 |

例如使用用户自选较低 LR、global batch 1024、每 200 步保存：

```bash
GLOBAL_BATCH_SIZE=1024 LR=1e-5 LR_MIN=5e-6 SAVE_STEPS=200 \
  bash experiment/robotwin/train_z6b_from_pica1300.sh
```

该示例是自定义 PICA 微调配置，不是原 Z6b 的默认训练参数。当前入口使用本仓库的 FSDP2、梯度检查点、关闭 compile，并保留 optimizer checkpoint 以供恢复；参考运行曾使用 compile 并跳过 optimizer 保存。已有输出目录会校验初始模型、增广配方、指令池和状态噪声，避免把其他实验当作本实验继续训练。

## 验证范围

```bash
python -m unittest discover -s tests -p test_z6b_augmentation.py -v
python -m unittest discover -s tests -p test_prepare_z6b.py -v
```

CPU 对照验证 50 个种子、3 个相机、3 帧 clip，共 150 个案例，在相同全局 RNG 状态下与原增广函数逐元素完全一致。测试覆盖原图分支、前景掩码、遮挡、共享几何、数值范围、指令池、配置生成和恢复隔离。记录见 [z6b_validation_20261024.json](records/z6b_validation_20261024.json)。本次集成没有启动正式训练，未验证 PICA1300 上的收敛或成功率。
