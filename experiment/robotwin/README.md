# Generate Lerobot Dataset from RoboTwin Data

## PICA action alignment on the V2 clean baseline

`train_pica_from20k.sh` uses
`/root/autodl-tmp/models/hggforget-lingbotvla-robotwin-clean-sft/global_step_20000`
as pretrained weights. It starts a **new** optimizer/schedule at step 0 and reads
only the existing 50-task clean manifest. Subsequent launches resume complete
PICA checkpoints in the new output directory. It does not overwrite the baseline.
The recipe is in `configs/vla/robotwin/robotwin_pica_from20k.yaml`.

```bash
# Validate baseline, clean datasets, normalization and auxiliary teacher assets.
CHECK_ONLY=1 bash experiment/robotwin/train_pica_from20k.sh

# On a GPU host with those assets installed, first run a short smoke test.
OUTPUT_DIR="$PWD/outputs/robotwin-pica-smoke" MAX_STEPS=2 \
  bash experiment/robotwin/train_pica_from20k.sh

# Default: four GPUs, microbatch 1, accumulation 4, global batch 16.
bash experiment/robotwin/train_pica_from20k.sh

# Example batch override; global batch must divide GPUs × microbatch exactly.
CUDA_VISIBLE_DEVICES=0,1,2,3 MICRO_BATCH_SIZE=2 GLOBAL_BATCH_SIZE=32 \
  bash experiment/robotwin/train_pica_from20k.sh
```

The supplied training note describes LingBot-VA. This implementation adapts its
appearance and layer alignment to **LingBot-VLA-V2**, whose 36-layer action expert
has hidden width 768. The student receives augmented images; the FP32 EMA teacher
receives the original images and exactly the same action noise and timestep.
Student block output 8 is projected through `Linear(768,512) → GELU → Linear(512,768)`
and aligned to teacher block output 20 using tokenwise
negative cosine similarity. State tokens and padded action tokens are excluded.
The teacher stops after block 20 and has no gradients. EMA decay is 0.9999 and is
applied once after each optimizer step, including when accumulating gradients.

Total loss retains the baseline action flow loss, current/future depth, current/
future visual feature losses and MoE regularizers, and adds `0.05 × PICA loss`.
Auxiliary visual/depth targets are computed from the **augmented** current/future
images. This V2 model has no future-video flow-matching noise variable; there is
no artificial video noise gap and no reduction of teacher action noise. Batch 16
means **16 observation/action-chunk samples**, not the paper's 16 full episodes.
The horizon stays 50. The default 30,000 updates, layer indices, projector size,
alignment weight and EMA decay follow the note; the `1e-5 → 5e-6` learning-rate
schedule is a conservative fine-tuning choice, not a reported paper setting.
The existing Muon/AdamW optimizer grouping is retained, including for the projector.

Augmentation is sampled independently for each camera, sharing parameters and
the same Gaussian noise field between that camera's current and future frames.
Mixture probabilities are 1/4 photometric, 1/4 noise, 1/2 both. Photometric order
is brightness, contrast, saturation, hue, gamma, exposure (an implementation
choice). Ranges follow the training note; noise std is 0.015 in [0,1]. With
probability 0.25, crop side length is 0.92–1.0 followed by rotation within ±8°.
No blur/fog group is enabled. Legacy image/state augmentation is disabled in
this launcher. Evaluation with `policy_eval=True` never uses PICA augmentation.

DCP checkpoints contain student, projector, EMA teacher, optimizer, scheduler,
dataloader and CPU/CUDA RNG states. Resume requires the same PICA configuration
and strict checkpoint loading. Data-worker/prefetch randomness retains the
existing dataloader's resume semantics; bitwise-identical augmented batches
across restarts are not promised. `hf_ckpt` contains the student policy with the
projector removed; no teacher is needed for inference. FSDP2 shards FP32 teacher
master weights; gathered forward weights use BF16. This adds a teacher model to
training memory usage, so establish a working microbatch on the target hardware.
The recipe currently requires FSDP2, EP=1, CUDA initialization, mixed precision,
no CPU offload/compile and non-reentrant gradient checkpointing.

TensorBoard reports `pica/alignment_loss` and `pica/alignment_loss_weighted`
(last microbatch on rank 0, matching the existing auxiliary metric reporting).
Negative alignment loss is expected. Compare the student `hf_ckpt` against the
20k baseline on **both clean and randomized evaluations with identical task,
seed, instruction and inference settings**. Use held-out trajectories for action
prediction error; rollout success measures closed-loop behavior. Training-loss
reduction alone does not establish either improvement. If testing an
augmentation-only ablation, retain the same paired augmentation and omit the
alignment term in a separate experiment; disabling `pica_params` disables both
the paired augmentation and alignment.

CPU regression tests (small substitute blocks, actual production forward methods):

```bash
PYTHONNOUSERSITE=1 /root/miniconda3/envs/lingbotvla/bin/python \
  -m unittest discover -s tests -p test_pica.py -v
```

These cover temporal/camera augmentation, shared noisy actions, layer capture,
early teacher exit, padding, stop-gradient, projector gradients, FP32 EMA,
transactional DCP save/restore and student-only HF export. CUDA kernels and
multi-GPU FSDP execution require the short GPU smoke run above.

This guide explains how to process raw data from **RoboTwin** and convert it into the **LerobotDataset** format following the official RoboTwin instructions.

## 1. Clone the Official RoboTwin Repository
```bash
git clone git@github.com:RoboTwin-Platform/RoboTwin.git
cd RoboTwin
git checkout 13c3c47ff4312dd62484bcd51be034af55c062d1
```

## 2. Create Required Directories
Navigate to the `policy/pi0` directory inside the cloned RoboTwin repository and create the folders:

```bash
cd ./policy/pi0
mkdir processed_data training_data
```

## 3. Convert RoboTwin Raw Data to HDF5

Download [official dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset) and unzip the dataset to '/path/to/RoboTwin/data'

**Example:**
```bash
data
└── adjust_bottle
    └── aloha-agilex_clean_50
```

Use the provided script [process_data_pi0.sh](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/policy/pi0/process_data_pi0.sh):

```bash
cd policy/pi0
bash process_data_pi0.sh ${task_name} ${task_config} ${expert_data_num}
```

**Example (clean demo):**
```bash
bash process_data_pi0.sh adjust_bottle aloha-agilex_clean_50 50
```

**Example (randomized demo):**
```bash
bash process_data_pi0.sh adjust_bottle aloha-agilex_randomized_500 50
```

If successful, the output folder:
```
processed_data/${task_name}-${task_config}-${expert_data_num}/
```

## 4. Prepare Training Data

Copy the required processed datasets into `training_data/${model_name}`:

```bash
cp -r processed_data/${task_name}-${task_config}-${expert_data_num} \
      training_data/${model_name}/
```

## 5. Ensure Sufficient Disk Space

The generated **LerobotDataset** will be stored under:

```
$XDG_CACHE_HOME/huggingface/lerobot/${repo_id}
```

By default, `XDG_CACHE_HOME` points to `~/.cache`, which must have sufficient free space.  
If space is low, change the cache location:

```bash
export XDG_CACHE_HOME=/path/to/your/cache
```

## 6. Generate LerobotDataset v2.1 Format

Run [generate.sh ](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/policy/pi0/generate.sh) to convert the HDF5 datasets to Lerobot.

Parameters:
- **hdf5_path**: Path to the HDF5 training data (e.g., `./training_data/${model_name}/`)
- **repo_id**: Name for the dataset (e.g., `my_repo`)

```bash
bash generate.sh ${hdf5_path} ${repo_id}
```

**Example:**
```bash
bash generate.sh ./training_data/demo_clean/ demo_clean_repo
```

Output:
```
${XDG_CACHE_HOME}/huggingface/lerobot/${repo_id}
```

---

## 7. Simulated Evaluation (Inference + RoboTwin Sim)

After training, evaluate your checkpoint on the 50 RoboTwin tasks (100 episodes each) with
[`start_robotwin_infer_and_eval.sh`](./start_robotwin_infer_and_eval.sh). It starts
`num_gpus * num_per_gpu` resident inference servers (one port each), then runs the sim tasks
in a **queue-scheduled** fashion against free slots — finishing a task frees its slot and
starts the next. All flags below also accept the equivalent environment variable
(`MODEL_PATH`, `EVAL_WORKDIR`, `OUTPUT_BASE`, `CONDA_SH`, `QWEN3VL_PATH`, `INFERENCE_ENV`, `SIM_ENV`).

### Prerequisites

1. **RoboTwin repo** cloned and its dependencies installed (steps 1–2 above). The launcher
   needs the repo **root** path (the one containing `envs/`, `assets/`, `task_config/`, `script/`).
2. **Two conda environments**:
   - inference side — e.g. `lingbotvla` (PyTorch + this repo's model code).
   - sim side — `RoboTwin` (sapien / mplib / curobo / open3d …), built against numpy 1.26.x.
3. **Qwen3-VL backbone** checkpoint used by the VLA vision-language encoder (`QWEN3VL_PATH`).
4. **Your trained HF checkpoint** (`model_path`), e.g. `.../global_step_xxxxx/hf_ckpt`.

> [!IMPORTANT]
> Release validation of the published checkpoint uses **FP32 inference**. The lower-memory
> BF16 mode is useful for pipeline checks, but can produce materially different success
> rates.

> The launcher **auto-copies** the eval client (`eval_policy_client_lingbotvla.py` + the small
> `deploy/` helpers) from this repo into `<RoboTwin>/script/`, and **self-heals** the curobo
> embodiment `.yml` and editable-install `.pth` paths to point at the RoboTwin checkout you
> pass. No manual path setup is needed when relocating RoboTwin.

### Run the full benchmark (50 tasks)

> Substitute every `/path/to/...` and the conda env / `conda.sh` path for your machine.

```bash
# from this repo's root (so inference_workdir = current working dir)
QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct \
bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
    --model_path     /path/to/your/checkpoint/hf_ckpt \
    --eval_workdir   /path/to/RoboTwin \
    --output_base    /path/to/VLABenchmarkResult \
    --conda_sh       /path/to/miniconda3/etc/profile.d/conda.sh \
    --inference_env  lingbotvla \
    --sim_env        RoboTwin \
    --task_config    demo_clean \
    --num_tasks 50 --num_gpus 4 --num_per_gpu 1
```

Use `--task_config demo_randomized` for the randomized benchmark.

**GPU / concurrency**
- `num_gpus` × `num_per_gpu` = number of concurrent sim slots (one inference server per slot).
- `--num_per_gpu 1` is the safe starting point. In our current software stack, one FP32
  policy server plus its simulator uses roughly 32 GB, so leave additional headroom.
- A 24 GB GPU generally requires BF16 (`--use_bf16 True --use_fp32 False`), which does not reproduce the published FP32 benchmark. Increasing `num_per_gpu` can OOM; the script retries each task up to 3 times, but persistent OOM skips the task.

### Smoke test (1 task, 1 GPU)

Verify the pipeline end-to-end without waiting for the full run:

```bash
QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct \
bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
    --model_path   /path/to/your/checkpoint/hf_ckpt \
    --eval_workdir /path/to/RoboTwin \
    --conda_sh     /path/to/miniconda3/etc/profile.d/conda.sh \
    --task_config  demo_clean \
    --num_tasks 1 --num_gpus 1 --num_per_gpu 1
```

The run dir is printed at startup (`Run directory: ...`). You should see `Success rate: N/N =>
...` lines appear in the task log. Each task evaluates **100 episodes**, so even a
single-task smoke takes tens of minutes — kill it once you've seen successes, then launch
the full run.

### Output layout

```
<output_base>/<exp>_<step>k_<task_config>_<timestamp>/
├── stats.txt                 # final per-task table + overall success rate
├── inference_logs/           # one log per inference server / port
├── eval_logs/                # one log per task (per-step progress, success rate)
└── eval_results/             # per-task videos: episodeN_success.mp4 ...
```

### Useful flags

| flag | meaning |
|------|---------|
| `--no_video` | disable per-episode video recording (faster, no videos saved) |
| `--keep_inference` | leave inference servers resident after sim finishes |
| `--start_port` | base port for inference servers (default 9330, slot *i* uses base + i) |
| `--use_length` | action-chunk length forwarded to the policy (default 50) |
| `--robo_name` | robot config name (default `robotwin`) |
| `--task_config` | RoboTwin setting: `demo_clean` or `demo_randomized` |
| `--num_episodes` | episodes per task (default 100); use 10 for a quick 50-task evaluation |
| `--resume_from` | previous run directory; reuse tasks with all requested episodes and a final result, rerun partial tasks |
| `--use_bf16` / `--use_fp32` | inference precision; release reproduction uses `False` / `True` |
| `--use_compile` | enable lazy `torch.compile` (default `True`; first request takes longer) |
| `--inference_script` | inference-side module (default `deploy/lingbot_vla_v2_policy.py`) |

### Monitor / stop

```bash
RUN=/path/to/VLABenchmarkResult/<exp>_<step>k_<timestamp>

# overall progress (done/skip/fail summary)
sed 's/\x1b\[[0-9;]*m//g' $RUN/eval_logs/*.log | grep -E "Success rate" | tail

# per-task latest success rate
for f in $RUN/eval_logs/*.log; do
  printf "%-24s %s\n" "$(basename $f .log)" \
    "$(grep -aoE 'Success rate: [0-9]+/[0-9]+ => [0-9.]+%' "$f" | tail -1)"
done
```

Stop everything with `Ctrl-C` (the launcher traps it and kills all child processes), or kill
the launcher plus any lingering `deploy.lingbot_vla_v2_policy` / `eval_policy_client_lingbotvla.py`
processes.

### 官方 LeRobot v3 + aug1，从 hgg step 20000 继续训练

本机启动脚本：`experiment/robotwin/train_z3a_official_from20k.sh`。也可从实验仓库
`/root/autodl-tmp/projects/lingbot-vla/tools/train_z3a_official_from20k.sh` 启动。
注意：实验仓库缓存版本为 `lingbot-experiments@ffbf35b`，没有
`Z3a-official-data-aug1` 配置；2026-10-22 尝试 pull 时 GitHub 要求认证。
这里是根据已能核对的官方数据流程、原生 Muon 和 level 1 增广实现的本地适配，
尚未确认与上游 Z3a 完全一致。直接使用发布数据内的指令，没有应用实验仓库的
`instruction_pool.json` seen 指令替换流程。

```bash
cd /root/autodl-tmp/projects/lingbot-vla-v2
# 生成配置、打印命令，不检查依赖或启动训练
DRY_RUN=1 bash experiment/robotwin/train_z3a_official_from20k.sh
# 验证数据、权重和依赖，不启动训练
CHECK_ONLY=1 bash experiment/robotwin/train_z3a_official_from20k.sh
# 正式训练；新增 1000 个优化器更新，每 100 步保存
MAX_STEPS=1000 SAVE_STEPS=100 bash experiment/robotwin/train_z3a_official_from20k.sh
```

- 初始权重：`/root/autodl-tmp/models/hggforget-lingbotvla-robotwin-clean-sft/global_step_20000`，
  沿用同目录上一级的 `robotwin_clean_norm.json` 和机器人关节映射。
- 数据：`data/downloads/RoboTwin_lerobot_v30.zip.part`，也接受下载完成后去掉 `.part` 的文件。
  通过固定 SHA-256 校验后解压到 `data/RoboTwin_lerobot_v30`；已校验的完整解压目录可直接复用。
  v3.0 数据共 2500 episodes、548893 frames。未下载完成或尚未解压完成会中止启动。
- `optimizer=muon`，保留模型原生矩阵参数 Muon / 非矩阵参数 AdamW 的实现；
  关闭额外 MoE 专家学习率倍率。默认 LR `1e-4` → `5e-5`、cosine、无 warmup。
- `LINGBOT_AUG_LEVEL=1`，`STATE_NOISE_STD=0.01`；同视角当前/未来帧共用几何增广参数。
  保留深度/DINO 辅助损失，PICA 关闭。
- global batch 固定 1024。默认每卡 micro batch 为 64，按实际可见 GPU 数计算累积：
  本机两张 A800 为 `2 × 64 × 8 = 1024`。可用 `CUDA_VISIBLE_DEVICES` 和
  `MICRO_BATCH_SIZE` 调整；不能整除时拒绝启动。启用 FSDP 全分片和梯度检查点。
- `MAX_STEPS=1000`、`SAVE_STEPS=100` 是本地默认值，可修改；这里的步数是优化器更新次数。
  原始 20k 目录只有模型权重，没有优化器状态，第一次运行会新建 Muon 状态，从本轮 step 0 开始。
  后续同一个 `OUTPUT_DIR` 支持训练器的完整检查点恢复；新实验应使用新的输出目录。
- 默认输出 `outputs/robotwin-z3a-local-from20k-gbs1024`，包含 `train.log`、
  `launch/config.yaml`、`launch/recipe.json`、训练数据清单和检查点。
  使用 `OUTPUT_DIR`、`INIT_MODEL`、`DATA_ROOT`、`DATA_ARCHIVE`、`NORM_FILE`、`LR`、`LR_MIN`、
  `DATA_NUM_WORKERS`、`PYTHON_BIN` 可覆盖本机路径或参数。

2026-10-22 检查通过了原生训练器配置解析、`2 × 64 × 8 = 1024` 的 batch 计算、现有增广单元测试，
以及真实数据的单样本读取和增广。初始 20k 权重的 6 个分片全部通过 SHA-256 校验。
已补齐 `weights/depth/moge2-vitb-normal.pt`、`weights/depth/model.pt`、
`weights/dino_video/teacher_step_10000.pth`，三者均通过官方 SHA-256 校验和实际模型加载；
DINO 配置通过官方 Git blob 哈希校验。已安装 MoGe 指定的 `utils3d` 提交
`3fab839f0be9931dac7c8488eb0e1600c236e183`，完整 `CHECK_ONLY=1` 预检查通过。
未启动 GPU 训练，尚未验证每卡 batch 64 的显存峰值。如需降低每卡 batch，可设置
`MICRO_BATCH_SIZE=32`，脚本自动将双卡累积改为 16，global batch 仍为 1024。

校验记录位于 `weights/z3a_assets_verified.json` 和 `weights/z3a_initial_model_verified.json`。
后续需要重新下载/验证辅助权重时可运行（支持 `.part` 断点续传，固定上游 revision）：

```bash
/root/miniconda3/envs/lingbotvla/bin/python tools/download_z3a_assets.py
```

本次准备和验证记录已归档到
[`records/z3a_official_from20k_20261022/`](records/z3a_official_from20k_20261022/)：
包含启动预检查、增广测试、三个辅助模型的加载和真实数据读取日志，以及数据/权重校验清单、
训练配置和机器人映射快照。`*.snapshot.yaml`、`train.snapshot.txt` 记录的是本机实际路径，
运行训练时由启动脚本重新生成配置。该归档记录训练启动前的检查，不包含后续训练 loss 日志。

### Z4：10% 原图、90% 在线随机组合增广

Z4 在训练 DataLoader 中、模型图像处理器之前执行。配置见
[`../../configs/augmentation/z4.yaml`](../../configs/augmentation/z4.yaml)，实现见
[`../../lingbotvla/data/vla_data/z4_augmentation.py`](../../lingbotvla/data/vla_data/z4_augmentation.py)。
按一个观测样本采样一次分支，三路相机和当前/未来帧共同使用该分支：
10% 概率直接保留输入图像，90% 进入组合增广。比例为长期采样概率，不强制每个 batch 精确配额；
正常的图像 resize/tokenize 仍然执行。Z4 不额外复制数据集或增大 batch，默认关闭状态噪声。

以下概率均以进入 90% 分支为条件，每个视角分别采样，随后按表格顺序组合：

| 操作 | 分支内概率 | 参数与补充默认值 |
|---|---:|---|
| 色彩/色温 | 80% | 亮度乘数 0.25–1.10、对比度 0.5–1.5、饱和度 0.4–1.6；色温用红/蓝通道反向 ±6% 增益，绿色不变 |
| 全图缩放/平移 | 50% | 缩放 0.9–1.1；水平/垂直位移各为宽/高的 ±6%；不旋转；露出部分补黑，后续噪声/模糊/JPEG 后仍保留黑边 |
| 随机矩形遮挡 | 95% 尝试 | 1 个矩形，目标面积 1%–6%，宽高比 0.4–2.5（对数均匀）；纯色、原区域马赛克、像素噪声各 1/3；马赛克块尺度约 8 像素 |
| 高斯 / 椒盐噪声 | 50% / 15% | 二者互斥，35% 不加噪声；高斯 σ=0.005–0.02（图像范围 0–1）；椒盐污染 0.1%–1% 像素，盐/椒各半 |
| 高斯 / 运动模糊 | 25% / 10% | 二者互斥，65% 不模糊；高斯 σ=0.3–1.0 像素；运动模糊核长 3/5/7 像素等概率，方向 0–180° |
| JPEG 压缩 | 30% | 质量整数 35–90，均匀采样，RGB 编解码，色度采样 4:2:0 |

为同时保持所要求的边际概率和非空操作组合，色彩/遮挡共享一次均匀随机数 `u`：
`u < 0.8` 开色彩，`u >= 0.05` 请求遮挡，所以色彩概率仍是 80%、遮挡仍是 95%，
并且至少请求一个操作。其他操作族独立采样。对应所有样本的概率分别为：色彩 72%、
缩放/平移 45%、遮挡尝试 85.5%、高斯噪声 45%、椒盐 13.5%、高斯模糊 22.5%、
运动模糊 9%、JPEG 27%。如果仅选中的遮挡被安全检查跳过，增广分支也可能返回未改变的图像；
10% 是明确保留原图的分支概率，不保证实际逐像素改变率恰好为 90%。

前景避让默认使用无需额外权重的保守图像检测：保护边缘和封闭轮廓、明显有颜色或纹理的区域，
额外保护中心宽/高各 60% 的操作区域，仅允许低纹理、低饱和度且与图像边界连通的候选背景区域。
保护区域向外扩张 5 像素，取当前/未来帧保护掩码并集，并随图像应用相同几何变换。
最多尝试 64 个矩形，必须与保护区域零重叠，否则跳过遮挡；不为了达到 95% 而覆盖检测到的前景。
这不是语义分割，不能保证识别所有任务物体，实际遮挡率依图像而定。
`augment_observation(..., foreground_masks=...)` 也支持外部提供逐相机 `[T,H,W]` 前景掩码。

同一视角当前/未来帧共享操作选择、参数、遮挡位置/纯色/像素噪声、传感器噪声场和压缩质量；
马赛克只对各帧原有对应区域降采样。不同相机使用不同随机流；动作、状态、语言、任务标签不被修改。
深度与 DINO 监督也读取增广后的图像。所有操作限于这里列出的二维图像变换/退化，
不执行背景/材质替换、真实物体粘贴、物体/场景三维几何编辑、重光照/光源布置或生成模型编辑。

```bash
cd /root/autodl-tmp/projects/lingbot-vla-v2
# 仅验证数据、权重、依赖和配置
CHECK_ONLY=1 bash experiment/robotwin/train_z4_official_from20k.sh
# 启动独立 Z4 训练；确认 GPU 已可用后执行
CUDA_VISIBLE_DEVICES=0,1 MAX_STEPS=1000 SAVE_STEPS=100 \
  bash experiment/robotwin/train_z4_official_from20k.sh
```

默认从相同 `global_step_20000` 权重初始化，Muon、每卡 batch 32、双卡累积 16 次、
global batch 1024，输出到 `outputs/robotwin-z4-from20k-gbs1024`。
`LINGBOT_Z4_CONFIG=/absolute/path/custom.yaml` 可指定参数文件，启动时会验证并冻结到
`launch/z4_augmentation.yaml`，进程使用冻结后的文件；`launch/recipe.json` 同时记录所有参数。
若已有检查点的目录记录了其他策略或不同 Z4 参数，启动器要求使用新的 `OUTPUT_DIR`。
新增 Z4 不会将当前已运行的 Z3a 进程切换为 Z4，也不会自动启动另一份训练。

验证：10 项 Z4 测试、5 项 B 增广和 7 项 PICA 回归测试通过；另验证了原生训练配置、
启动前检查、相同策略恢复/不同策略拒绝，以及 3 个真实数据样本的完整图像预处理。
实际样本检查确认原图分支逐像素一致、增广图像发生变化、动作/状态/语言/关节掩码不变。
记录见 [`records/z4_validation_20261022.json`](records/z4_validation_20261022.json)。

## PICA step 2000 → Z4 小学习率微调

`train_z4_from_pica2000.sh` 从
[`Xiaodong/lingbotvla-robotwin-clean-sft-20k-PICA/global_step_2000`](https://huggingface.co/Xiaodong/lingbotvla-robotwin-clean-sft-20k-PICA/tree/main/global_step_2000)
的学生 HF 权重开始新一轮 Z4 微调。保留 PICA 已学到的模型参数，本轮关闭 PICA 对齐损失、
EMA 教师及原 PICA 增广，只使用 Z4 的 10% 原图 / 90% 组合增广，状态噪声为 0。
这属于顺序微调；能否保持 PICA 的鲁棒性收益，需要比较 clean/randomized rollout 成功率。

下载器固定 revision `892824afa4cf7d77ad5d3916045d884f14d37170`，只下载 step 2000
的 3 个模型分片、tokenizer/config，以及仓库根目录的训练配置和归一化统计，
不会误用根目录的另一份模型权重。支持分片断点续传，LFS 文件校验 SHA-256，
普通文件校验 Git blob 哈希，全部完成后生成 `download_manifest.json`。

```bash
cd /root/autodl-tmp/projects/lingbot-vla-v2
/root/miniconda3/envs/lingbotvla/bin/python tools/download_pica_z4_model.py

# 检查数据、模型、辅助权重与依赖，不启动训练
CUDA_VISIBLE_DEVICES=0,1 CHECK_ONLY=1 \
  bash experiment/robotwin/train_z4_from_pica2000.sh

# GPU 可用后启动；默认新增 1000 步，每 100 步保存
CUDA_VISIBLE_DEVICES=0,1 MICRO_BATCH_SIZE=32 \
  LR=1e-5 LR_MIN=5e-6 MAX_STEPS=1000 SAVE_STEPS=100 \
  bash experiment/robotwin/train_z4_from_pica2000.sh
```

- 初始权重：`/root/autodl-tmp/models/Xiaodong-lingbotvla-robotwin-clean-sft-20k-PICA/global_step_2000`。
- 数据：本机已验证的 `data/RoboTwin_lerobot_v30`，2500 episodes；使用下载的
  `robotwin_clean_norm.json`，其内容与 20k clean 基线一致。
- HF 仓库未附机器人映射 YAML；使用仓库内 `configs/robot_configs/robotwin.yaml`，
  其动作/状态切片、相机映射与 20k clean 基线一致。可用 `ROBOT_CONFIG` 覆盖。
- Muon、cosine 学习率 `1e-5 → 5e-6`、无 warmup、关闭额外 MoE 专家学习率倍率。
  该学习率是本次微调的起点，低于来源训练配置和现有 Z4 的 `1e-4 → 5e-5`。
- 双卡每卡 batch 32、梯度累积 16、global batch 1024；也可每卡 64 / 累积 8。
  每卡 48 无法整除 1024，启动器会拒绝。
- 初次运行新建优化器和调度器，步数从 0 开始；没有恢复来源 PICA 的 optimizer/EMA。
  后续从本轮独立输出目录的完整 checkpoint 自动续训。
- 输出：`outputs/robotwin-z4-from-pica2000-gbs1024`；日志 `train.log`，
  HF 权重 `checkpoints/global_step_<本轮步数>/hf_ckpt`。
  不要复用已有 `robotwin-z4-from20k-gbs1024`；启动器会拒绝不同初始权重的目录续训。

`INIT_MODEL`、`NORM_FILE`、`OUTPUT_DIR`、`LR`、`LR_MIN`、`MAX_STEPS`、`SAVE_STEPS`
均可通过环境变量覆盖。HF 导出缺少每步 `lingbotvla_cli.yaml` 时，从该步 `config.json`
读取深度/未来视觉辅助分支配置，并将辅助模型路径改成本机已有权重。

验证记录见 [`records/z4_from_pica2000_validation_20261023.json`](records/z4_from_pica2000_validation_20261023.json)：
16 个文件哈希校验、5 项回归测试、原生配置解析、1708 个张量的模型键名/形状匹配、
2 个真实数据样本预处理和完整启动预检查通过。未执行新训练的 GPU 前反向测试。
