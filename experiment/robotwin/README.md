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
运行训练时由启动脚本重新生成配置。当前没有训练 loss 日志，因为尚未启动训练。
