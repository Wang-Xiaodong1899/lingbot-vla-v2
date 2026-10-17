# Lingbot-VLA-V2 动作修正 Flow：模型结构、训练与评测说明

文档时间：2026-10-15 12:35:44 UTC（北京时间 20:35:44）。

本文根据当前仓库实现和实际运行的 `run_config.json` 编写。所有命令均为复现说明，写入本文不会启动或恢复训练，也不会重复启动正在进行的评测。

## 1. 方法定位

这次实验保留已训练好的 Lingbot-VLA-V2，并在其后增加一个小型动作修正 Flow Matching 模块。

基本流程：

```text
当前三视角图像、语言指令、机器人状态
                    │
                    ▼
      冻结的 Lingbot-VLA-V2
      从随机噪声完整采样 10 次
                    │
                    ├── 候选未来动作 A_base：[B, 50, 55]
                    └── 视觉语言条件 c：[B, 2560]
                                      │
当前状态 s：[B, 55] ────────────────────┤
                                      ▼
                        新增动作修正 Flow
                        默认 4 次 Euler 更新
                                      │
                                      ▼
                        修正动作：[B, 50, 55]
                                      │
                                      ▼
                        原有反归一化与机器人动作转换
```

核心目标是学习如何将原模型的候选动作变成更接近真实示范的动作，而不是重新训练整个基础策略。

### 1.1 与最初讨论的历史动作 A2A 的区别

| 项目 | 本地原版 A2A 实现 | 当前修正 Flow 实现 |
|---|---|---|
| 流起点 | 历史实测状态的编码 | 冻结 Lingbot 生成的候选未来动作 |
| 预测空间 | 潜空间 | 原模型归一化后的动作空间 |
| 动作编解码器 | 有历史编码器、未来动作编码器、解码器 | 没有新增潜空间编解码器 |
| 视觉语言能力 | 原版基础策略没有当前 Lingbot 的语言主干 | 复用原 Lingbot 视觉语言特征 |
| 是否需要历史窗口 | 是，默认 8 帧 | 否 |
| 首次推理是否缺历史 | 原版通过重复初始观测补齐 | 不存在这一问题，当前观测即可生成候选动作 |

因此，当前方法更准确的名称是“候选动作到 GT 的第二阶段动作修正流”。它借鉴了使用有信息的起点进行 Flow Matching 的思路，不是对原版历史状态 A2A 的完整复现。

## 2. 基础模型及原动作预测过程

基础模型：

```text
/root/autodl-tmp/models/bs512-step4k-ct-bs768-step2k
```

本地 Qwen3-VL 路径：

```text
/root/autodl-tmp/models/Qwen3-VL-4B-Instruct
```

基础模型相关配置：

| 配置 | 数值 / 含义 |
|---|---|
| 当前观测步数 | `n_obs_steps = 1` |
| 输出动作步数 | `n_action_steps = 50` |
| 动作表示宽度 | `max_action_dim = 55` |
| 状态表示宽度 | `max_state_dim = 55` |
| 基础采样次数 | `num_steps = 10` |
| 基础动作专家 | Transformer，启用 MoE |
| 原 checkpoint 动作训练损失 | `L1_fm` |

55 是模型用于对齐不同动作特征的填充宽度，不代表机器人有 55 个真实可控关节。有效维度由动作 mask 指定。

设真实动作序列为 A，标准高斯噪声为 ε。原模型的流训练路径为：

\[
x_\tau = (1-\tau)A + \tau\epsilon,
\qquad u_\tau = \epsilon-A.
\]

基础模型推理从 τ=1 的噪声出发，以负时间步长积分到 τ≈0：

\[
x \leftarrow x - \frac{1}{10}v_{\text{base}}(x,\tau,\text{观测}).
\]

每次更新同时调整全部 50 个动作位置。它不是先生成动作 1，再自回归生成动作 2。动作 token 之间使用双向注意力，图像语言前缀只需计算一次并缓存 KV。

预测长度和执行长度是两个概念。当前评测配置使用 50 步动作 chunk，但基础接口也支持按部署端指定的长度取用动作。

## 3. 冻结范围与训练范围

### 3.1 完全冻结的部分

`FrozenLingbotProposal` 对整个基础策略调用：

```python
policy.requires_grad_(False).eval()
```

候选动作生成函数使用 `@torch.no_grad()`，并在每次采样前再次设置 `eval()`。

被冻结的部分包括：

- 图像编码器。
- Qwen3-VL 视觉语言主干。
- 原 MoE 动作专家及路由参数。
- 原状态和动作投影层。
- 原深度、视频对齐相关模块参数。

原模型仍然执行真实推理，但不保存用于其反向传播的激活，不计算其梯度，也没有其优化器状态。

### 3.2 唯一训练的部分

优化器只接收新增修正模块的参数：

```python
optimizer = torch.optim.AdamW(
    refiner.parameters(),
    lr=1e-4,
    weight_decay=0.01,
)
```

没有对主模型做全量微调、LoRA 或其他参数更新。

原模型的深度、未来深度和视频辅助损失不在这次训练循环中计算，也不加载这些任务的外部教师模型。基础模型已有的参数和前缀计算结构仍用于生成候选动作和条件特征。

## 4. 修正模块的输入和网络结构

### 4.1 张量定义

| 符号 | 形状 | 含义 |
|---|---|---|
| A_base | `[B, 50, 55]` | 冻结基础模型完整采样得到的候选动作 |
| A_GT | `[B, 50, 55]` | 同一当前时刻起的真实未来动作，仅用于训练 |
| x_t | `[B, 50, 55]` | 当前修正流状态 |
| J | `[B, 50, 55]` 或由 `[B, 55]` 广播 | 有效动作维度 mask |
| P | `[B, 50]` | episode 尾部动作填充标记，仅用于训练目标和损失 |
| s | `[B, 55]` | 当前机器人状态 |
| c | `[B, 2560]` | 当前视觉语言条件 |
| t | `[B]` | 修正流时间 |
| v_theta | `[B, 50, 55]` | 网络预测的修正速度 |

### 4.2 视觉语言条件来自哪里

在原模型计算图像语言前缀时，对以下模块安装临时 forward hook：

```text
core.qwenvl_with_expert.qwenvl.model.language_model.norm
```

根据 `lang_masks` 找到每个样本最后一个有效语言 token，提取其最终归一化特征作为 c。

这个 token 已通过原模型的前缀注意力读取图像和语言信息，因此可以作为视觉语言条件。但它是一个全局条件向量，不等于保留全部图像 patch token，也不直接包含作为 suffix 输入的当前状态；当前状态由独立投影输入修正模块。

hook 只复用原模型真实采样时已经计算出的特征，不额外执行一次视觉语言前向。采样结束后，hook 在 `finally` 中移除。

候选动作和 c 均不依赖 GT 动作，均不保留通向基础模型的梯度。

### 4.3 动作输入投影

对每个动作位置，将以下三项沿最后一维拼接：

```text
当前流状态 x_t：55 维
固定原始候选 A_base：55 维
有效动作维度 mask J：55 维
             ↓
           165 维
             ↓ Linear(165, 256)
           256 维
```

每次迭代都提供固定的 A_base，因此网络始终知道原始候选轨迹，而不是只看到当前中间状态。

### 4.4 条件编码与融合

状态编码：

```text
s：[B, 55] → Linear(55, 256)
```

视觉语言编码：

```text
c：[B, 2560] → LayerNorm(2560) → Linear(2560, 256)
```

流时间编码使用 128 个指数间隔频率，构造 128 维正弦和 128 维余弦特征，再通过：

```text
256 → Linear → SiLU → Linear → 256
```

三种条件相加，并广播加到所有 50 个动作 token 上。另加入形状为 `[1, 50, 256]` 的可学习位置编码。

等价表达为：

\[
h_i = W_a[x_{t,i}; A_{\text{base},i}; J_i]
      + p_i + W_s s + f_c(c) + f_t(t).
\]

### 4.5 Transformer 与输出

| 项目 | 配置 |
|---|---|
| 层数 | 4 |
| 隐藏宽度 | 256 |
| 注意力头数 | 8 |
| 单头宽度 | 32 |
| FFN 宽度 | 1024 |
| FFN 激活 | GELU |
| Dropout | 0 |
| 归一化布局 | Pre-Norm，`norm_first=True` |
| 动作注意力 | 双向自注意力，没有因果 mask |
| 输出层 | `LayerNorm(256) → Linear(256, 55)` |

所有 50 个动作位置都参加双向自注意力。动作维度 mask 不是 Transformer 的时间位置 padding mask；未来 episode 结束位置不会作为条件输入模型。

最后将输出乘以有效动作维度 mask，使无效维度的修正速度为零。

### 4.6 参数量与初始化

默认配置的精确可训练参数量为 **4,035,639**：

| 部分 | 参数量 |
|---|---:|
| 动作输入投影 | 42,496 |
| 状态投影 | 14,336 |
| 视觉语言条件投影 | 660,736 |
| 时间 MLP | 131,584 |
| 4 层 Transformer | 3,159,040 |
| 输出归一化与投影 | 14,647 |
| 位置编码 | 12,800 |
| 合计 | **4,035,639** |

输出 Linear 的权重和偏置初始化为零，所以初始速度场恒为零，完整修正过程的输出等于原模型候选动作。

这保证实验从基础策略行为出发。第一步主要先更新输出层；输出层变为非零后，梯度可以进一步传入前面的网络。

## 5. 修正 Flow 的训练目标

### 5.1 基本路径

修正模块采用与原模型不同的时间约定：t=0 是候选动作，t=1 是 GT。

每个样本独立采样：

\[
t \sim U(0,1).
\]

定义：

\[
\Delta = A_{\text{GT}}-A_{\text{base}},
\]

\[
x_t = A_{\text{base}} + t\Delta,
\qquad u_t=\Delta.
\]

网络预测：

\[
v_\theta(x_t,t,A_{\text{base}},s,c,J).
\]

默认 L1 损失：

\[
L = \frac{\sum M\,|v_\theta-u_t|}{\sum M},
\]

其中 M 是真实动作维度与真实未来时间步的联合 mask。也可通过 `--loss-type mse` 改为平方误差。

### 5.2 当前训练不展开完整修正过程

一次训练前向只随机选择一个 t，计算一次修正网络的速度预测和对应损失。

当前实现没有：

- 在每个训练样本上展开 4 次修正再反向传播。
- 对最终修正结果额外计算重建损失。
- 潜空间重建损失、KL 损失或对比损失。
- 回传原 Lingbot 的梯度。

`--refine-steps 4` 控制推理时的数值积分次数。它不表示一次训练需要运行 4 次修正网络。

### 5.3 为什么必须真实采样候选动作

训练时 A_base 来自基础模型真实的 10 次推理采样，和部署时一致。

不能将原模型训练中由 GT 加噪得到的某个中间结果当作候选动作，因为那个结果已经带有 GT 信息，不能代表部署时的候选分布。

GT 只用于修正流的插值状态和监督目标；基础模型采样和视觉语言条件提取不接收 GT 动作。

### 5.4 随机性

修正流本身不额外添加高斯噪声。但基础模型每次从高斯噪声生成候选，因此候选动作仍有随机性。

同一观测可以在不同训练访问中产生不同候选动作；当前实现在线生成候选，没有预先缓存整个数据集的候选轨迹。

## 6. 无效动作维度和 episode 尾部处理

### 6.1 有效监督 mask

定义：

\[
M_{b,i,d}=J_{b,i,d}\land\neg P_{b,i}.
\]

只对真实关节维度和真实未来动作计算损失。

### 6.2 尾部填充策略

数据读取时，如果未来 50 步超出当前 episode，就重复该 episode 最后一条动作，并用 `action_is_pad=True` 标记。

构造训练路径时，实际采用：

\[
\Delta = \operatorname{where}(M,A_{\text{GT}}-A_{\text{base}},0),
\qquad x_t=A_{\text{base}}+t\Delta.
\]

因此无效未来时间位置保留候选值，不通过 GT 填充值影响有效 token 的插值输入，也不计入速度损失。

未来 episode 结束标记不作为网络条件输入，因为部署时无法预先知道 GT 轨迹何时结束。

### 6.3 动作坐标一致性

候选动作、GT、中间状态和修正速度都使用基础策略的归一化、填充动作表示。

先修正，再调用原来的 `FeatureTransform.unapply` 反归一化和恢复机器人动作布局。不能先把候选动作反归一化，再送入当前修正模块。

## 7. 数据集与预处理

本次实际训练使用：

```text
/root/autodl-tmp/data/robotwin_clean50/robotwin_clean50.txt
```

当前清单包含 **50 个任务、每任务 50 个 episode，共 2500 个 episode、549,787 个当前帧样本**。

全部 episode 都进入训练数据集，不留验证集，不执行训练时评估，不根据验证成绩选择 best checkpoint。

每个样本包含：

- 当前顶部、左腕和右腕相机图像。
- 当前状态。
- 当前任务语言指令。
- 从当前帧开始的未来 50 步 GT 动作及尾部 mask。

### 7.1 本地 LeRobot v2.1 适配

本机安装的 LeRobot 为 v3，而训练数据为 v2.1。新增 `LocalV2RefinementDataset` 直接读取本地 v2.1 的：

```text
meta/info.json
meta/episodes.jsonl
meta/tasks.jsonl
data/.../episode_XXXXXX.parquet
videos/.../episode_XXXXXX.mp4
```

这样不需要修改已安装的 LeRobot 版本，也不需要转换原数据格式。

适配层复用仓库已有的 `FeatureTransform` 和视频解码函数，不自行重新定义归一化。

### 7.2 预处理配置

- 从基础 checkpoint 的 `lingbotvla_cli.yaml` 读取数据参数。
- 机器人特征映射使用 `configs/robot_configs/robotwin.yaml`。
- 归一化统计使用机器人配置引用的统计文件；默认是 `assets/norm_stats/robotwin.json`。
- 默认图像先 resize 至 256×256，再交给基础 Qwen processor。
- 保留原多视角、语言模板、有效维度映射和归一化语义。
- 本轮关闭图像增强，不读取未来图像，不产生深度或视频教师目标。
- 单进程内缓存最多 4 个 episode 的 Parquet 数据。
- `workers=0` 为默认值，避免视频解码器多进程退出时的问题。

采样器使用按 epoch 重排的 `DistributedSampler`，并对不足一个完整分布式 batch 的 epoch 尾部使用 `drop_last`。全部 episode 都具有被采样资格，但不保证每个 epoch 都恰好访问每一帧；下一 epoch 会重新打乱。

## 8. 实际训练配置

下表来自本轮实际保存的 `outputs/action-refiner-bs768/run_config.json`，不是命令行默认值列表。

| 项目 | 实际配置 |
|---|---|
| GPU | 4 张 A800 80GB |
| 基础模型精度 | FP32 |
| 修正模块精度 | FP32 |
| 每卡 micro batch | 192 |
| 梯度累积 | 1 |
| 有效全局 batch | **768** |
| 优化器 | AdamW，仅优化修正模块 |
| AdamW betas / eps | PyTorch 默认 `(0.9, 0.999)` / `1e-8` |
| 学习率 | `1e-4` |
| Weight decay | `0.01` |
| 学习率预热 | 100 次优化器更新 |
| 预热后调度 | 不设 max_steps 时保持 `1e-4` |
| 梯度裁剪最大范数 | `1.0` |
| 保存间隔 | 500 次优化器更新 |
| 日志间隔 | 10 次优化器更新，另外记录第 1 步 |
| 总训练步数上限 | 不设置，直到手动停止 |
| DataLoader workers | 0 |
| 随机种子 | 42，各 rank 初始化使用 `42 + rank` |
| 基础模型采样次数 | 10 |
| 修正推理次数 | 4 |
| 训练时评估 | 无 |

### 8.1 全局 batch 的计算

\[
B_{\text{global}}=N_{\text{GPU}}\times B_{\text{per-GPU}}\times G.
\]

本轮：`4 × 192 × 1 = 768`。

此前讨论的 `4 × 48 × 4 = 768` 也能从头训练，但不是本轮实际运行配置，不能直接拿来精确恢复本轮断点。

### 8.2 分布式与梯度归一化

每张 GPU 放一份完整、冻结的基础模型；只有小型修正模块使用 DDP 同步梯度。基础模型没有做 FSDP 分片。

对每个 micro batch，训练器跨 rank 汇总有效动作元素数。每个 rank 使用：

\[
L_r=\frac{N_{\text{GPU}}\cdot\text{本 rank 有效误差和}}
{\text{全局有效元素数}\cdot G}.
\]

DDP 的梯度平均与分子中的 GPU 数抵消，得到该 micro batch 的全局有效元素平均损失。

若 G>1，则对各 micro batch 的归一化损失再等权累积。若不同 micro batch 的有效元素数不同，它不严格等于把所有累积样本一次拼接后的元素加权均值。本轮 G=1，不涉及这一差别。

### 8.3 学习率日志的含义

`LambdaLR` 在初始化时设置首个更新的学习率，训练循环在 `optimizer.step()` 后调用 `scheduler.step()`，然后写日志。

因此，第 1 步日志显示的 `2e-6` 是该次调度后、供下一次更新使用的学习率；第一次更新本身使用约 `1e-6`。前 100 次更新完成预热，之后保持 `1e-4`。

如果显式传入 `--max-steps N`，代码会启用有限步数的余弦调度，预热后最终降至基础学习率的 10%。本轮没有设置该参数。

## 9. 单次训练更新的完整流程

```python
for optimizer_step in training_steps:
    refiner.train()
    optimizer.zero_grad(set_to_none=True)

    for micro_batch in accumulation:
        batch = next_training_batch()

        # 基础策略保持 eval、requires_grad=False；以下过程 no_grad。
        proposal, context = frozen_base(batch)

        target = batch["actions"]
        mask = joint_mask & ~action_is_pad[..., None]
        delta = where(mask, target - proposal, 0)
        t = uniform_time_per_sample()
        x_t = proposal + t[:, None, None] * delta

        # 只有这个前向及其损失参加反向传播。
        velocity = refiner(x_t, t, proposal, state, context, joint_mask)
        loss = distributed_masked_velocity_loss(velocity, delta, mask)
        loss.backward()

    clip_grad_norm_(refiner.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if optimizer_step % 500 == 0:
        save_refiner_optimizer_scheduler_and_rng()
```

代码检查非有限损失和梯度；遇到 NaN/Inf 会报错，而不是继续静默更新。

## 10. 训练、检查与恢复命令

### 10.1 从头训练：复现本轮实际配置

因为当前输出目录可能已有 checkpoint，下面从头训练使用独立的新目录。它会随机初始化新修正模块，不会恢复已有训练。

```bash
cd /root/autodl-tmp/projects/lingbot-vla-v2

CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
bash scripts/train_action_refiner.sh \
  --base-model /root/autodl-tmp/models/bs512-step4k-ct-bs768-step2k \
  --output-dir outputs/action-refiner-bs768-new \
  --base-dtype float32 \
  --batch-size 192 \
  --grad-accum 1 \
  --save-every 500
```

省略 `--max-steps` 表示持续训练，不会在默认 2000 步自动停止。应在 GPU 资源可用时运行，不要与占满同一批 GPU 的评测同时启动。

### 10.2 显存不足时从头训练的替代配置

| 每卡 batch | 累积次数 | GPU 数 | 全局 batch |
|---:|---:|---:|---:|
| 192 | 1 | 4 | 768 |
| 96 | 2 | 4 | 768 |
| 48 | 4 | 4 | 768 |
| 24 | 8 | 4 | 768 |
| 16 | 12 | 4 | 768 |

不同配置都是可用的从头训练配置，但严格断点恢复要求保持 checkpoint 记录的 micro batch、累积次数和 world size。

### 10.3 仅检查数据

```bash
bash scripts/train_action_refiner.sh --check-data
```

读取一个真实 batch，输出训练帧数与张量形状，不加载基础模型权重，不执行优化器更新。

### 10.4 检查完整采样和反向传播

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/train_action_refiner.sh --batch-size 1 --check-batch
```

加载真实基础模型，运行一次真实候选采样和修正模块反向传播，检查基础模型没有梯度、梯度数值有限、零初始化修正保持原输出。该检查不执行优化器更新，不保存训练 checkpoint。

### 10.5 从当前 latest.pt 恢复训练

```bash
cd /root/autodl-tmp/projects/lingbot-vla-v2

CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
bash scripts/train_action_refiner.sh \
  --base-model /root/autodl-tmp/models/bs512-step4k-ct-bs768-step2k \
  --output-dir outputs/action-refiner-bs768 \
  --base-dtype float32 \
  --batch-size 192 \
  --grad-accum 1 \
  --save-every 500 \
  --resume outputs/action-refiner-bs768/latest.pt
```

如果要固定从第 500 步恢复，可将最后一行替换为：

```bash
--resume outputs/action-refiner-bs768/snapshots/step500.pt
```

两者只有在 `latest.pt` 仍然是第 500 步时才等价。可以先读取 checkpoint 中的 `step` 确认。

恢复会加载修正模块、AdamW 状态、学习率调度器、各 rank 的 CPU/CUDA/Python/NumPy 随机数状态，并根据已完成更新数推算数据 epoch 和 batch 偏移。

恢复时仍会重新加载冻结的基础模型权重，因为基础权重不重复写入修正 checkpoint。

严格恢复要求 checkpoint 元数据和当前配置一致，包括基础模型路径及配置摘要、数据清单及摘要、episode 清单、精度、模型结构、batch、累积次数、GPU 数、种子和调度设置。

## 11. 保存文件及停止行为

```text
outputs/action-refiner-bs768/
├── run_config.json
├── metrics.jsonl
├── latest.pt
├── snapshots/
│   └── step500.pt
└── eval_step500_10episodes/
```

### 11.1 checkpoint 内容

| 字段 | 内容 |
|---|---|
| `format_version` | 当前为 1 |
| `step` | 已完成的优化器更新次数 |
| `refiner_config` | 修正模块结构及推理配置 |
| `refiner` | 修正模块 state_dict |
| `optimizer` | AdamW 状态，包括动量与二阶矩 |
| `scheduler` | 学习率调度器状态 |
| `metadata` | 基础模型、数据、batch、world size 等训练元数据 |
| `rng_by_rank` | 每个 rank 的随机数状态 |

`latest.pt` 不包含整个基础模型，也不是可直接替换基础 Hugging Face checkpoint 的完整 VLA 权重。

### 11.2 保存策略的实际边界

保存策略已于 2026-10-16 更新：每 500 次更新先完整写入并独立保留 `snapshots/step{step}.pt`，然后用 `os.replace` 原子更新 `latest.pt`。每个保存点都包含修正模块、优化器、调度器和随机数状态。

`latest.pt` 使用硬链接引用最新保存文件；后续原子替换它不会改变之前的快照。已存在的同名步数快照不会覆盖，重复写入会明确报错；从较早步数重训形成新分支时，应使用新的输出目录。

早期训练只滚动覆盖 `latest.pt`，当时仅额外固定了 `snapshots/step500.pt` 和 `snapshots/step2500.pt`，因此旧的 1500 步权重没有留存。新保存逻辑不会补回已经丢失的历史权重。

这份第 500 步快照的 SHA256 为：

```text
2aa934ad20c5c74ddac01940d0e53a3e8ca5005c11b7c9cdefc34d1947077072
```

手动停止训练不会自动抢救保存尚未到保存点的更新；恢复时回到最近一次成功写入的 checkpoint。未设置总步数上限时，也没有自动“最后一步保存”的终止点。

### 11.3 训练日志解释

`metrics.jsonl` 记录：

- `step`：优化器更新次数，不是 micro batch 数，也不是控制动作步数。
- `train_flow_loss`：本次优化器更新所处理 micro batch 的有效速度误差和除以有效元素数；不是最近 10 次更新的平均。
- `grad_norm`：裁剪函数返回的裁剪前梯度范数。
- `lr`：调用调度器后、供下一次更新使用的学习率。
- `elapsed_seconds`：本次进程进入训练循环以来的时间，不含基础模型加载时间；续训时重新计时。

训练 loss 是归一化动作空间的速度场误差，不能直接解释为弧度、米或任务成功率。

同一输出目录的日志采用追加写入，历史重启可能出现多个 `step=1`。判断一轮趋势时应先识别最后一次重启，不能把不同运行拼成连续曲线。

## 12. 推理与 RoboTwin 评测

### 12.1 修正积分

推理时只有当前观测，不需要 GT，也不需要真实历史动作。

从候选动作开始：

\[
x_0=A_{\text{base}}.
\]

默认 K=4，按以下时间点预测速度：

```text
t = 0.00, 0.25, 0.50, 0.75
```

每一步：

\[
x_{k+1}=x_k+\frac{1}{K}
v_\theta(x_k,k/K,A_{\text{base}},s,c,J).
\]

最后得到近似 t=1 的修正动作，再反归一化并交给机器人。

原模型 10 次更新和新增模块 4 次更新的单次计算成本不同，不能把它们简单理解为 14 次同等规模的原模型计算。不过相比只运行基础模型，新增修正通常会增加推理开销。

### 12.2 推理接入

新增 `RefinerPolicyServer` 继承原推理服务，复用原图像预处理、机器人特征映射、动作反归一化、chunk 缓存与 WebSocket 协议。

动作处理顺序为：

```text
原始观测 → 原 FeatureTransform → 冻结基础采样
        → 修正 Flow → 原 FeatureTransform.unapply
        → 机器人动作
```

服务会核对基础模型路径、配置文件摘要、基础精度和基础采样次数，并在启动日志打印修正 checkpoint 的 step。

当前修正特征提取依赖临时 forward hook，新增服务要求 `--use_compile False`。评测复用训练时的 FP32 基础模型，不能悄悄改成 BF16 后当作同一个评测条件。

单独启动服务示例：

```bash
cd /root/autodl-tmp/projects/lingbot-vla-v2

CUDA_VISIBLE_DEVICES=0 \
/root/autodl-tmp/conda/envs/lingbotvla/bin/python \
  -m deploy.lingbot_action_refiner_policy \
  --model_path /root/autodl-tmp/models/bs512-step4k-ct-bs768-step2k \
  --refiner_checkpoint outputs/action-refiner-bs768/snapshots/step500.pt \
  --use_bf16 False \
  --use_fp32 True \
  --use_compile False \
  --use_length 50 \
  --port 9530
```

该命令只启动推理服务，不会自动运行 RoboTwin 回合。完整评测由下一节的 driver 调度。

### 12.3 本次 500 步评测方案

| 项目 | 设置 |
|---|---|
| 权重 | 固定的 `snapshots/step500.pt` |
| Clean | `demo_clean` / seen 指令 |
| Randomized | `demo_randomized` / unseen 指令 |
| 任务数 | 每组 50 个 |
| 回合数 | 每任务 10 个 |
| 正式回合总数 | 1000 |
| 并发 | 4 GPU，每卡 1 个服务和对应模拟器槽位 |
| 精度 | FP32 |
| 动作 chunk | 50 |
| 视频 | 关闭 |
| 编译 | 当前新增服务关闭 |

driver 先做单任务单回合 smoke，再按 clean、randomized 顺序执行。smoke 结果不计入正式成绩。

可复现的 driver 命令如下。示例使用新的输出目录，不应与当前正在运行的同端口 driver 重复执行：

```bash
cd /root/autodl-tmp/projects/lingbot-vla-v2

/root/autodl-tmp/conda/envs/lingbotvla/bin/python -u \
  experiment/robotwin/eval_action_refiner.py \
  --checkpoint outputs/action-refiner-bs768/snapshots/step500.pt \
  --output-dir outputs/action-refiner-bs768/eval_step500_10episodes_repeat \
  --start-port 9530
```

driver 会等待需要的 GPU 各有至少 35,000 MiB 空闲显存；它不会主动停止训练或其他评测。

本次实际运行目录：

```text
outputs/action-refiner-bs768/eval_step500_10episodes/
├── driver.log
├── driver.pid
├── status.json
├── launcher_snapshot.sh
├── smoke/
├── clean/
└── randomized/          # 进入对应阶段后生成
```

只有每个任务都结束指定的 10 回合、结果文件齐全且指令类型匹配，driver 才将该阶段标记为完成；不能仅凭 launcher 的退出码宣称整个评测成功。

## 13. 已完成的实现检查

已验证：

1. 真实数据 batch 读取和形状检查，全部训练帧计数为 549,787。
2. 7 项单元测试：零初始化恒等修正、Euler 方向、解析损失、动作维度和时间 padding 屏蔽、无效填充值不影响有效维度、全 padding batch 拒绝、基础模型无 GT 输入和特征提取。
3. 使用真实基础 checkpoint 生成 `[1, 50, 55]` 候选并提取 `[1, 2560]` 条件；修正模块反向传播正常，基础模型无梯度，未执行优化器更新。
4. 两卡合成数据 DDP 训练、checkpoint 保存及恢复；从第 1 步断点恢复后的第 2 步修正权重与连续训练逐位一致。
5. 实际 RoboTwin 端到端 smoke，日志确认第 500 步修正权重已加载，首个 `lift_pot` 回合成功。

这些检查验证实现链路和恢复机制，不证明修正模块在完整 benchmark 上优于基础模型。

## 14. 与官方继续训练的比较边界

| 维度 | 官方代码继续训练 | 当前第二阶段修正训练 |
|---|---|---|
| 基础 checkpoint | 相同起点可作对照 | 固定相同起点 |
| 更新范围 | 按官方配置训练原模型 | 只更新新增 404 万参数 |
| 训练目标 | 原噪声到动作的速度场及配置的辅助目标 | 候选动作到 GT 的速度场 |
| 基础模型是否反向传播 | 通常是 | 否 |
| 是否额外进行基础采样 | 原训练通常不执行完整 10 步采样 | 每个训练样本执行 |
| 推理 | 基础模型生成动作 | 基础模型生成后再修正 |

公平对照需要对齐训练数据、有效全局 batch、累计训练样本量和评测任务/场景/指令条件，并单独记录训练成本与推理延迟。

本轮使用所有 episode，与先前留出 5 个 episode 的早期实现不同。当前代码已经移除了留出与训练时评估。

关于三组评测的命名：

- “原基线”指不加修正模块的 `bs512-step4k-ct-bs768-step2k`。
- `evaluation_step2500_10episodes.md` 中的 2.5k 是另一份 Lingbot checkpoint，不是本修正模块训练到 2500 步。
- “修正 Flow 500 步”指冻结原基线并叠加本文介绍的第 500 次优化器更新权重。

比较逐任务结果时，只对三者都结束全部 10 回合的同一批任务计算合计，不能将修正模型当前已结束的少量回合与基线完整 500 回合直接比较。

## 15. 当前方法的限制与后续实验方向

以下是方法边界和可选实验，不是当前已经实现的功能：

- 单个 2560 维全局条件可能丢失精细空间信息。可以研究对多个视觉语言 token 做 cross-attention，但会增加模块复杂度。
- 速度场 L1 下降不保证完整 4 步积分后的动作更准确，更不保证任务成功率提升。需要看实际 rollout。
- 当前没有直接约束相邻动作平滑度、接触稳定性或关节速度限制。
- 原模型给出的严重错误轨迹可能超出小修正模块的纠错能力。
- 对同一任务可能存在多个合理动作轨迹；示范 GT 与候选动作的差异不一定全部属于必须纠正的错误。
- 应使用一次性残差修正网络作对照，区分“增加参数与第二阶段训练”的收益和“采用 Flow Matching”的收益。
- 可测试不同修正积分步数，但改变步数后应重新评测成功率与延迟。
- 若将来联合解冻基础模型，候选分布和条件特征会变化，不能直接沿用“基础分布固定”的假设。

## 16. 对应代码与记录

| 文件 | 用途 |
|---|---|
| [action_refinement.py](../lingbotvla/models/vla/lingbot_vla/action_refinement.py) | 配置、修正 Transformer、mask、损失辅助函数、冻结基础采样封装、推理封装 |
| [train_action_refiner.py](../tasks/vla/train_action_refiner.py) | 数据初始化、在线候选生成、DDP、优化器、学习率、日志、保存、恢复 |
| [train_action_refiner.sh](../scripts/train_action_refiner.sh) | 单卡 / 多卡启动与本机 Python 环境选择 |
| [refinement_dataset.py](../lingbotvla/data/vla_data/refinement_dataset.py) | 本地 LeRobot v2.1 数据适配 |
| [utils.py](../lingbotvla/data/vla_data/utils.py) | 原 FeatureTransform 与特征映射 |
| [robotwin.yaml](../configs/robot_configs/robotwin.yaml) | 机器人动作状态映射及归一化统计路径 |
| [modeling_lingbot_vla_v2.py](../lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py) | 基础 FlowMatchingV2 和原候选动作采样 |
| [lingbot_action_refiner_policy.py](../deploy/lingbot_action_refiner_policy.py) | 新增 RoboTwin 修正策略服务 |
| [eval_action_refiner.py](../experiment/robotwin/eval_action_refiner.py) | 单回合检查、clean、randomized 持久评测调度 |
| [test_action_refinement.py](../tests/test_action_refinement.py) | 单元测试 |
| [smoke_action_refiner_training.py](../tests/smoke_action_refiner_training.py) | 合成数据 DDP 保存和精确续训检查 |
| [run_config.json](../outputs/action-refiner-bs768/run_config.json) | 本轮实际运行参数 |
| [metrics.jsonl](../outputs/action-refiner-bs768/metrics.jsonl) | 本轮训练日志 |
| [status.json](../outputs/action-refiner-bs768/eval_step500_10episodes/status.json) | 第 500 步评测实时阶段状态 |
| [robotwin_evaluation_results.md](../robotwin_evaluation_results.md) | 原基线评测报告 |
| [evaluation_step2500_10episodes.md](../evaluation_step2500_10episodes.md) | 另一份 2.5k Lingbot checkpoint 的评测报告 |

后续若代码、数据处理或训练配置发生变化，应以对应运行的 `run_config.json`、checkpoint 元数据和代码版本为准。本文记录的是上述时间点的实现。
