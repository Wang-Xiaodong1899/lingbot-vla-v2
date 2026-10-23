# 修改日志

## 2026-10-23

### 从 PICA step 2000 进行 Z4 小学习率微调

- 新增 `tools/download_pica_z4_model.py`，固定 Hugging Face 仓库 `Xiaodong/lingbotvla-robotwin-clean-sft-20k-PICA` 的 revision `892824afa4cf7d77ad5d3916045d884f14d37170`，下载 `global_step_2000` HF 权重及配套元数据，支持断点续传和 SHA-256 / Git blob 校验，避免下载根目录的其他模型权重。
- 新增 `experiment/robotwin/train_z4_from_pica2000.sh`：从 PICA 学生权重初始化，仅用 Z4 进行顺序微调，关闭 PICA 对齐与 EMA；默认 Muon、学习率 `1e-5 → 5e-6`、双卡 `32 × 2 × 16 = 1024`、新增 1000 步、每 100 步保存，使用独立输出目录。
- 公共配置生成器支持从 HF `config.json` 读取辅助分支配置，无需权重子目录内存在 `lingbotvla_cli.yaml`；支持 `ROBOT_CONFIG` 显式指定映射，记录初始化来源与学习率，并阻止已有 checkpoint 的输出目录切换初始权重。
- 新增配置生成回归测试，覆盖 HF-only 初始化、关闭 PICA、保留辅助架构、续训来源与策略隔离、旧 CLI 配置兼容以及学习率校验。新增 README 下载、预检查和训练命令。
- 本机已完成 16 个文件下载和哈希校验；5 项回归测试、原生配置解析、1708 个 checkpoint 张量的 meta 模型键名/形状匹配、2 个真实数据样本预处理及新旧启动器预检查通过。验证记录见 [`experiment/robotwin/records/z4_from_pica2000_validation_20261023.json`](experiment/robotwin/records/z4_from_pica2000_validation_20261023.json)；未启动新训练或执行 GPU 前反向测试。

## 2026-10-22

### Z4 在线随机组合增广

- 新增独立训练策略 `LINGBOT_AUG_POLICY=z4`：按观测样本以 10% 概率保留所有相机的当前/未来原图，90% 进入随机组合分支，接入训练 DataLoader 的图像预处理；原图分支跳过原有色彩增广，Z4 启动器关闭状态噪声。
- 在增广分支内配置六类操作：色彩/色温 80%（亮度 0.25–1.10、对比度 0.5–1.5、饱和度 0.4–1.6）；缩放/平移 50%（0.9–1.1、各轴不超过 6%、黑边）；矩形遮挡请求 95%（纯色/马赛克/像素噪声等概率）；高斯/椒盐噪声 50%/15%；高斯/运动模糊 25%/10%；JPEG 30%（质量 35–90）。同组噪声、模糊各自互斥；色彩和遮挡关联采样，在保持各自边际概率的同时避免空操作组合。
- 矩形遮挡使用保守的边缘、颜色、纹理和边界连通区域检测，并额外保护中心区域；避让当前/未来帧保护掩码的并集及安全边距，找不到安全位置时跳过。此检测不属于语义分割，不保证识别所有任务物体；95% 为尝试概率，实际遮挡率会更低。
- 同一视角当前/未来帧共享操作参数、几何变换、遮挡位置与随机噪声场；只执行指定的二维图像退化，不替换背景/材质、不粘贴真实物体、不编辑三维场景或光源、不使用生成模型、不修改动作/状态/语言标签。
- 新增 `configs/augmentation/z4.yaml`、独立启动脚本 `experiment/robotwin/train_z4_official_from20k.sh`，复用官方数据和 Muon 训练流程，默认双卡 `32 × 2 × 16 = 1024`，使用独立输出目录并冻结本轮增广参数快照。已有检查点的运行拒绝静默切换增广策略。
- 10 项 Z4 CPU 测试、5 项 B 增广回归测试和 7 项 PICA 回归测试通过；原生配置解析、Z4 启动预检查、续训策略保护及 3 个真实数据样本的预处理检查通过，确认原图分支逐像素一致、增广分支标签不变。检查记录见 [`experiment/robotwin/records/z4_validation_20261022.json`](experiment/robotwin/records/z4_validation_20261022.json)；未启动 Z4 GPU 训练。

### 官方 RoboTwin v3 数据的 aug1 / Muon 续训准备

- 新增 `experiment/robotwin/train_z3a_official_from20k.sh` 和 `tools/prepare_z3a_official.py`，从 `hggforget/lingbotvla-robotwin-clean-sft/global_step_20000` 初始化，使用官方 `RoboTwin_lerobot_v30` 数据和模型随附的归一化统计、机器人关节映射。
- 使用原生 Muon，关闭额外 MoE 专家学习率倍率；启用 level 1 视觉增广及 `0.01` 归一化状态噪声，保留深度/DINO 辅助损失，关闭 PICA。默认学习率 `1e-4 → 5e-5`、cosine、无 warmup。
- 默认每卡 micro batch 64，按可见 GPU 数自动计算梯度累积，global batch 固定为 1024；本机双卡为 `2 × 64 × 8 = 1024`。启用 FSDP 全分片与梯度检查点，默认新增训练 1000 个优化器更新，每 100 步保存。初始目录不含优化器状态，首次运行新建 Muon 状态，从本轮 step 0 开始。
- 启动脚本提供 `DRY_RUN=1` 和 `CHECK_ONLY=1`，检查 batch 整除关系、数据完整性、权重和运行依赖；支持 `.zip.part` 下载完成后的 SHA-256 校验及解压，不读取未完成的数据。
- 新增 `tools/download_z3a_assets.py`，固定官方仓库 revision，支持辅助权重断点续传和哈希校验。本机已补齐 MoGe、LingBot-Depth、DINO-VIDEO 权重及指定版本的 `utils3d` 依赖；三个辅助权重和初始 20k 模型的 6 个分片均通过 SHA-256 校验，DINO 配置通过 Git blob 哈希校验。
- 原生训练器配置解析、增广测试、真实数据单样本读取、三个辅助模型实际加载及完整启动预检查均通过；数据共 2500 episodes、548893 frames。配置快照、校验清单和检查日志归档到 [`experiment/robotwin/records/z3a_official_from20k_20261022/`](experiment/robotwin/records/z3a_official_from20k_20261022/)，模型和数据二进制保留在本地。
- 当前为本地适配版：缓存的 `Auranj/lingbot-vla` 分支版本 `ffbf35b` 不含 `Z3a-official-data-aug1` 配置，拉取更新时遇到认证限制，尚未核对上游 Z3a 的完整设置；直接使用发布数据中的指令，未应用实验仓库的 seen 指令替换流程。尚未启动正式训练、实测 batch 64 的训练显存或评估训练效果。

## 2026-10-21

### PICA-WAM 训练方法适配 LingBot-VLA-V2

- 基于 clean SFT 的 `global_step_20000` 新增 PICA 微调配置及启动脚本，只读取 clean 演示；默认 30,000 次优化、global batch 16、学习率 `1e-5 → 5e-6`。V2 的 batch 单位为观测/动作 chunk，学习率为本次适配选择，不等同论文的完整 episode 训练设置。
- 新增独立相机、时间共享的视觉增强：光度扰动、高斯噪声、小范围裁剪与旋转；教师读取原始图像，学生当前帧和未来帧共用增强参数与噪声场。
- 36 层、宽度 768 的 action 分支增加第 8 层到 EMA 教师第 20 层的表征对齐；投影头为 Linear(768,512)–GELU–Linear(512,768)，使用权重 0.05 的逐 token 负余弦损失，排除状态及 padding token。
- 学生与教师共享相同动作噪声及时间步；教师无梯度并在目标层提前停止。V2 没有论文的视频生成噪声变量，因此不引入虚构的视频噪声差。保留基线动作、深度、未来视觉特征和 MoE 损失，视觉辅助目标来自增强后的图像。
- 采用 FSDP2 分片的 FP32 EMA，decay 为 0.9999，每次 optimizer step 后更新。DCP 保存/恢复学生、投影头、教师及优化器等训练状态；HF 推理权重移除投影头，继续使用标准学生策略。
- 新增 TensorBoard PICA 对齐损失、训练资源预检、使用说明与 7 项 CPU 回归测试；测试已通过，覆盖共享噪声、增强一致性、梯度、提前停止、EMA、DCP 恢复及 HF 导出。当前环境缺少 CUDA 与配置引用的部分数据/权重，未执行 GPU 训练或 clean/randomized 效果评测。

## 2026-10-20

### 可选 B 训练增广

- 参考 `lingbot-vla` 的 `B-aug1-2k-works`，移植 patch 0001/0002 的在线视觉增广和归一化状态噪声。
- 新增 `LINGBOT_B_AUG=1` 总开关，默认关闭；支持独立覆盖视觉级别和状态噪声强度。
- 启用 B 时开启训练图像增广；关闭时仍遵循原有 `image_augment` 配置。动作标签不加噪，推理路径不启用额外增广。
- README 补充参数、作用范围和训练用法；新增并通过 5 项 CPU 单元测试，覆盖默认行为、开关、时序一致性、状态噪声和推理隔离。本次未运行训练或重新验证成功率。

## 2026-10-19

### VLA-Corrector

- 新增冻结基础策略的视觉动力学 corrector：按 episode 提取视觉特征、量化缓存、划分训练与验证集，并训练残差 MLP。
- 新增分布式训练、快照保存和严格恢复，以及缓存完整性检查与训练/评测交接监控工具。
- 新增逐动作观测的推理服务，支持鲁棒异常检测、丢弃过期计划及 OGG 恢复引导。
- 新增 RoboTwin 10k corrector 训练与评测入口、专用客户端、批次检查工具及单元测试。
- 新增基线与多 checkpoint 顺序评测监控脚本；实验路径和运行设置见配套说明，尚不据此宣称 benchmark 提升。
- 保留引入的上游 MLP 许可和来源记录。

### 推理与验证

- 基础推理支持显式指定归一化文件；训练配置中的绝对路径不可用时，在模型目录及父目录查找同名文件，并明确报告缺失。
- AutoHorizon GPU 冒烟工具区分生产 MoE 重复采样抖动与注意力采集一致性，使用参考 MoE 检查精确动作一致性，并验证生产动作前缀切片。
- 更新 AutoHorizon 已有验证记录和评测状态说明。

### 提交范围

- 仅提交源码、测试、启动脚本、许可和文档；不包含模型权重、数据集、特征缓存、checkpoint 或运行日志。

## 2026-10-17

### AutoHorizon 推理适配

- 新增 Lingbot 10k 的 AutoHorizon 服务和 clean/randomized 评测启动脚本，沿用模型随附归一化文件。
- 保留上游双向 soft-pointer 算法及 Apache-2.0 许可；按上游默认规则采集第 3 次去噪的动作注意力。
- 兼容 Lingbot grouped-query attention、完整 key 归一化和动作 token 切片；通过独立只读采集保留原模型输出。
- 采用 eager、关闭编译；新增上游一致性、注意力提取、状态重置测试和真实模型固定种子动作一致性验证工具。
- 全部 12 个单元测试通过；真实 10k GPU 验证尚未完成，现有 action refiner 训练占用各卡约 67 GB，未启动完整 AutoHorizon 评测。

### RoboTwin 推理与评测

- 新增 Lingbot VLA v2 的 ChunkTrust AHS 动态动作执行步长适配，无需训练 QHA。
- 从 flow matching 采样器导出去噪速度轨迹；默认采样接口保持兼容。
- 仅在 14 个有效归一化动作维度上计算 AHS 证据，排除填充维度。
- 评测客户端回传实际执行动作数，每回合重置 selector，并用回合种子初始化状态；未执行的动作不写入历史。
- 推理端向上查找 `lingbotvla_cli.yaml`，兼容 HF 仓库的 `global_step_10000` 目录布局。
- 新增 10k checkpoint 评测脚本，默认 4 卡、每卡 2 路，依次评测 clean 和 randomized，各 50 个任务、每任务 10 回合。
- 评测启动器支持指定回合数、复用已完成任务、重跑未完成任务、服务就绪检查及客户端同步。
- 使用模型仓库随附的 `robotwin_clean_norm.json`；本地原有默认 `assets/norm_stats/robotwin.json` 已移除，运行时需提供对应归一化文件。

### 数据准备与训练

- 新增 RoboTwin clean50 下载、转换、并行预处理及归一化统计工具。
- 新增本地 clean50 数据清单与 4k、从 4k 继续训练、恢复训练配置及启动脚本。
- 查询动作序列时先选择数据列，避免不必要的相机图像解码。
- 兼容 LeRobot 不同版本的常量导入路径。
- 新增训练资源监控、训练损失与 action refiner 损失绘图工具；机器本地路径需要按部署环境调整。

### Checkpoint 完整性

- 分布式保存先写入临时目录，检查元信息、分片字节范围及各 rank 状态，再 fsync 并原子发布。
- 保存成功标记 `_SUCCESS.json`，默认降低 DCP 写入线程数，并记录保存阶段的内存状态。
- 严格恢复时不再吞掉 optimizer 加载错误，也不在所有 checkpoint 恢复失败后静默从头训练。
- 同步导出 HF 权重时让所有 rank 等待，减少与下一步训练的内存峰值重叠。
- 新增 checkpoint 完整性检查及分布式保存/恢复冒烟验证工具。

### Action refiner

- 新增冻结 Lingbot 基础策略的动作 flow refiner、数据集适配、训练入口、恢复与推理服务。
- 提供训练和服务启动脚本，以及动作掩码、填充隔离、梯度与数值行为测试。
- 此模块与 ChunkTrust AHS 是独立功能；以下 AHS 评测结果不代表 action refiner 的效果。

### 验证与评测结果

- ChunkTrust 状态测试与 action refiner 单元测试通过；新增/修改 Python 文件完成语法编译检查，Shell 脚本通过语法检查。
- 10k checkpoint 的 6 个权重分片均通过 HF 元信息 SHA256 校验。
- AHS 的 `lift_pot` 单回合仿真冒烟测试成功。
- 完整 AHS 评测：clean **362/500（72.4%）**，randomized **195/500（39.0%）**；两组各 50 个任务、每任务 10 回合，FP32 推理、开启编译。
- 两组沿用同一 unseen 语言指令协议；seen/unseen 在此指 clean/randomized 环境域。
- 以上没有同配置固定步长基线对照，不能据此推断 AHS 带来的提升。

### 版本管理

- 新增 `.gitignore`，排除 `data/`、`outputs/`、`weights/`、评测环境、下载模型、checkpoint、缓存、日志及生成的训练曲线。
- 仅提交源代码、配置、测试、文档和小型数据清单；模型、原始数据和完整评测日志保留在本地。
