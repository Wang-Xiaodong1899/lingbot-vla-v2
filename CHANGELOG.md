# 修改日志

## 2026-10-17

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
