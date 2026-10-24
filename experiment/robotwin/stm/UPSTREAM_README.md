# LingbotVLA STM source snapshot

整理于2026-10-23，包含本地STM源码副本及训练、推理、评估配套实现。

- `src/`：LingbotVLA源码，包括STM模型、数据边界修复、训练入口、推理缓存以及原有依赖模块和LICENSE。
- `tests/`：STM及推理缓存测试。
- `training/initial_candidate/`：早期500步候选配置及Ascend启动脚本；不是8×H20正式实验配置。
- `evaluation/queue/`：权重发布、单流服务、历史采样和队列。
- `evaluation/lazy_render/`：按需历史渲染及验证脚本。
- `evaluation/parallel/`：连接隔离、多仿真任务池与验证脚本。
- `evaluation/optimized/`：通用评估调度等支持脚本。
- `docs/`：详细中文实现文档，含源码快照。
- `MANIFEST.json`：逐文件来源与SHA256。

这是代码快照，不包含模型权重、训练数据、RoboTwin/CuRobo资产或完整现场部署环境。现场生成的STM评估client和正式H20训练配置未从远端取回。各目录原README中的“当前”指对应记录时间，不代表实时状态。现有脚本保留原路径与参数，运行前需调整环境；部分调度入口含进程管理逻辑，应先阅读。源码未在本次重新运行训练或闭环评估。

模型服务的FP32正式评估参数与早期launch_stm.sh中的BF16通用示例不同，参见详细文档。原始源码及依赖模块许可证保留在各自目录。
