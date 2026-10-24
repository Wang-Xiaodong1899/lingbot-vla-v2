# LingBot-VLA v2：当前帧 + −0.5s / −1s patch 时序记忆

已在独立源码副本实现。原4000步增广训练源码、配置和进程未修改。远端完整代码位于：

`/home/tiger/workspace/lingbot_vla2_910b_20261012/temporal-memory-20261021/src`

本地 `src/` 是本次审阅涉及的源码快照；远端 `src/` 是可运行的完整包。本地 `temporal-memory.patch` 可应用于本次记录的原源码版本，`manifest.json` 记录修改前后哈希。不要把补丁直接覆盖到正在运行的训练目录。

## 实现

- 配置 `model.temporal_memory` 默认为 false；只有开启时创建新参数、读取历史帧。历史窗口固定为 `[0.5, 1.0]` 秒。
- ViT/patch merger 输出之后添加1层逐视角、逐空间位置的时间注意力。256维瓶颈、4头；当前patch查询当前和两个历史patch。相对时间编码使用实际帧间隔。
- 只返回当前时刻的视觉tokens，语言模型序列长度保持不变。DeepStack沿用当前图像特征；本版不对DeepStack添加时间融合。
- 残差输出投影为零初始化，新增模块初始精确等价于基线；其他层正常初始化以便输出投影获得梯度。历史缺失时即使适配器训练后，也完全走原当前帧特征。
- 默认试验配置冻结主ViT，历史编码不构建反向图；其他原本可训练参数及适配器继续训练。没有将全模型冻结成adapter-only。
- base SFT首次加载只允许整个新增适配器缺失，并仅初始化它；若基础参数缺失或适配器只缺部分参数，报错。加载已训练的适配器时保留其权重。

## 数据

每个样本的顺序是：当前、−0.5s、−1s、可选未来辅助监督帧。历史与未来监督走独立字段。三视角及当前/历史/未来图像共享同一组随机颜色增广参数。当前与历史的patch网格必须一致。

根据数据集metadata fps将时间间隔向更早帧取整：`-ceil(seconds*fps)`，不读取晚于指定时间的帧。实际RoboTwin数据为50fps，恰好对应−25和−50帧；15fps数据则为−8和−15帧。真实时间差单独传入模型。

episode开头通过时间戳生成history_valid，缺失历史被mask，不跨episode读取。训练输入新增：

- history_images: `[B,2,V,L,D_pixel]`
- history_grid_thw: `[B,2,V,3]`
- history_img_masks: `[B,2,V]`
- history_ages: `[B,2]`

真实样本验证得到history_images `[2,3,256,1536]`；collator输出 `[B,2,3,256,1536]`。

## 推理接口

默认关闭时现有接口不变。开启时需要环境时间戳（秒），不能用网络到达时间或墙钟时间代替。每个episode必须reset；多条流必须使用不同stream_id。

服务器支持两条路径：

1. **时间戳特征缓存**：发送带 `timestamp`、`episode_id`、`stream_id` 的正常观测；非决策时可以发送 `observe_only: true` 更新缓存，不计算动作。历史按目标时间之前最近的特征查询，允许最多早0.1秒；缺失/太旧则mask。缓存有条目数和时间窗口上限，episode变化清空；现有默认reset会清空全部流缓存。已缓存的历史不重复编码。
2. **动作chunk客户端缓冲（推荐现有RoboTwin使用）**：`deploy.temporal_client.TemporalObservationClient` 在每个环境/相机观测时刻保存原图，在需要新动作chunk时选出两张历史观测，随当前观测一次发送。服务器只编码这三个选中时刻，不必在每个控制步运行ViT。此路径历史未做跨请求特征复用，不能宣称每次只编码当前帧。

```python
from deploy.temporal_client import TemporalObservationClient
memory = TemporalObservationClient(websocket_client, stream_id="robot0")
memory.reset(episode_id="unique-episode-id", robo_name="robotwin")
# 每个环境观测步都执行（即使还在执行上一个动作chunk）：
memory.observe(observation, timestamp=simulation_time_seconds)
# 只有需要新动作chunk时执行：
result = memory.infer()
```

客户端复制原观测数组，避免环境复用内存污染历史。缺失历史发送None。服务器验证历史的episode、stream、时间区间和patch网格，不接受未来帧。当前推理应 `use_compile=False`。客户端辅助类用于单流；服务器batch模式要求每条观测有唯一stream_id。

如果评估器每执行50动作才采集一次图像，必须把observe接到执行chunk期间的相机观测循环；只有两次决策图像无法凭空得到−0.5s/−1s。当前基线评估脚本未被强制替换。

## 测试与运行边界

已在开发机CPU执行13项测试，全部通过：零初始化基线一致、适配器梯度、无历史/NaN padding、视角和patch隔离、checkpoint roundtrip、仅新增适配器初始化、时间因果性、episode重置、缓存上限、客户端数组复制、推理流隔离、实际视觉接线及DeepStack保留等。

真实RoboTwin adjust_bottle 数据读取与collator测试通过，检查index0/40/100的无历史/一段历史/两段历史掩码；同一数据集fps为50。详细结果见 `unit-test.log` 与 `data-test.log`。

尚未执行6B模型的NPU完整前后向、16卡容量/吞吐测试或RoboTwin闭环成功率评估：现有16卡正在跑用户授权的4000步基线，本次未占用其NPU。历史编码会增加训练数据读取和视觉前向成本，不能把原micro16吞吐视为本版本吞吐。

`train.temporal.yaml` 为未启动的500步候选配置，micro16/累积2/global512，冻结主ViT，保存每250步。正式启用前先进行短程NPU前后向与保存/重载验收；本次未启动新训练或修改当前正式训练。
