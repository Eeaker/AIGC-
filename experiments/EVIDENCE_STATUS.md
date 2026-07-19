# 实验证据状态

正式结论只采用包内能够定位到配置/代码、指标与产物的实验。

| 状态 | 实验 | 包内证据 |
|---|---|---|
| 可复核 | A 九 checkpoint 精确重建 | `models/task_a_lofo/`、`tools/rebuild_task_a_from_checkpoints.py`、`task_a/checkpoint_rebuild_manifest.json` |
| 可复核 | B RAFT/MIBA 近似与学习清理 | `task_b/autodl/` |
| 可复核 | B 全局 TPS、局部残差融合 | `task_b/topology_visibility_v2/` |
| 可复核 | B 部件图形变、Assisted-10/30 | `task_b/part_graph_warp/`、`task_b/assisted_local_correction/` |
| 可复核 | C 五档人工量—效果曲线 | `task_c/c_profile_metrics.json` |
| 记录但不量化 | AnimeInbet、JoSTC、GlueStick、soft-clDice 历史试验 | 缺少完整配置—命令—输出链，不用于正式数值结论 |

`diagnostic-only` 表示该实验可能读取成品作分析或调参，不等于可用于盲测成绩。
