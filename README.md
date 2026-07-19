# 灵图算法笔试 - A/B/C 完整提交（v17）

本目录按“正式结果、算法基线、人机协同、实验证据”分层，避免把看过参考答案后的诊断上限混入正式成绩。正式结果共 18 张 TGA：A 3 张、B 6 张、C 9 张。

## 一眼看懂

- `outputs/official/task_a`：题面要求的严格四色描原；平均 2 px F1 0.9886，四色符合率 100%。参考成品实际存在绿线，因此另在 `outputs/supplementary/task_a_production_5color` 提供五色生产版。
- `outputs/official/task_b`：六张 blind 中割，不加载任何由缺失帧成品训练的权重。A2-A5 平均 2 px F1 0.9322；大转头 A7/A8 为 0.5109/0.5724，失败如实保留。旧 cleanup 结果降为 `outputs/diagnostic/task_b_shot_adapted`。
- `outputs/official/task_c_assisted`：153 个仅依据线稿、设定图和色卡的关键视图 paint-bucket 标签；面积精度 99.56%，正确覆盖 95.63%，原线变化 0。225 点版本仅作诊断，不列为正式结果。
- `outputs/algorithm_only/task_c_automatic`：0 标签自动基线；面积精度 99.97%，但正确覆盖仅 59.25%。
- `outputs/supplementary/ktk05`：题面三项进阶的 A/B 双层结果；其中 C 的 A 层使用成品关键帧监督，已在 `docs/ADVANCED_KTK05.md` 明确披露，不与主镜头正式成绩混算。
- `experiments`：AutoDL、拓扑/可见性标注、负结果、C 五档信息边界和统一实验索引。

## 完整验收

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements-core.txt
.venv\Scripts\python run_all.py --data-root ..\2026.07.13
```

默认不覆盖预生成正式结果，只独立重算 18 张图的指标并检查：尺寸、精确色规、C 输入非白线逐像素不变、C 语义点数量。按需使用 `--regenerate-a-from-checkpoints`、`--regenerate-b`、`--regenerate-c`；正式 B 是不加载同镜头成品监督权重的 blind 版本，`--regenerate-b-shot-adapted` 只写入诊断目录。

## 入口

- 4 页报告：`report.pdf`
- 可阅读长版：`report.md`
- 方法摘要：`docs/METHOD_SUMMARY.md`
- 数据/评测边界：`docs/DATA_AND_EVAL_BOUNDARY.md`
- 复现说明：`docs/REPRODUCIBILITY.md`
- AI 工具说明：`docs/AI_TOOL_USAGE.md`
- 实验索引：`experiments/EXPERIMENT_INDEX.csv`
- 负结果：`experiments/NEGATIVE_RESULTS.md`

完整包含九个 A 推理 checkpoint、模型代码和融合配置，可精确重建研究结果；RTX 4090 实测约 34 秒且 SHA-256 逐文件一致。由于外部约 17.3 GB 预训练语料的逐项来源/许可清单未完整保留，这些 checkpoint 明确标为 **research-only**，不作商业可用声明。精简包不携带模型，保留可商用依赖的 CPU 规则基线和全部正式结果；边界见 `models/THIRD_PARTY.md`。
