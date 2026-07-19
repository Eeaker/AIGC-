# 面向赛璐璐动画生产的可审计 A/B/C 算法提交


本仓库完成 KTK_04_246B 的描原、中割、上色三题，共 **18 张正式 TGA**，并提供 KTK_05_140 的 A/B 双层进阶结果。项目按 **正式结果 / 算法基线 / 诊断上限 / 负结果** 分层，重点保证数据边界、指标和代码能够互相对账。

## 核心结果

| 任务 | 正式配置 | 主结果 | 关键约束 |
|---|---|---:|---|
| A 描原 | 严格四色，3 折 LOFO | 平均 F1@2px **0.9886** | 四色符合率 100%；泄漏率 0.34% |
| B 中割 | 6 张 blind 输出 | A2--A5 平均 **0.9322** | A7/A8 为 0.5109/0.5724；不加载成品监督 cleanup |
| C 上色 | 153 个关键视图区域标签 | 精度/覆盖 **99.56%/95.63%** | 线稿像素变化 0 |
| C 自动基线 | 0 标签 | 精度/覆盖 **99.97%/59.25%** | 高精度、低覆盖 |

## 推荐阅读顺序

1. **技术报告：** `report.md`
2. **数据与评测边界：** `docs/DATA_AND_EVAL_BOUNDARY.md`
3. **方法摘要：** `docs/METHOD_SUMMARY.md`
4. **复现说明：** `docs/REPRODUCIBILITY.md`
5. **负结果：** `experiments/NEGATIVE_RESULTS.md`
6. **AI 工具披露：** `docs/AI_TOOL_USAGE.md`

## 快速验收

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements-core.txt
.venv\Scripts\python run_all.py --data-root ..\2026.07.13
```

默认命令**不覆盖**包内预生成正式结果，而是从原始素材重新计算指标，并检查：

- 18 张正式 TGA 的尺寸与可读性；
- A 的严格四色、B 的生产线色规范；
- A/B 的 exact、1 px、2 px 双向 F1 与逐色 F1；
- A 的连通域闭合泄漏；
- C 的区域精度/覆盖以及输入非白线逐像素零变化；
- C 正式配置恰好使用 153 个干净区域标签。

验收同时重写 `outputs/summary/official_metrics.json` 和 `official_metrics.csv`，两者由同一数据源导出，避免手工表格漂移。

## 结果目录

```text
outputs/
├─ official/
│  ├─ task_a/                 # 3 张严格四色描原
│  ├─ task_b/                 # 6 张 blind 中割
│  └─ task_c_assisted/        # 9 张 153-label 正式上色
├─ algorithm_only/
│  ├─ task_a_fast_baseline/   # 无研究 checkpoint 的 CPU 规则基线
│  └─ task_c_automatic/       # 0-label 自动上色基线
├─ diagnostic/
│  └─ task_b_shot_adapted/    # 使用 A2--A5 成品监督的诊断版本，不计正式成绩
└─ supplementary/
   ├─ task_a_production_5color/
   └─ ktk05/
```

## 复现层级

### 1. CPU 独立审计

执行上面的默认命令即可。它只评测正式结果，不依赖 GPU 或研究 checkpoint。

### 2. CPU 重建 B/C

```powershell
.venv\Scripts\python run_all.py --data-root ..\2026.07.13 --regenerate-b --regenerate-c
```

### 3. GPU 重建 A 研究结果

```powershell
.venv\Scripts\pip install -r requirements-gpu.txt
.venv\Scripts\python run_all.py --data-root ..\2026.07.13 --regenerate-a-from-checkpoints
```

九个 A checkpoint 可确定性重建正式/补充结果；RTX 4090 单次实测约 34 秒。由于约 17.3 GB 外部预训练语料的逐项来源和许可记录未完整保留，这些 checkpoint 明确标为 **research-only**，不作商业可用声明。商业部署应使用 CPU 规则基线，或在许可清晰的数据上重新训练；详见 `models/THIRD_PARTY.md`。

## 评测诚信说明

- **A：** 每个 LOFO 折的当前留出帧参考只用于最终评分；这是同镜头帧间留出，不宣称跨镜头泛化。
- **B：** 正式版本只读取 A1/A6/A9、律表与规则参数，不加载由 A2--A5 成品训练的 cleanup 权重；A7/A8 参考仅用于评分。
- **C：** 正式 153 个标签来自 A1/A5/A7/A9 的线稿、设定图和色卡；225 标签及 oracle/reviewed 配置包含答案后审计，只作为诊断。
- **KTK_05：** C 的 A 层使用 A1/A3/A5 成品作为明确训练关键帧，与 KTK_04 正式成绩分开报告。

## 依赖与环境

- CPU 审计：Python 3.11、OpenCV contrib 4.13、NumPy 2.2、SciPy 1.16
- GPU 实验：RTX 4090 24 GB、CUDA 版 PyTorch
- OpenCV 版本固定，以降低亚像素光流和栅格化差异

## 许可证与第三方材料

代码依赖、研究 checkpoint、外部论文实现和训练数据需要分别审核；“代码开源”不自动意味着数据或权重可商用。仓库不对缺失的许可信息作推测，完整边界见 `models/THIRD_PARTY.md`。
