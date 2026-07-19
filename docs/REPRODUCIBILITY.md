# 复现说明

## 1. 已验证环境

- Windows 11 / Linux（CPU 审计）
- Python 3.11
- OpenCV contrib 4.13
- NumPy 2.2
- SciPy 1.16
- GPU 研究实验：RTX 4090 24 GB，CUDA 版 PyTorch

OpenCV 的亚像素光流和距离变换数值会影响最终栅格，因此固定主版本；其他平台的细微像素差异应以独立验收脚本为准，而非人工比对文件哈希。

## 2. 独立审计（推荐入口）

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements-core.txt
.venv\Scripts\python run_all.py --data-root ..\2026.07.13
```

默认命令不覆盖预生成正式结果。它重新读取原始数据并检查 18 张 TGA 的尺寸、色规、A/B 线集指标、A 闭合泄漏、C 区域指标、C 线稿零变化和 153 个正式区域标签；随后从同一个内存对象导出：

- `outputs/summary/official_metrics.json`
- `outputs/summary/official_metrics.csv`
- 各正式任务目录中的 `metrics.json`

任一硬约束失败时脚本非零退出。

## 3. 重建 B/C

```powershell
.venv\Scripts\python run_all.py --data-root ..\2026.07.13 --regenerate-b --regenerate-c
```

- B 写入 `outputs/official/task_b`，默认 `use_shot_cleanup=False`；
- C 同时重建 153-label 正式版和 0-label 自动基线；
- 运行后自动进入独立验收。

## 4. 重建 A 研究结果

```powershell
.venv\Scripts\pip install -r requirements-gpu.txt
.venv\Scripts\python run_all.py --data-root ..\2026.07.13 --regenerate-a-from-checkpoints
```

重建结果写入 `outputs/reconstructed/task_a`，不覆盖正式目录。完整包包含九个 LOFO 分支 checkpoint、网络结构、融合参数和 SHA-256 清单；RTX 4090 单次实测约 34 秒。A 预训练 5,000 步约 9 分钟，每个 700 步 LOFO 折约 28 秒，这些仅为一次硬件实测。

外部约 17.3 GB 预训练语料的逐项名称、来源 URL、许可证和衍生权重条款未完整保留，因此 checkpoint 仅作 research-only 复现。无需这些权重的 CPU 基线为：

```powershell
.venv\Scripts\python run_all.py --data-root ..\2026.07.13 --regenerate-a-fast
```

## 5. 进阶 KTK_05

```powershell
.venv\Scripts\python tools\rebuild_ktk05_task_c.py --data-root ..\2026.07.13
.venv\Scripts\python tools\validate_supplementary.py --data-root ..\2026.07.13
```

KTK_05 的数据边界与主镜头不同，尤其 C-A 明确使用 A1/A3/A5 成品作为训练关键帧；详见 `docs/ADVANCED_KTK05.md`。

## 6. 可复现性限制

- A 的正式研究结果可重建，但外部数据许可链不完整；
- B/C 为确定性 CPU 规则/传播系统，仍可能受 OpenCV 版本和线程实现影响；
- 本轮未可靠记录 C 人工标注分钟数、完整 GPU 账单与精确 token 账单，因此不以估算值作为性能结论；
- 当前数据规模只支持当前镜头的工程验证，不支持统计显著性或跨镜头泛化声明。
