# 复现说明

已测试环境：Windows 11、Python 3.11、OpenCV contrib 4.13、NumPy 2.2、SciPy 1.16；GPU 实验使用 RTX 4090 24 GB。

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements-core.txt
.venv\Scripts\python run_all.py --data-root ..\2026.07.13
```

默认命令只评测包内正式结果并重写 `outputs/summary/official_metrics.json`。完整重算 B/C：

```powershell
.venv\Scripts\python run_all.py --data-root ..\2026.07.13 --regenerate-b --regenerate-c
```

A 研究结果是外部几何预训练 + 三折 LOFO 的融合输出。完整包提供九个分支 checkpoint、网络与确定性融合代码；先安装 `requirements-gpu.txt`，再执行：

```powershell
.venv\Scripts\python run_all.py --data-root ..\2026.07.13 --regenerate-a-from-checkpoints
```

结果写入 `outputs/reconstructed/task_a`，不会覆盖正式目录。RTX 4090 实测约 34 秒。外部约 17.3 GB 原始预训练语料的逐项来源/许可记录不完整，因此 checkpoint 仅标为 research-only；`--regenerate-a-fast` 是不依赖该外部语料的 CPU 规则基线。A 预训练 5,000 步约 9 分钟，每个 700 步 LOFO 折约 28 秒；这些为一次实测，不是硬件无关承诺。

OpenCV 数值行为会影响亚像素流与最终栅格，故固定 4.13。验收脚本失败即退出，不以“文件存在”代替指标对账。
