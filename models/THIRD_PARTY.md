# 模型与许可证说明

`b_cleanup.pth` 是本题自训的 B 题残差 U-Net，只使用 A0002-A0005 成品监督；A0007/A0008 未参与训练。它不再进入正式 blind 推理，只用于 `outputs/diagnostic/task_b_shot_adapted`，阈值 0.92、输入线门控 12 px。

研究阶段曾评估 SIGGRAPH 2018 *Real-Time Data-Driven Interactive Rough Sketch Inking* 的线宽归一化权重。其许可为 CC BY-NC-SA 4.0，不适合公司商业生产，因此最终提交构建不包含该权重，`src/task_a.py` 也默认使用规则基线。若本地研究目录仍存在 `line_thinning_siggraph2018.pth`，它不属于最终可交付物。

九个 `task_a_lofo` checkpoint 使用过约 17.3 GB 外部几何预训练语料，但本次研究记录没有保存到可逐项核验的数据集名称、来源 URL、许可证和衍生权重条款。因此这些 checkpoint 仅作 **research-only** 复现实验，不声明可商用，也不应进入公司生产。可商业部署口径应使用不依赖这些权重的 `--regenerate-a-fast` CPU 规则基线，并在取得具有明确商业许可的 rough-clean 数据后重新训练。

外部论文代码、权重和数据集的许可证必须分别审核；“代码开源”不自动意味着训练数据或预训练权重可商用。此次缺失的信息不作推测或补写。
