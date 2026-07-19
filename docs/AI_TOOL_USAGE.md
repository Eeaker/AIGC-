# AI 工具与人工责任披露

## 使用范围

ChatGPT/Codex 参与了以下工作：

- 题面拆解与验收条件整理；
- Python/OpenCV 代码实现和重构；
- 像素级指标审计、结果对账与异常定位；
- 动画线稿中割/区域上色相关论文和作者实现调研；
- AutoDL 实验编排、日志归档与负结果整理；
- README、方法说明和研究报告的结构化写作与排版。

未使用付费图像生成 API 直接生成正式输出帧。

## 人工负责事项

以下决策由人工复核并承担责任：

- 正式、baseline、diagnostic、supplementary 的数据边界；
- B 正式版本停用成品监督 cleanup 权重；
- C 关键视图区域语义与低置信拒绝策略；
- A 外部数据/权重许可证风险披露；
- 指标选择、淘汰阈值、失败结论和最终提交内容。

AI 生成的代码和文字均不能替代对输入数据、评测脚本和许可证的审查。

## 资源估计

项目经历多轮长上下文对话与实验，界面未提供可导出的精确 token 账单。按上下文规模保守估计总用量约 **20--40 万 token**，主要消耗依次为：

1. B 大位移、遮挡和拓扑失败的反证实验；
2. A 三折 LOFO、融合与像素审计；
3. C 区域标签、残差白区和精度/覆盖对照；
4. 文档口径统一与复现检查。

使用一张 AutoDL RTX 4090 完成 A 预训练/LOFO 以及 B 的部分诊断消融。GPU 金额和完整累计时长未可靠记录，因此不编造精确成本；已知单次 A 5,000 步预训练约 9 分钟，A checkpoint 重建约 34 秒。

## 可审计材料

- 数据边界：`docs/DATA_AND_EVAL_BOUNDARY.md`
- 实验索引：`experiments/EXPERIMENT_INDEX.csv`
- 负结果：`experiments/NEGATIVE_RESULTS.md`
- 第三方与许可证：`models/THIRD_PARTY.md`
- 独立验收：`tools/validate_submission.py`
