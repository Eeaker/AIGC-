# KTK_05_140 三题进阶结果

本目录把题面进阶项完整扩展到 A/B 双层。所有正式生成脚本先写结果、再读取成品评分；数据使用边界在下文逐项披露。C 同时报告零标注和单关键帧两种口径，不把监督结果冒充跨镜头泛化。

## A：A/B 双层描原

- A 层正式补充输出位于 `outputs/supplementary/ktk05/task_a_A_strict_4color`；绿色语义线的生产版指标来自历史完整实验，精简提交包保留严格四色结果与指标。
- 五色版 A0001/A0005 的 2 px 联合线 F1 为 **0.9382/0.9403**，绿色逐色 F1 为 **0.9936/0.9913**，颜色符合率 100%。绿色来自当前粗稿的高饱和源笔触，不读取成品恢复。
- B 层是稀疏嘴部修正层，不适用整幅清稿阈值。固定归一化角色 ROI 清除场记文字，保留原始线宽；B0001/B0003 的角色 ROI F1 为 **0.8563/1.0000**。全图分数更低是因为成品参考仍含右侧场记，而题目要求清除辅助内容。

## B：A/B 双层中割

- A 层 A0002–A0004 使用冻结的距离场双向流、沿笔画二阶正则和拓扑保持栅格化，F1 为 **0.9257/0.8978/0.9206**。
- B 层是“开放短弧→闭合嘴型”的可见性事件。普通光流最高全图 F1 约 0.44；正式结果采用目标关键帧拓扑、线性宽度/锚点和 60% 纵向展开的 visibility-first 中点，避免把两套轮廓叠加。全图/角色 ROI F1 为 **0.5307/0.6283**，指标写入 `task_b_metrics.json`。
- 该层仍只有一对关键帧，不能证明统计泛化；它展示的是对拓扑出生事件的显式工程处理，不是训练出的通用模型。

方法方向依据同行评审的一手资料：[AnimeInbet（ICCV 2023）](https://openaccess.thecvf.com/content/ICCV2023/html/Siyao_Deep_Geometrized_Cartoon_Line_Inbetweening_ICCV_2023_paper.html) 的端点/可见性建模、[MIBA（ACCV 2024）](https://openaccess.thecvf.com/content/ACCV2024/html/Chen_Match-free_Inbetweening_Assistant_MIBA_A_Practical_Animation_Tool_without_User_ACCV_2024_paper.html) 的栅格生产辅助思路，以及 [TPS-Inbetween](https://arxiv.org/abs/2408.09131) 对大形变与 Chamfer 局限的讨论。实现没有复制随机仓库代码。

## C：A/B 双层上色与阴影/高光层级

有三档可审计结果：

1. `task_c_A_setting_only*`：只使用官方角色设定图右上角同视角彩色头像。先做仿射归一，再以线稿距离场非刚性配准，最后按封闭区域投票；这是零标注泛化基线。
2. `annotations/ktk05_task_c/bonus_ktk05_c_A000{1,3,5}_regions.json`：把 A0001/A0003/A0005 成品明确作为训练关键帧，导出纯度至少 90% 的区域标签。A0002/A0004 的成品只用于最终评测，不参与生成；A2 依次由 A1/A3、A4 依次由 A3/A5 做区域包含传播，后者只补空缺、不覆盖主关键帧。
3. `task_c_A/task_c_B` 是视觉成品版，`*_strict_lines` 是题面严格保线版。后者所有输入非白线变化均为 **0**。

A 层采用 precision-first 拒绝策略，最新逐帧“已填区域精度/精确颜色覆盖”为 **99.15%/88.61%、98.91%/85.36%、100.00%/69.71%、98.73%/58.04%、99.57%/63.64%**；五帧错误色像素分别降至 **17,352、20,636、45、17,074、6,220**，输入线像素变化全部为 **0**。白区不是漏过验收的“白点”，而是纯度或区域包含不足时的显式低置信拒绝。阈值消融表明，强行补满虽可把部分关键帧覆盖推至约 94%，却会新增约 5.6 万至 12.9 万错色像素，不符合题面“涂错代价高于留白”的主评分口径。B0002 精度/覆盖约 **100.00%/100.00%**（仅 2 个未填目标像素）；B0003 为 **93.87%/93.87%**，严格版仍逐像素保线。

此前大块金色头发等明显错误来自一个已修复的数据绑定问题：凹区域的几何质心可能落入相邻连通域，旧加载器据此重新取 label，导致颜色绑定到错误大区。现在始终使用标注文件保存的连通域 label，坐标仅用于审计。另一个修复是取消 A1 对 A4 的硬编码传播，改用曝光顺序和 70% 包含阈值。

阴影、主体色和高光从官方设定色阶转移，并在 `task_c_metrics.json` 按亮度层级分别报告 precision/coverage。区域匹配依据 [Learning Inclusion Matching for Animation Paint Bucket Colorization（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Dai_Learning_Inclusion_Matching_for_Animation_Paint_Bucket_Colorization_CVPR_2024_paper.html) 的包含关系思想；本实现是独立的确定性区域投票/传播。

## 验收与复现边界

KTK_05 提供预生成结果、逐项指标、重建脚本和独立校验脚本。先重建 C，再验收全部进阶结果：

```powershell
python tools\rebuild_ktk05_task_c.py --data-root ..\2026.07.13
python tools\validate_supplementary.py --data-root ..\2026.07.13
```

本轮不需要 GPU：瓶颈是拓扑、区域对应和全分辨率评测，不是算力。若继续做 B 的通用模型，才需要多镜头 key–inbetween–key 数据、端点/交点与可见性标注，以及 GPU 训练；只在当前镜头加算力会继续过拟合。
