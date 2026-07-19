# 灵图 B 题：端点/交点、拓扑与 Visibility 标注包 v2.2

这是一套面向 `KTK_04_246B` B 题的**保守型标注包**。它只读取题目允许作为输入的三张关键帧：`A0001 / A0006 / A0009`，没有读取 `A0002–A0005 / A0007–A0008` 中间帧参考答案。

## 当前可直接使用的内容

- 三帧按黑/蓝/红/绿分色构建的骨架图：节点、笔画边、连通分量、方向直方图、重叠组。
- 端点/交点候选对应：TPS 几何约束、SIFT 局部描述、双向互检、拓扑一致性。
- 61 个语义控制点：A0001→A0006 共 28 个，A0006→A0009 共 33 个。
- 24 条主要语义笔画及 edge visibility：兜帽轮廓、中心缝、开口、眼睛、脸轮廓、主要发丝、鼻部蓝线、领口。
- 双向 visibility / birth / death / merge / occlusion 复核队列。
- 可编辑的 `review_decisions.json`，以及支持 `accept / reject / correct_target` 的 Streamlit 复核工具。

## 当前统计

| 项目 | A0001 | A0006 | A0009 |
|---|---:|---:|---:|
| 节点 | 256 | 273 | 240 |
| 笔画边 | 233 | 250 | 229 |
| 端点 | 155 | 166 | 139 |
| 交点 | 101 | 107 | 101 |

| 帧对 | 候选 | 已接受 | 待复核 | 自动拒绝 | 语义控制点 |
|---|---:|---:|---:|---:|---:|
| A0001→A0006 | 216 | 65 | 124 | 27 | 28 |
| A0006→A0009 | 163 | 18 | 115 | 30 | 33 |

A0006→A0009 存在明显转头和拓扑变化，因此这里采取**低覆盖、高精度**策略：没有为了凑数量而强行接受错误匹配。18 对已接受节点中包含 15 对语义锚点派生的人工语义复核链接；其余候选保留在队列中。

## 最重要的文件

- `annotations/reviewed_labels.json`：当前保守版最终标签。
- `annotations/review_decisions.json`：全部复核决定，可继续编辑。
- `annotations/stroke_graph.json`：完整节点和 edge 图。
- `annotations/manual_landmarks.json`：61 个语义 TPS 控制点。
- `annotations/manual_node_links.json`：语义控制点能够可靠吸附到节点时生成的节点对应。
- `annotations/candidate_matches.json`：全部候选及几何/描述子指标。
- `annotations/visibility_events.json`：双向消失、出生、遮挡和合并候选。
- `annotations/edge_visibility.json`：主要语义笔画 visibility。
- `annotations/validation_report.json`：结构一致性校验报告。

CSV 版本：

- `node_match_review_queue.csv`
- `visibility_review_queue.csv`
- `edge_visibility_review_queue.csv`
- `review_index.csv`

## 启动复核界面

```bash
cd lingtu_B_topology_visibility_v2
pip install -r requirements.txt
streamlit run review/review_app.py
```

节点页支持：

- 接受当前对应；
- 拒绝错误对应；
- 从同色目标节点中选择正确目标；
- 保存备注。

visibility 页支持补充 `occluded / absent / merged / split / birth / out_of_frame / uncertain`。语义笔画页可复核 edge visibility。

完成复核后执行：

```bash
python src/apply_review.py --package-dir .
```

输出：`annotations/reviewed_labels_user.json`。

## 重建标注包

以下命令只需要三张输入关键帧：

```bash
python src/build_graph_frame.py --input-dir <中割关键帧目录> --output-dir annotations --frame A0001
python src/build_graph_frame.py --input-dir <中割关键帧目录> --output-dir annotations --frame A0006
python src/build_graph_frame.py --input-dir <中割关键帧目录> --output-dir annotations --frame A0009
python src/assemble_and_match.py --input-dir <中割关键帧目录> --output-dir .
python src/enhance_semantic_review.py --package-dir .
python src/validate_package.py --package-dir .
```

## 使用边界

这套数据可以用于：

- 当前镜头 TPS / 局部仿射 / 图匹配的小实验；
- 端点和交点损失的稀疏约束；
- 主要笔画的 edge visibility；
- 候选重排、拒绝策略和失败案例分析。

它**不应被描述为通用 visibility 网络的人工真值数据集**。只有一个镜头、三个关键帧，剩余待复核事件仍很多。报告里应明确写成“自动预标注 + AI 语义复核 + 最终提交者复核”。
