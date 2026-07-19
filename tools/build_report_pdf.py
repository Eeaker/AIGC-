"""Build the required four-page PDF report with deterministic layout."""
from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)


def fit_image(path: Path, max_w: float, max_h: float) -> Image:
    image = Image(str(path))
    ratio = min(max_w / image.imageWidth, max_h / image.imageHeight)
    image.drawWidth = image.imageWidth * ratio
    image.drawHeight = image.imageHeight * ratio
    return image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    root = args.root.resolve()
    out = root / "report.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)

    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    bold_path = Path(r"C:\Windows\Fonts\msyhbd.ttc")
    pdfmetrics.registerFont(TTFont("CJK", str(font_path), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("CJKB", str(bold_path), subfontIndex=0))
    accent = colors.HexColor("#245C73")
    pale = colors.HexColor("#EAF3F6")
    ink = colors.HexColor("#20282C")
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", fontName="CJKB", fontSize=21, leading=28,
                           textColor=ink, alignment=TA_CENTER, spaceAfter=5*mm)
    subtitle = ParagraphStyle("subtitle", fontName="CJK", fontSize=9.5, leading=15,
                              textColor=colors.HexColor("#526269"), alignment=TA_CENTER)
    h1 = ParagraphStyle("h1", fontName="CJKB", fontSize=15, leading=21,
                        textColor=accent, spaceBefore=2*mm, spaceAfter=2.5*mm)
    h2 = ParagraphStyle("h2", fontName="CJKB", fontSize=11, leading=16,
                        textColor=ink, spaceBefore=2*mm, spaceAfter=1*mm)
    body = ParagraphStyle("body", fontName="CJK", fontSize=9.2, leading=14.2,
                          textColor=ink, alignment=TA_LEFT, spaceAfter=2.2*mm)
    small = ParagraphStyle("small", parent=body, fontSize=7.8, leading=11.5,
                           textColor=colors.HexColor("#46565D"), spaceAfter=1.2*mm)
    callout = ParagraphStyle("callout", parent=body, fontName="CJKB", fontSize=10,
                             leading=15, textColor=accent, leftIndent=4*mm,
                             rightIndent=4*mm, borderColor=accent, borderWidth=.7,
                             borderPadding=3*mm, backColor=pale, spaceAfter=4*mm)

    def p(text: str, style=body) -> Paragraph:
        return Paragraph(text, style)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#C9D7DC")); canvas.setLineWidth(.5)
        canvas.line(20*mm, 14*mm, 190*mm, 14*mm)
        canvas.setFont("CJK", 7.5); canvas.setFillColor(colors.HexColor("#64767D"))
        canvas.drawString(20*mm, 9*mm, "灵图算法笔试 | A/B/C")
        canvas.drawRightString(190*mm, 9*mm, f"{doc.page} / 4")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=16*mm, bottomMargin=18*mm,
                            title="动画智能制作算法笔试报告")
    story = []

    # Page 1 - outcome and protocol
    story += [Spacer(1, 5*mm), p("动画智能制作算法笔试报告", title),
              p("KTK_04_246B 正式 A/B/C + KTK_05_140 三项进阶 | v17", subtitle),
              Spacer(1, 5*mm),
              p("提交结论", h1),
              p("完成 18 张正式 TGA（A 3、B 6、C 9），并把正式结果、0 标签算法基线、人机协同结果和参考答案诊断严格分层。A 按题面提交四色版；B 如实保留大转头失败；C 正式版不使用中间成品反馈。", callout)]
    table_data = [
        [p("题目", small), p("正式自评", small), p("约束/判断", small)],
        [p("A 描原", small), p("F1@2px 0.9886", small), p("四色符合率 100%；闭合泄漏 0.34%", small)],
        [p("B A2-A5", small), p("平均 0.9322", small), p("小运动段较强", small)],
        [p("B A7/A8", small), p("0.5109 / 0.5724", small), p("blind；发丝/遮挡拓扑仍失败", small)],
        [p("C assisted", small), p("精度 99.56%<br/>覆盖 95.63%", small), p("153 干净区域标签；原线变化 0", small)],
        [p("C automatic", small), p("精度 99.97%<br/>覆盖 59.25%", small), p("0 标签；高精度低覆盖", small)],
    ]
    t = Table(table_data, colWidths=[29*mm, 43*mm, 96*mm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), accent), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,-1), "CJK"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#B8C8CE")),
        ("BACKGROUND", (0,1), (-1,-1), colors.HexColor("#F7FAFB")),
        ("LEFTPADDING", (0,0), (-1,-1), 6), ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story += [t, Spacer(1, 4*mm), p("评测协议", h1),
              p("A/B 将所有非白像素视为线集，报告 exact、1 px、2 px 双向容差 F1，并逐色复算。C 在输入线稿四连通白区内做精确 RGB 比色，精度与覆盖分列；验收脚本逐帧断言 C 的输入非白线不变。A/B 的训练/生成数据和仅评测参考在 DATA_AND_EVAL_BOUNDARY.md 中逐项披露。"),
              p("正式结果的 18 张 TGA、调色盘、语义点、人工成本占位和统一 JSON/CSV 指标均随包提交。", small),
              PageBreak()]

    # Page 2 - A and B
    story += [p("A 描原：字面四色优先，绿色生产线另附", h1),
              p("几何分支采用外部中心线/截断距离场预训练，再对三张成对数据做三折 LOFO。九个 checkpoint 与确定性融合代码已随包，RTX 4090 实测约 34 秒可逐字节重建正式结果。题面要求白/黑/蓝/红四色，因此正式目录严格四色；参考成品实际存在绿色语义线，五色版单独放在 supplementary，绿色只由当前粗稿高饱和笔触恢复。"),
              fit_image(root / "outputs" / "comparisons" / "task_a.png", 174*mm, 41*mm),
              Spacer(1, 3*mm), p("B 中割：拓扑保持栅格流", h1),
              p("端点线稿转截断距离场，以双向 DIS 估计粗运动；沿端点/交点之间的有序笔画正则位移，再直接变形八连通像素图的边。A6-A9 用 A0009 单拓扑反向场，红/绿线独立估计。正式 blind 输出不加载 A2-A5 成品监督的 cleanup 权重；该版本仅作 shot-adapted 诊断。"),
              fit_image(root / "outputs" / "comparisons" / "task_b.png", 174*mm, 82*mm),
              p("上排 A3 展示小运动成功；下排 A7 明确展示大转头失败，不用成功帧掩盖核心问题。", small),
              PageBreak()]

    # Page 3 - B evidence and C
    story += [p("B 更新标注包与云端反证", h1),
              p("topology_visibility_v2 是结构合法的保守预标注：A6-A9 有 33 个人工语义点和 24 条边可见性记录，但 871 个 visibility 事件仍待复核，不能称完整人工 GT。全局 TPS 的 A7/A8 最佳约 0.209/0.240，部件图直接形变仅 0.165/0.230；局部流残差也仅 +0.00010。"),
              p("预算化 Assisted-10 从 0.55303 到 0.55390，未过预定 +0.002 多指标门槛；30 点未胜出。RAFT/MIBA 近似为 0.201/0.346；学习清理虽为 0.544/0.593，却把线量放大到 2.19/2.33 倍。瓶颈是笔画出生/消失、遮挡和对应，不是缺 GPU。", callout),
              p("C 上色：信息边界清晰的人机协同", h1),
              p("从设定图精确统计调色盘，分解四连通白区。正式 153 个标签描述四个关键视图 A1/A5/A7/A9，颜色只来自线稿、设定图和色卡，不使用 A2-A8 中间成品误差修补；区域 mask 通过距离场运动和 inclusion 投票传播，冲突留白。输出从输入副本开始，只修改原白像素。"),
              fit_image(root / "outputs" / "comparisons" / "task_c.png", 174*mm, 55*mm),
              Spacer(1, 2*mm)]
    cdata = [
        [p("配置", small), p("标签", small), p("面积精度", small), p("正确覆盖", small), p("用途", small)],
        [p("automatic", small), p("0", small), p("99.97%", small), p("59.25%", small), p("算法基线", small)],
        [p("assisted", small), p("153", small), p("99.56%", small), p("95.63%", small), p("正式", small)],
        [p("dense-assisted", small), p("225", small), p("99.33%", small), p("99.30%", small), p("诊断（非盲标）", small)],
        [p("reviewed/oracle", small), p("参考审计", small), p("约 99.95%", small), p("约 99.94%", small), p("仅诊断", small)],
    ]
    ct = Table(cdata, colWidths=[37*mm, 24*mm, 34*mm, 34*mm, 39*mm], repeatRows=1)
    ct.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),accent),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                            ("GRID",(0,0),(-1,-1),.35,colors.HexColor("#B8C8CE")),
                            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("ALIGN",(1,1),(-2,-1),"CENTER"),
                            ("BACKGROUND",(0,1),(-1,-1),colors.HexColor("#F7FAFB")),
                            ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    story += [ct, PageBreak()]

    # Page 4 - advanced, improvements, sources and AI disclosure
    story += [p("进阶、失败边界与下一步", h1),
              p("KTK_05 三项进阶均已提交 A/B 双层。A 层严格四色两帧 F1@2px 约 0.939；B 中割 A 层三帧为 0.926/0.898/0.921，B 层 visibility-first 中点全图/角色 ROI 为 0.531/0.628。C 修复凹区域 label 错绑和错误关键帧传播后，A1-A5 已填区域精度为 99.15%/98.91%/100.00%/98.73%/99.57%，原线变化为 0；低置信区域按题面错涂重于留白的原则拒绝填色。"),
              p("1. A：在授权的多镜头四色 rough-clean 对上按镜头划分训练/验证/测试。<br/>2. B：完成 visibility 人工复核，学习笔画图对应，并以 F1、线量、组件数、P95 距离联合选模。<br/>3. C：建立区域邻接/包含图，只在双向一致时自动填色，并记录真实人工分钟数。"),
              p("AI 工具与算力", h2),
              p("使用 ChatGPT/Codex 辅助题面拆解、实现、像素审计、一手论文/作者实现调研、AutoDL 实验与报告。无法导出精确 token 账单，按上下文规模估计约 20-40 万 token。RTX 4090 用于 A 预训练/LOFO 与 B 消融。A 外部预训练语料的逐项许可记录不完整，因此 checkpoint 明确为 research-only；商业口径保留 CPU 规则基线。"),
              p("主要一手资料", h2),
              p("Deep Sketch Vectorization (SIGGRAPH 2024): <link href='https://cragl.cs.gmu.edu/sketchvector/'>cragl.cs.gmu.edu/sketchvector</link><br/>"
                "AnimeInbet (ICCV 2023): <link href='https://openaccess.thecvf.com/content/ICCV2023/html/Siyao_Deep_Geometrized_Cartoon_Line_Inbetweening_ICCV_2023_paper.html'>CVF paper page</link><br/>"
                "MIBA (ACCV 2024): <link href='https://openaccess.thecvf.com/content/ACCV2024/html/Chen_Match-free_Inbetweening_Assistant_MIBA_A_Practical_Animation_Tool_without_User_ACCV_2024_paper.html'>CVF paper page</link><br/>"
                "TPS-Inbetween: <link href='https://arxiv.org/abs/2408.09131'>arXiv 2408.09131</link>; JoSTC: <link href='https://markmohr.github.io/JoSTC/'>author project page</link><br/>"
                "Learning Inclusion Matching (CVPR 2024): <link href='https://openaccess.thecvf.com/content/CVPR2024/html/Dai_Learning_Inclusion_Matching_for_Animation_Paint_Bucket_Colorization_CVPR_2024_paper.html'>CVF paper page</link>", small),
              Spacer(1, 4*mm),
              p("结论：当前 A 达到强同镜头工程结果，C 达到高质量可审计人机协同；B 小运动段可用，但大转头离生产级仍有明显距离。新标注方向正确，现阶段继续堆当前镜头训练只会强化过拟合。", callout)]

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(out)


if __name__ == "__main__":
    main()
