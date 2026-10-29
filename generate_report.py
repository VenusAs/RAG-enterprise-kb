"""
评估报告自动生成器（可视化增强版 · MRR 独立纵轴 0-1 + 美化）
读取 evaluation_result.json，输出：
1. Markdown 报告（带时间戳）
2. 可视化图表（PNG，带时间戳，MRR 纵轴固定 0-1）
3. 报告内嵌图表引用
"""

import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# ============================================================
# 中文字体配置
# ============================================================
def setup_chinese_font():
    candidates = [
        "SimHei", "Microsoft YaHei", "PingFang SC",
        "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Arial Unicode MS"
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None


FONT_NAME = setup_chinese_font()


# ============================================================
# 时间戳 & 目录
# ============================================================
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
REPORT_DIR = f"report_{TIMESTAMP}"
CHART_DIR = os.path.join(REPORT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)


# ============================================================
# 工具函数
# ============================================================
def load_results(path="evaluation_result.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_percent(v):
    if v is None:
        return 0.0
    s = str(v).replace("%", "").replace("$", "").replace("s", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def generate_table(results, columns):
    if not results:
        return "（无数据）\n"
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = [
        "| " + " | ".join(str(r.get(c, "-")) for c in columns) + " |"
        for r in results
    ]
    return "\n".join([header, separator] + rows) + "\n"


def find_best(results, metric, higher_is_better=True):
    if not results:
        return None
    key = max if higher_is_better else min
    return key(results, key=lambda r: parse_percent(r.get(metric, "0%")))


# ============================================================
# 图表绘制
# ============================================================
def plot_grouped_bar(results, metrics, title, filename,
                     ylabel="百分比 (%)", ylim=None):
    """分组柱状图（美化版）"""
    if not results:
        return None

    strategies = [r["策略"] for r in results]
    n_metrics = len(metrics)
    n_strategies = len(strategies)
    x = range(n_strategies)
    width = 0.8 / n_metrics

    palette = ["#5B8FF9", "#61DDAA", "#F6BD16", "#E8684A", "#9270CA"]

    fig, ax = plt.subplots(figsize=(max(9, n_strategies * 1.8), 5.2))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")

    for i, metric in enumerate(metrics):
        values = [parse_percent(r.get(metric, 0)) for r in results]
        offset = (i - (n_metrics - 1) / 2) * width
        bars = ax.bar(
            [xi + offset for xi in x],
            values,
            width=width * 0.92,
            label=metric,
            color=palette[i % len(palette)],
            edgecolor="white",
            linewidth=0.8,
            zorder=3
        )
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max(max(values),1),
                f"{v:.2f}" if v < 10 else f"{v:.1f}",
                ha="center", va="bottom", fontsize=8.5, color="#333333"
            )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=14, color="#222222")
    ax.set_ylabel(ylabel, fontsize=11, color="#444444")
    ax.set_xticks(list(x))
    ax.set_xticklabels(strategies, rotation=12, ha="right", fontsize=9.5, color="#333333")
    ax.tick_params(axis="y", labelsize=9.5, colors="#444444")

    if ylim:
        ax.set_ylim(*ylim)

    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#CCCCCC")

    ax.legend(fontsize=9.5, frameon=False, loc="upper right")

    plt.tight_layout()
    path = os.path.join(CHART_DIR, filename)
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_single_metric(results, metric, title, filename,
                       higher_is_better=True, color="#5B8FF9",
                       xlim=None):
    """单指标横向柱状图（美化版）"""
    if not results:
        return None

    strategies = [r["策略"] for r in results]
    values = [parse_percent(r.get(metric, 0)) for r in results]

    pairs = sorted(zip(strategies, values),
                   key=lambda p: p[1], reverse=higher_is_better)
    strategies, values = zip(*pairs)

    # 用于计算文字偏移的安全基准值
    base = max(max(values), 1e-6)

    fig, ax = plt.subplots(figsize=(9, max(3.2, len(strategies) * 0.7)))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")

    bars = ax.barh(strategies, values, color=color,
                   edgecolor="white", linewidth=0.8, zorder=3, height=0.6)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_width() + base * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f" {v:.4f}" if v < 1 else f" {v:.2f}",
            va="center", fontsize=9.5, color="#333333"
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=14, color="#222222")
    ax.set_xlabel(metric, fontsize=11, color="#444444")
    ax.tick_params(axis="x", labelsize=9.5, colors="#444444")
    ax.tick_params(axis="y", labelsize=10, colors="#333333")

    if xlim:
        ax.set_xlim(*xlim)

    ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#CCCCCC")

    ax.invert_yaxis()
    plt.tight_layout()
    path = os.path.join(CHART_DIR, filename)
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def generate_all_charts(results):
    """为四个实验生成图表，返回 {实验名: [图路径]}"""
    charts = {}

    # ---------- 实验1：召回策略 ----------
    if "召回策略" in results:
        charts["召回策略"] = []
        charts["召回策略"].append(plot_grouped_bar(
            results["召回策略"],
            ["Recall@3", "引用准确率"],
            "实验1：召回质量对比（Recall@3 / 引用准确率）",
            "exp1_recall_quality.png"
        ))
        charts["召回策略"].append(plot_single_metric(
            results["召回策略"],
            "MRR",
            "实验1：MRR 对比（纵轴 0–1，越高越好）",
            "exp1_mrr.png",
            higher_is_better=True,
            color="#5B8FF9",
            xlim=(0, 1)
        ))
        charts["召回策略"].append(plot_single_metric(
            results["召回策略"],
            "总幻觉率",
            "实验1：总幻觉率对比（越低越好）",
            "exp1_hallucination.png",
            higher_is_better=False,
            color="#E8684A"
        ))
        charts["召回策略"].append(plot_single_metric(
            results["召回策略"],
            "平均耗时",
            "实验1：平均耗时对比（越低越好）",
            "exp1_latency.png",
            higher_is_better=False,
            color="#9270CA"
        ))

    # ---------- 实验2：切分方法 ----------
    if "切分方法" in results:
        charts["切分方法"] = []
        charts["切分方法"].append(plot_grouped_bar(
            results["切分方法"],
            ["Recall@3", "引用准确率"],
            "实验2：切分方法召回质量对比",
            "exp2_recall_quality.png"
        ))
        charts["切分方法"].append(plot_single_metric(
            results["切分方法"],
            "MRR",
            "实验2：MRR 对比（纵轴 0–1）",
            "exp2_mrr.png",
            higher_is_better=True,
            color="#5B8FF9",
            xlim=(0, 1)
        ))
        charts["切分方法"].append(plot_single_metric(
            results["切分方法"],
            "总幻觉率",
            "实验2：切分方法幻觉率对比（越低越好）",
            "exp2_hallucination.png",
            higher_is_better=False,
            color="#E8684A"
        ))

    # ---------- 实验3：向量化模型 ----------
    if "向量化模型" in results:
        charts["向量化模型"] = []
        charts["向量化模型"].append(plot_grouped_bar(
            results["向量化模型"],
            ["Recall@3", "引用准确率"],
            "实验3：向量化模型召回质量对比",
            "exp3_recall_quality.png"
        ))
        charts["向量化模型"].append(plot_single_metric(
            results["向量化模型"],
            "MRR",
            "实验3：向量化模型 MRR 对比（纵轴 0–1）",
            "exp3_mrr.png",
            higher_is_better=True,
            color="#5B8FF9",
            xlim=(0, 1)
        ))
        charts["向量化模型"].append(plot_single_metric(
            results["向量化模型"],
            "总幻觉率",
            "实验3：向量化模型幻觉率对比（越低越好）",
            "exp3_hallucination.png",
            higher_is_better=False,
            color="#E8684A"
        ))

    # ---------- 实验4：防幻觉 Prompt ----------
    if "防幻觉" in results:
        charts["防幻觉"] = []
        charts["防幻觉"].append(plot_grouped_bar(
            results["防幻觉"],
            ["有答案类幻觉率", "无答案类幻觉率", "总幻觉率"],
            "实验4：防幻觉 Prompt 效果对比（越低越好）",
            "exp4_hallucination.png"
        ))
        charts["防幻觉"].append(plot_grouped_bar(
            results["防幻觉"],
            ["答案正确率", "拒答准确率"],
            "实验4：回答质量对比（越高越好）",
            "exp4_quality.png"
        ))

    return charts


# ============================================================
# 报告生成
# ============================================================
def generate_report(results, output_path=None):
    if output_path is None:
        output_path = os.path.join(REPORT_DIR, f"evaluation_report_{TIMESTAMP}.md")

    report = []

    report.append("# RAG 系统评估报告\n")
    report.append(f"**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**报告目录**：`{REPORT_DIR}`\n")
    if FONT_NAME:
        report.append(f"**图表中文字体**：{FONT_NAME}\n")
    else:
        report.append("**图表中文字体**：未找到中文字体，图表中文可能显示为方块\n")
    report.append("---\n")

    charts = generate_all_charts(results)

    # ---------- 实验1 ----------
    if "召回策略" in results:
        report.append("## 实验 1：四种召回策略对比\n")
        columns = ["策略", "Recall@3", "MRR", "引用准确率",
                   "总幻觉率", "拒答准确率", "平均耗时", "成本"]
        report.append(generate_table(results["召回策略"], columns))

        for p in charts.get("召回策略", []):
            rel = os.path.relpath(p, os.path.dirname(output_path))
            report.append(f"\n![召回策略图表]({rel})\n")

        best_recall = find_best(results["召回策略"], "Recall@3")
        best_mrr = find_best(results["召回策略"], "MRR")
        best_hallu = find_best(results["召回策略"], "总幻觉率", higher_is_better=False)

        report.append("\n**分析**：\n")
        if best_recall:
            report.append(f"- Recall@3 最高：**{best_recall['策略']}**（{best_recall['Recall@3']}）\n")
        if best_mrr:
            report.append(f"- MRR 最高：**{best_mrr['策略']}**（{best_mrr['MRR']}）\n")
        if best_hallu:
            report.append(f"- 总幻觉率最低：**{best_hallu['策略']}**（{best_hallu['总幻觉率']}）\n")
        report.append("\n---\n")

    # ---------- 实验2 ----------
    if "切分方法" in results:
        report.append("## 实验 2：切分方法对比\n")
        columns = ["策略", "Recall@3", "MRR", "引用准确率",
                   "总幻觉率", "拒答准确率", "平均耗时"]
        report.append(generate_table(results["切分方法"], columns))

        for p in charts.get("切分方法", []):
            rel = os.path.relpath(p, os.path.dirname(output_path))
            report.append(f"\n![切分方法图表]({rel})\n")

        best_recall = find_best(results["切分方法"], "Recall@3")
        best_mrr = find_best(results["切分方法"], "MRR")

        report.append("\n**分析**：\n")
        if best_recall:
            report.append(f"- Recall@3 最高：**{best_recall['策略']}**（{best_recall['Recall@3']}）\n")
        if best_mrr:
            report.append(f"- MRR 最高：**{best_mrr['策略']}**（{best_mrr['MRR']}）\n")
        report.append("\n---\n")

    # ---------- 实验3 ----------
    if "向量化模型" in results:
        report.append("## 实验 3：向量化模型对比\n")
        columns = ["策略", "Recall@3", "MRR", "引用准确率",
                   "总幻觉率", "拒答准确率", "平均耗时", "成本"]
        report.append(generate_table(results["向量化模型"], columns))

        for p in charts.get("向量化模型", []):
            rel = os.path.relpath(p, os.path.dirname(output_path))
            report.append(f"\n![向量化模型图表]({rel})\n")

        best_recall = find_best(results["向量化模型"], "Recall@3")
        best_cost = find_best(results["向量化模型"], "成本", higher_is_better=False)

        report.append("\n**分析**：\n")
        if best_recall:
            report.append(f"- Recall@3 最高：**{best_recall['策略']}**（{best_recall['Recall@3']}）\n")
        if best_cost:
            report.append(f"- 成本最低：**{best_cost['策略']}**（{best_cost['成本']}）\n")
        report.append("\n---\n")

    # ---------- 实验4 ----------
    if "防幻觉" in results:
        report.append("## 实验 4：防幻觉 Prompt 对比\n")
        columns = ["策略", "Recall@3", "MRR", "引用准确率",
                   "有答案类幻觉率", "无答案类幻觉率", "总幻觉率", "拒答准确率"]
        report.append(generate_table(results["防幻觉"], columns))

        for p in charts.get("防幻觉", []):
            rel = os.path.relpath(p, os.path.dirname(output_path))
            report.append(f"\n![防幻觉图表]({rel})\n")

        best_hallu = find_best(results["防幻觉"], "总幻觉率", higher_is_better=False)
        best_reject = find_best(results["防幻觉"], "拒答准确率")

        report.append("\n**分析**：\n")
        if best_hallu:
            report.append(f"- 总幻觉率最低：**{best_hallu['策略']}**（{best_hallu['总幻觉率']}）\n")
        if best_reject:
            report.append(f"- 拒答准确率最高：**{best_reject['策略']}**（{best_reject['拒答准确率']}）\n")
        report.append("\n---\n")

    # ---------- 综合结论 ----------
    report.append("## 综合结论\n")
    report.append("### 最优方案推荐\n")

    if "召回策略" in results:
        best = find_best(results["召回策略"], "MRR")
        if best:
            report.append(f"1. **召回策略**：推荐 **{best['策略']}**\n")
            report.append(f"   - Recall@3：{best['Recall@3']}\n")
            report.append(f"   - MRR：{best['MRR']}\n")
            report.append(f"   - 总幻觉率：{best['总幻觉率']}\n")

    if "切分方法" in results:
        best = find_best(results["切分方法"], "MRR")
        if best:
            report.append(f"2. **切分方法**：推荐 **{best['策略']}**\n")
            report.append(f"   - Recall@3：{best['Recall@3']}\n")
            report.append(f"   - MRR：{best['MRR']}\n")

    if "向量化模型" in results:
        best = find_best(results["向量化模型"], "MRR")
        if best:
            report.append(f"3. **向量化模型**：推荐 **{best['策略']}**\n")
            report.append(f"   - Recall@3：{best['Recall@3']}\n")
            report.append(f"   - 成本：{best['成本']}\n")

    if "防幻觉" in results:
        best = find_best(results["防幻觉"], "总幻觉率", higher_is_better=False)
        if best:
            report.append(f"4. **防幻觉 Prompt**：推荐 **{best['策略']}**\n")
            report.append(f"   - 总幻觉率：{best['总幻觉率']}\n")
            report.append(f"   - 拒答准确率：{best['拒答准确率']}\n")

    report.append("\n### 指标说明\n")
    report.append("| 指标 | 含义 |\n")
    report.append("|:---|:---|\n")
    report.append("| Recall@3 | 前3个检索结果中包含正确答案的比例 |\n")
    report.append("| MRR | 正确答案排名的倒数平均值 |\n")
    report.append("| 引用准确率 | 引用来源与标准来源匹配的比例 |\n")
    report.append("| 答案正确率 | 回答包含正确答案的比例 |\n")
    report.append("| 漏答率 | 有答案类问题中被拒答的比例 |\n")
    report.append("| 有答案类幻觉率 | 有答案类问题中编造答案的比例 |\n")
    report.append("| 无答案类幻觉率 | 无答案类问题中硬编答案的比例 |\n")
    report.append("| 总幻觉率 | 所有编造答案数 / 评估集总数 |\n")
    report.append("| 拒答准确率 | 无答案类问题中正确拒答的比例 |\n")
    report.append("\n---\n")

    report.append("## 附录：原始评估数据\n")
    report.append("```json\n")
    report.append(json.dumps(results, ensure_ascii=False, indent=2))
    report.append("\n```\n")

    content = "\n".join(report)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✅ 评估报告已生成: {output_path}")
    print(f"✅ 图表目录: {CHART_DIR}")
    return content, output_path


# ============================================================
# 入口
# ============================================================
def main():
    print("=" * 60)
    print("评估报告生成器（可视化增强版 · MRR 纵轴 0-1）")
    print("=" * 60)

    if not os.path.exists("evaluation_result.json"):
        print("❌ evaluation_result.json 不存在，请先跑 evaluate.py")
        return

    results = load_results()
    generate_report(results)


if __name__ == "__main__":
    main()