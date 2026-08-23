#!/usr/bin/env python3
"""Deterministically convert the audited Chinese Markdown report to LaTeX body.

The generated body is committed so compiling the report does not require Python.
Run this converter only when ``TECHNICAL_REPORT_zh.md`` changes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import mistune


PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE / "TECHNICAL_REPORT_zh.md"
OUTPUT = PACKAGE / "technical_report_zh" / "report_body.tex"


CITATIONS = {
    "1--4": "zheng2023transfer,shen2023pgfeda,shen2024thesis,zheng2025phydanet",
    "5": "nie2023patchtst",
    "6--8": "ganin2016dann,sun2016coral,bendavid2010impossibility",
    "9": "geirhos2020shortcut",
    "10--12": "wang2021tent,niu2022eata,wang2022cotta",
    "13": "zhou2022online",
}


def figure(path: str, caption: str, label: str, width: str = r"\linewidth") -> str:
    return rf"""
\begin{{figure}}[htbp]
  \centering
  \includegraphics[width={width}]{{{path}}}
  \caption{{{caption}}}
  \label{{{label}}}
\end{{figure}}
"""


FIGURES_AFTER = {
    "报告摘要与主要结论": figure(
        "fig2_same_lineage_metrics_and_block_resampling.pdf",
        "同一模型谱系下的三阶段指标与静态至在线的配对变化。所有数值均由冻结逐炮账本重算；重采样区间仅用于开发流描述，不构成独立确认。",
        "fig:three-stage-metrics",
    ),
    "数据合同与标签依据": figure(
        "fig1_protocol_and_causal_contract.pdf",
        "数据角色、标签权威源和在线信息访问顺序。当前炮必须先完成预测与封存，随后才允许作出查询并揭示被选炮标签。",
        "fig:data-contract",
    ),
    "跨装置迁移的主要技术障碍": figure(
        "fig0_end_to_end_model_architecture.pdf",
        "实际执行的源域监督、目标域小样本适配、EAST 私有支路与因果在线更新结构。颜色表示计算角色，不表示证据等级。",
        "fig:architecture",
    ),
    "EAST 小样本迁移": figure(
        "fig10_target_only_vs_transfer.pdf",
        "严格 EAST-only、仅复用源缩放器的目标模型与 J-TEXT 至 EAST 静态迁移的三随机种子比较。",
        "fig:target-only-transfer",
    ),
    "EAST 因果在线适配": figure(
        "concept_causal_online_v1.png",
        "炮内临时适配、稀疏查询、信息边界、保护更新与回滚的概念示意。该图由生成式图像工具制作，仅表达方法结构，不包含测量数据、模型输出或实验数值。",
        "fig:concept-causal-online",
        r"0.80\linewidth",
    ),
    "多参数报警函数和状态机": figure(
        "fig5_adaptive_alarm_state_machine.pdf",
        "四个报警状态参数在 balanced360 顺序开发流中的真实演化轨迹。参数只在已查询炮揭示后更新，并受投影边界约束。",
        "fig:alarm-state",
    ),
    "动态误报保护": figure(
        "fig6_causal_false_alarm_rescue_shot73790.pdf",
        "安全炮73790的开发性案例分析：风险、物理支持与因果 episode veto 的关系。单炮案例不构成总体泛化证据。",
        "fig:false-alarm-case",
    ),
    "同查询标签预算前置训练消融": figure(
        "fig8_equal_label_budget_frontload_ablation.pdf",
        "同标签访问量前置训练消融。评分严格排除被用于前置训练的查询炮；普通前置重训提高灵敏度的同时显著扩大误报。",
        "fig:frontload-ablation",
    ),
    "因果随机查询在线对照": figure(
        "fig9_active_vs_causal_random_query.pdf",
        "主动查询与按20炮块匹配标签数的因果随机查询对照。实际更新尝试数不相等，因此该图描述完整耦合机制，而非严格等 FLOPs 的选择器效应。",
        "fig:active-random",
    ),
    "固定窗口、batch形状与optimizer-attempt的查询名单对照": figure(
        "fig11_equal_compute_selector_screen.pdf",
        "固定窗口、批形状和 optimizer-attempt 的查询名单筛查。归档主动名单在该固定同化算子下未显示稳定优于随机名单。",
        "fig:equal-compute",
    ),
    "负对照与失败实验记录": figure(
        "fig7_negative_controls_and_alarm_robustness.pdf",
        "负对照和前置报警器参数鲁棒性。负对照用于限制机制解释；鲁棒性结果不替代最终在线链路的独立验证。",
        "fig:negative-controls",
    ),
    "时间漂移与有效性风险": figure(
        "fig4_chronological_blocks_and_active_query_value.pdf",
        "连续时间块异质性、查询错误富集和静态至在线累计报警变化。时间块结果揭示整体均值可能掩盖 campaign 漂移。",
        "fig:chronological-risk",
    ),
    "PADFIX 上下文捷径修复及负迁移": figure(
        "fig3_source_shortcut_and_padfix_tradeoff.pdf",
        "源域上下文长度捷径与 PADFIX 权衡。PADFIX 降低 padding 可探测性，但直接沿用旧目标策略时发生负迁移。",
        "fig:padfix-tradeoff",
    ),
}


SPECIAL_TEXT = {
    "→": r"\ensuremath{\rightarrow}",
    "≥": r"\ensuremath{\geq}",
    "≤": r"\ensuremath{\leq}",
    "×": r"\ensuremath{\times}",
    "±": r"\ensuremath{\pm}",
    "Δ": r"\ensuremath{\Delta}",
    "ρ": r"\ensuremath{\rho}",
    "–": "--",
    "—": "---",
    "‑": "-",
    "−": r"\ensuremath{-}",
}


def escape_text(value: str) -> str:
    """Escape prose while preserving registered citation markers."""
    value = value.replace("[1–4]", "[1--4]").replace("[6–8]", "[6--8]")
    value = value.replace("[10–12]", "[10--12]")
    pattern = re.compile(r"\[(1--4|5|6--8|9|10--12|13)\]")
    pieces: list[str] = []
    cursor = 0
    for match in pattern.finditer(value):
        pieces.append(escape_plain(value[cursor : match.start()]))
        pieces.append(r"\citep{" + CITATIONS[match.group(1)] + "}")
        cursor = match.end()
    pieces.append(escape_plain(value[cursor:]))
    return "".join(pieces)


def escape_plain(value: str) -> str:
    result: list[str] = []
    mapping = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "/": r"/\allowbreak{}",
    }
    for char in value:
        if char in SPECIAL_TEXT:
            result.append(SPECIAL_TEXT[char])
        else:
            result.append(mapping.get(char, char))
    return "".join(result)


def escape_code(value: str) -> str:
    escaped = escape_plain(value)
    escaped = escaped.replace(r"\_", r"\_\allowbreak{}")
    return escaped


def heading_text(node: dict[str, Any]) -> str:
    return "".join(child.get("raw", "") for child in node.get("children", []))


def strip_registered_number(value: str) -> str:
    # Mistune removes the punctuation after an ATX heading number (for example,
    # ``### 7.2. Title`` is exposed as ``7.2 Title``).  Accept both forms so
    # LaTeX remains the only source of visible section numbering.
    return re.sub(r"^\d+(?:\.\d+)*(?:\.)?\s+", "", value).strip()


class LatexRenderer:
    def __init__(self) -> None:
        self.in_unumbered_intro = True
        self.skip_references = False

    def render_children(self, node: dict[str, Any]) -> str:
        # Mistune deliberately splits an unresolved Markdown reference such as
        # ``[10–12]`` into three adjacent text nodes.  Coalesce adjacent prose
        # before escaping it so registered numeric citations become real
        # BibTeX references rather than decorative bracketed text.
        rendered: list[str] = []
        prose: list[str] = []

        def flush_prose() -> None:
            if prose:
                rendered.append(escape_text("".join(prose)))
                prose.clear()

        for child in node.get("children", []):
            if child.get("type") == "text":
                prose.append(child.get("raw", ""))
            else:
                flush_prose()
                rendered.append(self.render(child))
        flush_prose()
        return "".join(rendered)

    def render(self, node: dict[str, Any]) -> str:
        kind = node.get("type")
        if kind in {"blank_line", "block_html"}:
            return ""
        if kind == "text":
            return escape_text(node.get("raw", ""))
        if kind == "softbreak":
            return " "
        if kind == "linebreak":
            return r"\\" + "\n"
        if kind == "strong":
            return r"\textbf{" + self.render_children(node) + "}"
        if kind == "emphasis":
            return r"\emph{" + self.render_children(node) + "}"
        if kind == "codespan":
            raw = node.get("raw", "")
            if re.fullmatch(r"[A-Za-z0-9_./:+@-]+", raw):
                pieces = re.split(r"([_./:+-])", raw)
                return "".join(
                    r"\texttt{" + escape_plain(piece) + "}"
                    + (r"\allowbreak{}" if re.fullmatch(r"[_./:+-]", piece) else "")
                    for piece in pieces
                    if piece
                ) + r"\allowbreak{}"
            return r"\texttt{" + escape_code(raw) + r"}\allowbreak{}"
        if kind == "inline_math":
            return "$" + node.get("raw", "").strip() + "$"
        if kind == "block_math":
            return "\n\\[\n" + node.get("raw", "").strip() + "\n\\]\n\n"
        if kind == "link":
            url = node.get("attrs", {}).get("url", "")
            return r"\href{" + url.replace("%", r"\%").replace("#", r"\#") + "}{" + self.render_children(node) + "}"
        if kind == "paragraph":
            return self.render_children(node).strip() + "\n\n"
        if kind == "block_text":
            return self.render_children(node).strip()
        if kind == "block_code":
            raw = node.get("raw", "").rstrip("\n")
            return "\n\\begin{ReportVerbatim}\n" + raw + "\n\\end{ReportVerbatim}\n\n"
        if kind == "thematic_break":
            return "\n\\FloatBarrier\n\\clearpage\n"
        if kind == "block_quote":
            return "\n\\begin{evidencebox}\n" + self.render_children(node) + "\\end{evidencebox}\n\n"
        if kind == "list":
            return self.render_list(node)
        if kind == "list_item":
            content = self.render_children(node).strip()
            return "\\item " + content + "\n"
        if kind == "table":
            return self.render_table(node)
        if kind == "heading":
            return self.render_heading(node)
        raise ValueError(f"Unsupported Markdown AST node: {kind!r}")

    def render_list(self, node: dict[str, Any]) -> str:
        ordered = bool(node.get("attrs", {}).get("ordered"))
        env = "enumerate" if ordered else "itemize"
        return f"\n\\begin{{{env}}}\n" + self.render_children(node) + f"\\end{{{env}}}\n\n"

    def render_heading(self, node: dict[str, Any]) -> str:
        raw = heading_text(node).strip()
        level = int(node.get("attrs", {}).get("level", 2))
        if level == 1:
            return ""
        if raw == "目录":
            return ""
        if re.match(r"^35(?:\.)?\s+参考文献", raw):
            self.skip_references = True
            return ""

        numbered = bool(re.match(r"^\d", raw))
        title = strip_registered_number(raw)
        title_tex = escape_plain(title)
        if level == 2:
            if numbered:
                self.in_unumbered_intro = False
                out = "\n\\FloatBarrier\n\\section{" + title_tex + "}\n"
            else:
                out = (
                    "\n\\FloatBarrier\n\\section*{" + title_tex + "}\n"
                    "\\addcontentsline{toc}{section}{" + title_tex + "}\n"
                )
        elif level == 3:
            if self.in_unumbered_intro:
                out = "\n\\subsection*{" + title_tex + "}\n"
            else:
                out = "\n\\subsection{" + title_tex + "}\n"
        else:
            out = "\n\\subsubsection{" + title_tex + "}\n"

        out += FIGURES_AFTER.get(title, "")
        return out

    def render_table(self, node: dict[str, Any]) -> str:
        head = node["children"][0]
        body = node["children"][1]
        header_cells = head.get("children", [])
        rows = body.get("children", [])
        columns = max(
            len(header_cells),
            max((len(row.get("children", [])) for row in rows), default=0),
        )
        if columns == 0:
            return ""
        spec = "@{}" + "".join(r">{\RaggedRight\arraybackslash}X" for _ in range(columns)) + "@{}"
        size = r"\scriptsize" if columns >= 6 else (r"\footnotesize" if columns >= 4 else r"\small")

        def cells(values: list[dict[str, Any]]) -> str:
            rendered = [self.render_children(value).strip() for value in values]
            rendered += [""] * (columns - len(rendered))
            return " & ".join(rendered) + r" \\" + "\n"

        header_line = cells(header_cells)
        output = [
            "\n{",
            size,
            "\n\\renewcommand{\\arraystretch}{1.18}\n",
            f"\\begin{{xltabular}}{{\\textwidth}}{{{spec}}}\n",
            "\\toprule\n\\rowcolor{auditblue!10}\n",
            header_line,
            "\\midrule\n\\endfirsthead\n",
            "\\toprule\n\\rowcolor{auditblue!10}\n",
            header_line,
            "\\midrule\n\\endhead\n",
        ]
        for row in rows:
            output.append(cells(row.get("children", [])))
        output.extend(["\\bottomrule\n\\end{xltabular}\n}\n\n"])
        return "".join(output)


def main() -> None:
    markdown = mistune.create_markdown(renderer="ast", plugins=["table", "math"])
    ast = markdown(SOURCE.read_text(encoding="utf-8"))
    renderer = LatexRenderer()
    chunks: list[str] = [
        "% Generated from TECHNICAL_REPORT_zh.md. Do not edit manually.\n",
        "% Regenerate with scripts/convert_technical_report_zh_to_latex.py.\n\n",
    ]
    started = False
    for node in ast:
        if node.get("type") == "heading" and node.get("attrs", {}).get("level") == 2:
            started = True
        if not started:
            continue
        rendered = renderer.render(node)
        if renderer.skip_references:
            break
        chunks.append(rendered)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("".join(chunks).rstrip() + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
