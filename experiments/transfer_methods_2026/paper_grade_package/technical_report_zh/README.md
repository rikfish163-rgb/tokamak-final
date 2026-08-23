# 中文技术报告 LaTeX 工程

本目录是中文正式技术报告的标准、自包含 XeLaTeX 工程。仓库提交 `main.tex`、由审计版 Markdown 确定性生成的 `report_body.tex`、本地 `references.bib`、图件和最终 PDF，因此仅编译报告时不依赖 Python 或个人环境变量。

## 编译

```bash
cd experiments/transfer_methods_2026/paper_grade_package/technical_report_zh
make
```

依赖：XeLaTeX、`latexmk`、Noto Serif/Sans/Mono CJK SC 字体，以及 TeX Live 中的 `polyglossia`、`xltabular`、`natbib`、`tcolorbox` 和 `fvextra`。编译产物写入上一级目录：

```text
experiments/transfer_methods_2026/paper_grade_package/TECHNICAL_REPORT_zh.pdf
```

如先修改了 `../TECHNICAL_REPORT_zh.md`，执行：

```bash
make sync
make
```

`make sync` 使用系统 Python 3 和 Mistune，只负责把经审计的 Markdown 正文同步为 LaTeX；不要直接修改 `report_body.tex`。

## 图件边界

- `../figures/fig0_*.pdf` 至 `fig11_*.pdf`：由归档实验账本和统计结果确定性生成，是数值证据图。
- `figures/concept_*.png`：使用 OpenAI 内置 image generation 生成，仅用于表达研究架构和因果信息流，不包含实验数值，也不参与任何统计结论。
- 生成式图件的提示词、SHA256 和使用边界见 `IMAGEGEN_PROVENANCE.md` 与 `imagegen_manifest.json`。

原始 J-TEXT/EAST HDF5 数据按仓库合同保持在 Git 外部。复现时必须先通过内容清单和 SHA256 校验数据，不能用历史炮表覆盖 HDF5 标签。
