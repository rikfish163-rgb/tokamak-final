# Image generation provenance

以下两幅图由 OpenAI 内置 image generation 模式生成，日期为 2026-08-23。两图的用途均为 `conceptual_non_data`：只表达方法结构，不含实验数据、坐标轴、指标、模型输出或定量结论。论文中的所有性能图继续由归档逐炮账本确定性生成。

## `concept_graphical_abstract_v1.png`

- SHA256: `406ce80c981d5bb6707f5be44efa18d54967f8f06e62aeaa034c127de50a680e`
- 像素尺寸: `1536 × 1024`
- 生成模式: OpenAI built-in image generation
- 最终提示词：

```text
Use case: scientific-educational
Asset type: publication graphical abstract for a Chinese technical report on cross-tokamak disruption prediction
Primary request: create a rigorous, clean left-to-right scientific visual showing three stages: a compact source tokamak, a central machine-learning transfer bridge, and a larger target tokamak with chronological online adaptation.
Scene/backdrop: pure white editorial background with generous margins.
Subject: on the left, a small blue tokamak cross-section with a restrained cyan plasma ring and a compact stack of multichannel causal signal traces; in the center, an abstract transformer made of aligned rectangular token blocks and attention links, with seventeen shared blue feature strands crossing the bridge while eight amber target-private diagnostic strands enter only near the target; on the right, a larger teal EAST-like tokamak cross-section followed by a chronological row of discharge pulses, a sparse query token symbol, a guarded feedback loop, and a rollback shield.
Style/medium: precise Nature-family scientific graphical abstract, flat vector-like geometry with subtle technical depth, crisp thin strokes, no decorative illustration style.
Composition/framing: landscape 3:2, strong left-to-right reading order, balanced three-part composition, ample negative space for LaTeX overlay labels.
Color palette: navy #16324F, blue #0072B2, cyan #56B4E9, teal #009E73, amber #E69F00, neutral gray.
Constraints: no text, no letters, no numbers, no equations, no axes, no legends, no logos, no watermark; do not depict human figures; avoid photorealism; visually distinguish shared transfer from target-private inputs and show that feedback affects only later discharges.
Avoid: glossy 3D, neon, dark background, gradients that reduce print clarity, fake data values, dense clutter.
```

## `concept_causal_online_v1.png`

- SHA256: `9e663d4df091c5d227e4cd34b88918b214f93603194e55edef7fa0a03b5c7090`
- 像素尺寸: `1536 × 1024`
- 生成模式: OpenAI built-in image generation
- 最终提示词：

```text
Use case: scientific-educational
Asset type: conceptual method figure for a peer-reviewed technical report
Primary request: create a clean scientific illustration of causal intra-shot and inter-shot adaptation with an explicit information barrier.
Scene/backdrop: pure white background, generous margins, print-safe.
Subject: a horizontal sequence of six simplified discharge cards. Inside each card, show a restrained teal risk trajectory and small blue physics-signal traces; within the current card, a small temporary adaptation loop updates only a translucent fast module and resets at the card boundary. Beneath selected cards, show a sparse amber query token that crosses a clear dashed information barrier only after that card's prediction is sealed. After the barrier, show a compact error-type memory, a guarded parameter update, a shield/check gate, and a rollback arrow; the output arrow must lead only to later cards, never back into the current or earlier card.
Style/medium: precise Nature-family scientific methods schematic, flat vector-like design, consistent thin strokes, publication-ready, subtle depth only.
Composition/framing: landscape 3:2, readable causal chronology from left to right, clear separation between prediction-time upper lane and post-reveal lower lane, ample negative space for LaTeX labels.
Color palette: navy #16324F, blue #0072B2, cyan #56B4E9, teal #009E73, amber #E69F00, neutral gray.
Constraints: no text, no letters, no numbers, no equations, no axes labels, no legends, no logos, no watermark; no fake experimental values; visually emphasize that the query is costly and sparse, the fast state resets every shot, and the supervised update affects future shots only.
Avoid: dark background, glossy 3D, neon, decorative gradients, dense circuitry, photorealism, humans, backward arrows that imply leakage.
```
