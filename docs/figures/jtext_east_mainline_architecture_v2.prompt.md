# J-TEXT → EAST README architecture figure

This file records the provenance and design contract for
`jtext_east_mainline_architecture_v2.png`.

- Generator: Codex built-in `image_gen` tool
- Mode: image generation using the previous README diagram as a structural reference
- Intended use: GitHub README architecture overview
- Generated asset: `jtext_east_mainline_architecture_v2.png`
- Source-of-truth for implementation details: `IMPLEMENTATION_DETAILS_CN.md`

The generated image is an explanatory overview, not a machine-readable model
specification. Exact dimensions, losses, label contracts, and active versus shadow
branches remain defined by the tracked code, configuration, and implementation
document.

## Prompt

```text
Create a publication-grade, wide landscape scientific architecture schematic for
the J-TEXT to EAST tokamak disruption prediction pipeline. Preserve the factual
three-stage flow of the previous README diagram, but completely redesign the visual
style.

Use a white background, flat vector-like 2D line art, thin crisp strokes, generous
whitespace, large Arial/Helvetica-like labels, and a colorblind-safe palette. Avoid
gradients, shadows, 3D effects, glossy cards, decorative illustrations, tiny
paragraphs, metrics, watermarks, and excessive boxes. Use lowercase Nature-style
panel labels a, b, c.

Panel a: "J-TEXT source learning"
Causal signals → Shared17 encoder → Risk model → Source seal.

Panel b: "Few-shot EAST adaptation"
Frozen source and Support110 feed a target shared adapter and an EAST-private raw8
path, which produce the Static EAST model.

Panel c: "Ordered online active learning"
Causal prefix → Fast intra-shot update → Predict + seal → Query value Q →
Reveal if selected → Persistent inter-shot update. Put a vertical dashed boundary
immediately before label reveal and label it "NO OUTCOME ACCESS". Add a restrained
feedback arrow from persistent update to the next shot, labeled "NEXT SHOT".

Color semantics: frozen/source in deep blue, target-supervised operations in
vermilion/orange, EAST-private/deployed operations in bluish green, query control in
muted purple, and neutral structure in charcoal. The figure must remain legible in
grayscale through line and layout differences.
```

