# Corpus search provenance

Date of retrieval: 2026-11-26. Scope: literature base for J-TEXT -> EAST
disruption-forecast transfer, covering disruption prediction and warning,
multi-machine/cross-machine transfer, EAST and J-TEXT, MIT devices
(Alcator C-Mod, SPARC), JET/ITER, physics diagnostics and precursor signals,
plus directly applicable work on transfer alignment, sequential alarms,
calibration and conformal prediction.

## Indexes and access

| Index | Endpoint | Role | Result |
|---|---|---|---|
| OpenAlex | `https://api.openalex.org/works` | primary metadata + abstracts | 11,584 records over 58 queries |
| arXiv | `https://export.arxiv.org/api/query` | preprints, full abstracts | 928 records over 21 queries |
| Crossref | `https://api.crossref.org/works` | metadata/abstract fallback | reachable; not needed in bulk |
| Semantic Scholar | Graph API | planned abstract fallback | HTTP 429; not used |

OpenAlex abstracts were reconstructed from `abstract_inverted_index`; arXiv
abstracts came from the Atom `summary` field. Text was whitespace-normalized
and stripped of HTML/LaTeX formatting only. No abstract was paraphrased,
translated or synthesized.

## Queries

OpenAlex queries (58):

```
tokamak disruption prediction
disruption prediction machine learning tokamak
deep learning disruption prediction tokamak
plasma disruption forecasting
disruption warning system tokamak
disruption prediction ITER
disruption prediction JET
disruption prediction EAST tokamak
disruption prediction J-TEXT
disruption prediction KSTAR
disruption prediction ASDEX Upgrade
disruption prediction DIII-D
disruption prediction HL-2A
disruption prediction WEST tokamak
disruption prediction SPARC
disruption prediction Alcator C-Mod
multi-machine disruption prediction
transfer learning disruption prediction tokamak
cross-machine disruption prediction
domain adaptation tokamak disruption
few-shot disruption prediction
foundation model tokamak
transformer disruption prediction tokamak
LSTM disruption prediction tokamak
convolutional neural network disruption prediction
anomaly detection tokamak disruption
tearing mode disruption prediction
locked mode disruption prediction
mode locking disruption warning
disruption avoidance control tokamak
real-time disruption prediction
physics-informed disruption prediction
interpretable disruption prediction
uncertainty disruption prediction
tokamak control reinforcement learning
plasma state estimation tokamak
machine learning EAST tokamak
J-TEXT tokamak
J-TEXT disruption
tokamak diagnostic signal prediction
disruption database machine learning
early warning plasma instability
neural network plasma equilibrium
shot-to-shot transfer tokamak
pretrained model plasma physics
synthetic data disruption prediction
runaway electron prediction
disruption mitigation shattered pellet
plasma disruption prediction survey
tokamak disruption prediction review
time series foundation model scientific
test-time adaptation time series
transfer learning calibration uncertainty
concept drift time series prediction
alarm system calibration threshold
sequential alarm prediction
domain generalization time series classification
sim-to-real transfer control
```

arXiv queries (21):

```
all:"disruption prediction" AND all:tokamak
all:disruption AND all:tokamak AND all:"machine learning"
all:disruption AND all:tokamak AND all:"deep learning"
all:tokamak AND all:"transfer learning"
all:tokamak AND all:"domain adaptation"
all:tokamak AND all:forecasting
all:tokamak AND all:"neural network"
all:EAST AND all:tokamak
all:"J-TEXT"
all:tokamak AND all:"reinforcement learning"
all:tokamak AND all:"foundation model"
all:plasma AND all:"disruption" AND all:"warning"
all:tokamak AND all:"tearing mode"
all:"mode locking" AND all:tokamak
all:tokamak AND all:"anomaly detection"
all:"time series" AND all:"transfer learning" AND all:forecasting
all:"test-time adaptation"
all:"domain generalization" AND all:"time series"
all:"concept drift"
all:"uncertainty calibration" AND all:"deep learning"
all:"conformal prediction" AND all:"time series"
```

OpenAlex was paged with `per-page=200` and cursor pagination. Its free daily
budget returned HTTP 401 on later pages (from roughly query 47 onward), so
some of the last queries contain only their first page. That truncation is
reflected in the stored counts and did not affect the already-retrieved core
disruption queries.

## Retrieval and filtering

Raw records were gated into three channels using title + abstract text:

- `disruption_prediction`: a disruption term, a fusion-device term, and a
  prediction/warning/avoidance/mitigation term.
- `tokamak_ml_diagnostics`: a fusion-device term plus machine learning,
  forecasting, control, reconstruction, anomaly or diagnostic content.
- `transfer_calibration_methods`: transfer, adaptation, drift, calibration,
  conformal or sequential-alarm methods applicable to time series, without a
  fusion requirement.

Ambiguous acronyms (`EAST`, `WEST`, `JET`, `ITER`, `TCV`, `MAST`, `ARC`) were
matched case-sensitively to avoid false positives from ordinary English words
(for example "east" in a rainfall paper). Off-domain disruption uses (supply
chain, finance, business continuity) were excluded. The gated pool contained
3,039 records: 537 disruption, 1,199 tokamak-ML, 1,303 methods.

Selection then took the highest-consensus records per channel (ranked by the
number of distinct queries that returned them, then relevance score and
recency), with a manual relevance pass over titles. Earlier relevance gating
had already removed supply-chain and non-paper records; a final manual pass
removed eight more from the shortlist:

- a Dataverse dataset record whose title/abstract belonged to a different paper;
- an arXiv preprint duplicated by its journal version;
- an arXiv tutorial duplicated by its published Foundations & Trends version;
- five speculative or non-paper 2026 preprints whose claims were not
  corroborated by the rest of the literature.

## Deduplication

- DOIs normalized to lowercase with the `https://doi.org/` prefix removed;
  exact DOI duplicates dropped.
- Titles normalized with Unicode NFKD, LaTeX/HTML stripped, punctuation
  collapsed; exact normalized-title duplicates dropped.
- Fuzzy title matching (`difflib` ratio > 0.92, or identical 55-character
  prefix) applied to catch punctuation and subtitle variants.
- Abstract-prefix comparison caught three remaining duplicate clusters, which
  were resolved as described above.

Final checks: 281 rows, 0 duplicate DOIs, 0 duplicate normalized titles, 0
empty titles or URLs.

## Corpus composition

| Channel | Papers |
|---|---|
| Disruption prediction / warning / physics | 200 |
| Tokamak ML, diagnostics, control, forecasting | 42 |
| Transfer / calibration / sequential-alarm methods | 39 |
| **Total** | **281** |

Publication years span 1994-2026 (54 items dated 2026, mostly preprints).
Device mentions across title + abstract (not mutually exclusive): ITER 131,
JET 108, EAST 103, DIII-D 84, J-TEXT 42, ASDEX 36, ADITYA 29, HL-3 26, NSTX
22, HL-2A 20, SPARC 17, KSTAR 14, TCV 14, MAST 14, Alcator 13, WEST 7.

## Abstract and identifier coverage

- 281/281 records have an abstract of at least 200 characters.
- Minimum 461 characters; median 1,443; mean about 1,526.
- 280 abstracts are English; 1 is Chinese (kept verbatim, marked
  `present_nonenglish`).
- 253 records have a DOI (18 of these also carry an arXiv ID), 13 are
  arXiv-only, and 15 have neither but do have a repository or publisher URL.
- `abstract_status` is `present` for 280 rows and `present_nonenglish` for 1;
  no row is marked absent.

This satisfies the "200+ readable abstracts" check at abstract level. Full
texts were not read or verified for all 281 records.

## Known limitations

- Abstract-level screening only; no systematic full-text review.
- OpenAlex metadata and abstract assignment can be wrong. Three duplicate
  clusters were found and fixed; residual misassignments are possible.
- Preprints from arXiv and Zenodo are included and are not peer-review
  filtered. Several 2026 entries may be unrefereed or later revised.
- No Scopus/Web of Science; no citation-network expansion (Semantic Scholar
  was rate-limited).
- English-language queries only; Chinese-language literature is likely
  under-represented.
- OpenAlex quota truncation reduced recall for the last few method queries.
- Fifteen records lack a DOI/arXiv identifier; their URLs are repository
  handles and may be less stable.

## Column schema (`corpus.tsv`)

| Column | Meaning |
|---|---|
| `paper_id` | local stable ID, P001-P281 |
| `doi` | lowercase DOI, empty if none |
| `arxiv_id` | arXiv ID without version suffix, empty if none |
| `title` | cleaned title |
| `year` | publication year from the index |
| `venue` | journal/conference/repository name where available |
| `url` | DOI resolver, publisher landing page, arXiv abs page or repository handle |
| `abstract` | verbatim abstract text, formatting-normalized |
| `abstract_status` | `present` or `present_nonenglish` |
| `abstract_chars` | character count of the stored abstract |
| `source_index` | `openalex` or `arxiv` (source of the stored abstract) |
| `category` | `disruption_prediction`, `tokamak_ml_diagnostics` or `transfer_calibration_methods` |
| `query_hits` | number of distinct harvest queries that returned the record |
