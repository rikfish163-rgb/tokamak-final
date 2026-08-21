# Nature Portfolio submission-readiness audit

Audit date: 2026-08-20  
Target style: Communications Physics / Nature Portfolio machine-learning reporting  
Current evidence status: `DEVELOPMENT_EVIDENCE_NOT_INDEPENDENT_CONFIRMATION`  
Submission verdict: **NOT READY for a confirmatory performance or deployment claim**

This document separates reporting quality from evidence strength. A complete,
honest manuscript can still be scientifically underpowered for the claim it
would like to make. Passing the repository validator means the archived local
evidence is internally consistent; it does not waive the failed scientific
gates below.

Authoritative external checklists:

- Nature Portfolio machine-learning reporting checklist:
  https://www.nature.com/documents/ML-reporting-summary.pdf
- Communications Physics submission guidelines:
  https://www.nature.com/commsphys/submit/submission-guidelines
- Communications Physics editorial criteria:
  https://www.nature.com/commsphys/aims

## 1. Claim-level verdict

| Claim | Current ceiling | Verdict | Reason |
|---|---|---|---|
| The three-stage chain executes causally | Local replay/audit claim | PASS | Per-shot prediction, query, reveal and update ledgers preserve `predict → seal → query → selected reveal → later-shot update` |
| The reported metrics match archived ledgers | Local reproducibility claim | PASS | Metrics are rebuilt from per-shot ledgers and HDF5 truth; J-TEXT lead is independently recomputed from archived alarm time and HDF5 DownTime; artifact hashes and tests are retained |
| The source model generalizes prospectively on J-TEXT | External-validity claim | FAIL | VAL160/TEST220 are outcome-balanced, count-aligned, non-chronological comparators |
| Few-shot transfer generalizes to unseen EAST | External-validity claim | FAIL | support110 both fits target components and derives the threshold; balanced360 was used for method development |
| More target labels alone explain the online operating point | Within-stream alternative explanation | REJECTED DIAGNOSTICALLY | Exact selected-label and block-matched random front-load controls raise FPR enough to lower TPR@30-FPR; queried training shots are excluded from scoring, but the analysis remains post-selection |
| Active querying beats random querying | Comparative causal claim | PARTIAL / DEVELOPMENT ONLY | Active TPR@30-FPR exceeds three block-matched random-query replicates per seed in all three seeds, but realized error-triggered update attempts differ, three-sign $p=0.125$, and balanced360 informed development |
| Online adaptation improves a future EAST stream | Prospective claim | FAIL | balanced360 informed architecture and 972-candidate selection; no pristine future stream remains |
| The source representation is shortcut-free | Mechanistic claim | FAIL | Primary padding/context probe AUC is 0.955/0.929/0.937 |
| The system is deployment ready | Operational claim | FAIL | Worst temporal blocks fail, very-early alarms remain, and latency/actuator/fail-safe validation is absent |

## 2. Nature ML reporting checklist mapping

| Requirement | Status | Current evidence | Required before submission |
|---|---|---|---|
| Data sources and label authority | PASS | HDF5 `meta/IsDisrupt` and `meta/DownTime`; exact roster hashes | Keep the machine evidence and describe acquisition/diagnostic provenance in author-supplied Methods |
| Dataset sizes, class counts and preprocessing | PASS | Counts, shared/private feature contracts, masks and scalers are documented | Add any institution-specific data quality/exclusion policy not present in code |
| Train/validation/test separation | PARTIAL | Zero shot overlap is verified | Replace or supplement J-TEXT count-aligned roles with a strict chronological evaluation |
| Split mimics intended application | FAIL | J-TEXT VAL/TEST are balanced; EAST balanced360 is balanced and development-selected | Use realistic-prevalence chronological confirmation cohorts |
| Leakage controls | PASS for the executed online order | Prediction-time A1 loading excludes outcome metadata; queried labels are lazily read after seal; support360 overlap is zero | Preserve this protocol on the future stream and audit all data-derived normalizers |
| Independent test dataset | FAIL | No demonstrably pristine local future EAST stream | Acquire and seal a new chronological EAST stream before outcomes are opened |
| Metric definitions and uncertainty | PARTIAL | Lead-qualified TPR/FPR and moving-block sensitivity are explicit | Pre-register the operating rule; run the learner on prospective data; avoid post-selection inference |
| Simple, relevant and state-of-the-art baselines | PARTIAL | strict EAST-only, source-scaler target-only, static, no-label, lag, flip, feature, selected/random front-load and causal block-matched random-query controls are archived | Add externally selected methods only if justified; isolate realized compute or retain it explicitly as a mediator |
| Ablations | PARTIAL | padding repair, temporal-zero and raw8-zero are preserved | Re-evaluate a repaired source parent with a target policy selected outside balanced360 |
| Bias/shortcut analysis | PASS as diagnosis | Padding, shot-ID, duration, temporal blocks and >3 s alarms are audited | Pass the gates on a new prospective cohort |
| Compute resources and cost | PARTIAL | GPU, CUDA environment, source wall times and update counts are recorded | Record target/online wall time, peak memory, FLOPs or a reproducible proxy, and energy if required |
| Data availability | FAIL for external reproduction | Exact local contract exists but raw HDF5 has no DOI/access route | Deposit data or document controlled access, contact, eligibility and expected response time |
| Code availability | FAIL for archival submission | Local release and source parents exist, but historical intermediate target states are not bundled | Publish a versioned repository and immutable DOI; archive target states or state explicitly that only deterministic re-execution is supported |
| Model card | PASS for the local package | `MODEL_CARD.md` | Update authorship/version/DOI on release |
| Author/contribution/conflict declarations | FAIL | Not inferred or fabricated | Authors must supply names, affiliations, CRediT, funding, acknowledgements and conflicts |
| Generative-AI disclosure | PASS in development draft | Main manuscript states the role and non-authorship of AI assistance | Named authors must verify the journal-specific wording |

## 3. Statistical adjudication

The displayed uncertainty uses 5,000 paired circular moving-block resamples of
the observed chronological ledger, with block size 20 and sensitivity at
10/20/30/45. Identical sampled positions are used for static/online and all
three seeds. This is preferable to treating shots as iid, but it still has two
non-removable limitations:

1. it resamples the sealed outputs and does not rerun the adaptive learner under
   the resampled history; and
2. the selected policy came from 972 candidates evaluated on the same stream.

Therefore the intervals are labelled **descriptive, unadjusted, post-selection
sensitivity intervals**. They cannot be used as confirmatory p-values or as
confidence intervals for a new campaign.

## 4. Minimum claim-changing experiment

Before the first shot of a new EAST stream, freeze and archive:

1. code/environment/data-roster hashes and all three parent checkpoints;
2. source, target and alarm thresholds, state-machine bounds and early-alarm rule;
3. query budget, active selector and an equal-budget random selector;
4. target-only/EAST-only and static branches with matched target sampling and explicitly logged capacity/FLOP differences;
5. overall and worst-block TPR/FPR/lead gates, campaign-correlation limits and
   >3 s alarm handling; and
6. data/code release or controlled-access procedures.

Run static, active, causal random-query, label-front-load and no-label branches
from identical parents, with sampled windows, optimizer steps and FLOPs logged.
For every shot, seal prediction and query decision before any outcome reveal.
Report the complete stream regardless of failure. No result in the current
balanced360 development stream may change the frozen choices.

## 5. Final audit statement

The current package is suitable as an auditable development manuscript and as
the preregistration basis for the missing confirmation experiment. It is not
logically defensible to label it a complete Nature Portfolio confirmation or a
deployment-ready result. That boundary is an evidence limitation, not a
formatting defect, and cannot be repaired by stronger prose, additional
bootstrap repetitions or further tuning on balanced360.
