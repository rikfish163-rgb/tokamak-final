# Five-way independent audit

Audit date: 2026-08-22  
Scope: current v3c J-TEXT → EAST transfer → chronological online-active
development lineage  
Model used by all reviewers: `gpt-5.6-luna`, maximum reasoning, read-only

## Verdict matrix

| Review lane | Local package integrity | Nature confirmatory readiness |
|---|---|---|
| Statistical methods and uncertainty | PASS | FAIL |
| Leakage, label lineage and provenance | PASS | FAIL |
| Causal online adaptation semantics | PASS | FAIL |
| Reproducibility and software release | PASS | FAIL |
| Claims, manuscript and visualization | PASS | FAIL |

The five PASS decisions concern the package's declared local scope: files,
hashes, ledgers, metrics, PDFs and disclosures are internally consistent. They
do not mean that the development stream is an independent test. All five
reviewers independently rejected a confirmatory Nature-family performance,
generalization or deployment claim.

## Local findings repaired after the first review

The first review identified several fail-closed coverage and release issues.
The repository now:

1. validates the recoverable Q30 identities, positions, static-roster hash and
   explicit lost-ledger boundary;
2. validates, for every primary seed, that the prediction/query/static ledgers
   have identical 360-shot order, that selected queries equal reveals and
   query receipts, that prediction/query rows contain no outcome fields, and
   that queried/unqueried finalization receipts partition the stream;
3. validates the whole-file SHA-256 of all three sealed EAST target parents;
4. verifies both input and output hashes in figure and table manifests;
5. rejects workstation-specific absolute paths in released text artifacts and
   serializes compact random-query summaries with portable paths;
6. excludes regenerable online run caches and the parent-fidelity-invalid v1
   front-load run while retaining the corrected v2 evidence;
7. aligns `WORKFLOW.md` with the v3c archived-parent replay and adds the paper
   package validator to CI; and
8. expands the package validator from 109 to 117 passing checks.

## Findings that remain and cannot be repaired with the current evidence

1. `balanced360` informed architecture and 972-candidate method selection; no
   untouched chronological EAST future stream remains.
2. EAST `support110` is 15 disruptive / 95 safe by HDF5 and is used for both
   target fitting and threshold selection; there is no independent target
   validation cohort.
3. J-TEXT VAL160/TEST220 are count-aligned, outcome-balanced, non-chronological
   comparators and cannot establish prospective source-domain generalization.
4. The primary source parent fails the registered padding/context shortcut
   gate. PADFIX passes that gate but causes negative target transfer under the
   frozen policy.
5. Active querying produces about 2.21 times as many realized error-triggered
   inter-shot update attempts as the matched random schedules. The current
   comparison tests the full active mechanism, not a selector-only equal-FLOP
   effect; the exact one-sided three-sign probability is 0.125.
6. The Q30 shot identities and original ledger SHA are preserved, but the
   original 360-row decision ledger was lost after a tmpfs reboot. Its
   historical cutoff/value/route sequence cannot be reconstructed honestly.
7. Worst chronological blocks, very-early alarms and shortcut diagnostics do
   not support campaign-general or deployment-ready claims.
8. Data DOI/controlled access, immutable code DOI, authorship/CRediT/funding/
   conflicts and device-side latency/fail-safe evidence remain external tasks.

## Defensible claim ceiling

The package supports an auditable development claim:

> Under HDF5-authoritative labels and the archived v3c lineage, the local
> replay implements `predict → seal → query → selected reveal → later-shot
> update`; its three-stage point estimates, development controls, negative
> results and provenance boundaries are internally reproducible from the
> supplied parents, ledgers and external HDF5 contract.

It does **not** support an independent-confirmation, future-campaign,
shortcut-free mechanism, equal-FLOP selector-superiority, or deployment claim.
The claim-changing experiment is a newly acquired chronological EAST stream
with all parents, thresholds, query schedules, compute budgets and success
gates frozen before any outcome is revealed.
