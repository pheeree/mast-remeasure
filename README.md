# mast-remeasure

Re-measuring the failure-mode distribution of **MAST** (Multi-Agent System Failure Taxonomy;
Cemri et al. 2025, [arXiv:2503.13657](https://arxiv.org/abs/2503.13657), NeurIPS 2025 D&B)
on current-generation models.

## Hypothesis

The published MAST distribution is a snapshot of the GPT-4o / Claude 3 era (919 + 323 of the
1,242 released traces). If we re-run the measurement with current-generation models:

- **model-sensitive failure modes** (e.g., FM-2.6 reasoning–action mismatch) should shrink,
- **design-rooted failure modes** (e.g., FM-1.5 unawareness of termination conditions) should persist.

If that holds, it is an empirical decomposition of the "is it the design or the model?" debate.
If it does not hold, that is equally informative — it would mean the taxonomy's prevalence
structure is not model-bound in the way the framing suggests.

## Plan (pilot)

| Phase | What | Verify |
|---|---|---|
| 1. Judge calibration | Re-annotate the 19 human-labelled traces (MAST-Data-human) with 3 modern judges: 2× Claude (latest tier pair) + Gemini (cross-family) | κ vs. human labels, side by side with the original o1 judge (κ = 0.77); inter-mode correlation report |
| 2. Fresh traces | 30–50 runs on 1–2 of the original frameworks with current models, annotated by the calibrated judges | Distribution comparison centered on FM-2.6 vs. FM-1.5; direction call on the hypothesis |
| 3. Write-up | Results back into the research agenda; workshop submission decided after seeing the data | — |

Known interpretive limit, stated upfront: re-measurement only sees distribution shift *inside*
MAST's observation window. Failure classes outside it (adversarial, structural, homogeneity —
the governance-only residual) stay invisible by construction.

## Data

We do **not** vendor the dataset. Fetch it from the source
([mcemri/MAST-Data](https://huggingface.co/datasets/mcemri/MAST-Data), CC-BY-4.0):

```bash
bash scripts/download_data.sh
```

Two discrepancies between the paper and the released data, measured 2026-07-03:
the release contains **1,242** traces (paper: 1,642) and **19** human-labelled traces
(paper: 21). Both are recorded as-is and used as the operative ground truth.

We reference the MAST definitions and few-shot prompt structure from the original
[repo](https://github.com/multi-agent-systems-failure-taxonomy/MAST); the annotation calls are
re-implemented in a thin wrapper (the released `agentdash` package is OpenAI-only).

## Status — Phase 1 concluded (2026-07-08)

Judge calibration ran on the 19 human-labelled traces with the original agentdash
prompt (definitions + few-shot included), free-text responses, and the original
regex parser side-by-side with a strict line parser.

| judge / prompt order | acc | prec | recall | F1 | κ |
|---|---|---|---|---|---|
| o1, few-shot (paper Table 2) | 0.94 | 0.833 | 0.77 | 0.80 | **0.77** |
| gemini-2.5-flash, original order | 0.662 | 0.216 | 0.333 | 0.262 | **0.056** |
| gemini-2.5-flash, definitions-first | 0.624–0.647 | 0.217–0.233 | 0.417 | 0.286–0.299 | **0.064–0.087** |

- Parser artifacts account for only 6/266 cells (the shipped agentdash v0.1.0 parser
  has a reproducible misattribution bug — unescaped mode-ID dot + non-greedy DOTALL);
  the disagreement is real, not parsing noise.
- The judge does not even agree with itself across a semantically neutral prompt
  reordering: self-agreement κ = 0.460, with modes 3.2 ↔ 3.3 flipping wholesale.
- Pre-registered gate for Phase 2 (fresh-trace re-measurement) was κ ≥ 0.6.
  **No free-tier Gemini judge passes → Phase 2 is frozen.** The Phase-1 finding is
  the finding: the MAST LLM-judge does not transfer to this judge, so any
  re-measured distribution would reflect the scale, not the weight.
- Free-tier limits measured along the way: gemini-2.5-pro 0 req/day,
  gemini-2.5-flash 20 req/day.

Details: `results/calibration/report.md`. Resume path: enable paid tier (est. $2–5
for the full Phase 2 design: AG2 × GSM × 30–50 runs + same-judge re-annotation of
original traces) or add a non-Gemini judge.

## Research log

The running log is written in Korean at [pheeree.github.io](https://pheeree.github.io)
(tag: `mast-remeasure`). This repo carries the code, the numbers, and the English summary.

## License

- Code: MIT
- Text, annotations, and figures produced here: CC-BY-4.0
- Upstream data: CC-BY-4.0, © the MAST authors — cite [arXiv:2503.13657](https://arxiv.org/abs/2503.13657)
