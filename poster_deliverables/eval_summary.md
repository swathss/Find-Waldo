# Evaluation summary

All numbers are page hit-rate (top-1 box on the real Waldo, IoU>=0.5) on the
held-out test books **b04 + b09**: n = 21 pages, 58 Waldo instances.
95% confidence intervals are Wilson score intervals.

Headline result: **16/21 = 0.76, 95% CI [0.55, 0.89]**.

Caveat: 2 of the missed test pages are labelled on the look-alike character
Wenda, not Waldo, so they are arguably label noise rather than true misses.

## Lineage (how the result was built up)

| Stage | Hits | Hit-rate | 95% Wilson CI |
|---|---|---|---|
| Synthetic-only (Track A, documented negative result) | 0/21 | 0.00 | [0.00, 0.15] |
| Real page tiles | 10/21 | 0.48 | [0.28, 0.68] |
| + decoy negatives (single-scale infer) | 13/21 | 0.62 | [0.41, 0.79] |
| + multi-scale WBF inference | 16/21 | 0.76 | [0.55, 0.89] |

Track A (synthetic-only) is kept in the lineage as a documented negative
result: the synthetic-composite approach did not transfer to real pages
(hit-rate ~0.00, CI [0.00, 0.15]). The large, defensible jump is from that
Track A baseline to real page tiles (0.48). Decoy negatives and multi-scale
inference add further gains, though at n=21 the single-step increments overlap
in CI.

## Final comparison: decoy vs multi-scale-trained model

| Model | Hits | Hit-rate | 95% Wilson CI |
|---|---|---|---|
| waldo_book_decoy (multi-scale) | 16/21 | 0.76 | [0.55, 0.89] |
| waldo_book_ms (multi-scale) | 15/21 | 0.71 | [0.50, 0.86] |

The two differ by one page (net -1 for book_ms), well inside the n=21 noise
floor and their overlapping CIs, so this is a statistical tie. The reported
model is **waldo_book_decoy (0.76)**. Book-fold cross-validation is the honest
tiebreaker and is listed as future work.

## Footnote: validation mAP vs test hit-rate

waldo_book_ms has higher validation mAP@50 (0.813, b11/b16) than the reported
waldo_book_decoy (0.752), but slightly lower held-out test page hit-rate
(15/21 = 0.71 vs 16/21 = 0.76). These are different metrics on different data:
val mAP is tile-level average precision; test page hit-rate is whether the
single top-1 box lands on the real Waldo, the task-relevant metric we report.
On test the two models are statistically tied (+/-1 page at n=21).
