# Poster deliverables - manifest

Reported model: **waldo_book_decoy**, page hit-rate **0.76 [0.55, 0.89]** on
held-out books b04 + b09.

## Documents
- `README.md` - project front page, locked 0.76 result.
- `PROJECT_OVERVIEW.md` - full project: pipeline, components, models, web demo, limitations.
- `METRICS.md` - every metric: headline, lineage, decoy ablation, training/val, model comparison, OOD test, failure analysis, inference settings.
- `eval_summary.md` - headline + lineage + Wilson CIs + Wenda caveat + val-vs-test footnote.
- `dataset_stats.md` - final-model dataset, decoy source, and the multi-scale dataset.
- `SOURCES.md` - provenance for data and tools (decoy dataset, book pages, YOLOv8, WBF, SAHI).

## Data / results
- `eval_headtohead.csv` - per-page hit / IoU / confidence for both models on the 21 test pages.
- `waldo_book_decoy_results.csv` - reported model training curve (100 epochs, best val mAP50 0.752 @ ep92).
- `waldo_book_ms_results.csv` - multi-scale model training curve (88 epochs, best val mAP50 0.813 @ ep68).

## Weights
- `weights/waldo_book_decoy_best.pt` - reported model.
- `weights/waldo_book_ms_best.pt` - multi-scale experiment model.

## Figures
- `figures/box_size_histogram.png` - Waldo target size (28-356 px, mean 102 px).
- `figures/multiscale_success_b04_p005.jpg` - a clean multi-scale detection.
- `figures/failure_camouflaged_b09_p030.jpg` - a camouflaged failure page.
- `figures/decoy_vs_ms_b04_p014.jpg` - the one page where the two models differ.

## Result images (per model)
- `result_images/` - input and output for every model (folders 1_yolov8n,
  2_waldo_synth_a, 3_waldo_book_a, 4_waldo_book_decoy, 5_waldo_book_ms).
  See `result_images/README.md` for the mapping and notes.
