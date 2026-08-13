# Dataset stats - final model (waldo_book_decoy)

Training data = real book-page tiles (`data/book_yolo`) plus decoy hard-negatives
(`data/decoy_neg`). Split grouped by book so no book crosses splits.

## Source labelling (book_pages)

| Item | Count |
|---|---|
| Books | 12 |
| Labelled pages | 128 (119 with Waldo, 9 empty negatives) |
| Waldo instances (boxes) | 120 |

Note on Wenda: all 120 boxes are labelled single-class as `waldo`. How many are
actually Wenda (who looks almost identical) is **not automatically determinable**
and would need a manual visual review of each crop. Flagged as a data-quality
check, not yet done.

## Tiles per split (640px)

| Split | Positive tiles | Negative tiles | neg:pos | Books |
|---|---|---|---|---|
| train | 219 | 889 (669 background + 220 decoy) | 4.1 | b01,b02,b03,b05,b07,b08,b10,b12 |
| val | 71 | 210 | 3.0 | b11, b16 |
| test | 58 | 174 | 3.0 | b04, b09 |

Decoy negatives (220) are real Odlaw and Wizard crops pasted onto Waldo-free
tiles, train split only. Val and test are real tiles from the held-out books.

## Decoy source (hard negatives)

Odlaw and Wizard crops were mined from a 5-class Roboflow dataset
(where-s-waldo-zu227 v1, mohaneds-workspace, CC BY 4.0) and re-composited as
unlabelled negatives. Wilma and woof were excluded (Wilma is too Waldo-like;
woof is a red/white dog with noisy labels). Full provenance in `SOURCES.md`.

## Second dataset: book_yolo_ms (multi-scale + copy-paste experiment)

Used to train waldo_book_ms. Same book split (test b04/b09, val b11/b16).

| Split | Positive tiles | Negative tiles | neg:pos |
|---|---|---|---|
| train | 1457 | 4371 | 3.0 |
| val | 188 | 572 | 3.0 |
| test | 217 | 655 | 3.0 |

Built with tiles at three scales (320, 512, 768), 700 copy-paste positives
(real Waldo crops pasted onto backgrounds at 28-356 px with rotation,
brightness jitter, feathered edges), and the 220 decoy negatives. This model
tied with the reported model on the test (see METRICS.md), so it is kept as an
experiment, not the reported model.
