# Dataset inventory and use

All raw images are copyrighted *Where's Waldo / Where's Wally* illustrations.
They are kept out of Git and used only for non-commercial academic research.
Dataset and annotation licences (ODbL, MIT, CC BY 4.0) cover the annotation or
compilation layer, not the underlying artwork, which remains copyright Martin
Handford and the respective publishers.

## What the reported model trains on

The reported model (`waldo_book_decoy`) uses only two sources:

| Source | Role | Local inventory | Licence |
|---|---|---|---|
| Book pages (our labelling of 12 Handford books) | primary positives and all evaluation | 12 books, 128 labelled pages (119 with Waldo, 9 empty negatives), 120 boxes | Labels are ours; book art is copyright Handford / publisher; kept local |
| where-s-waldo-zu227 (Roboflow, obtained via a Kaggle mirror `kaggle_mohaneddz`) | Odlaw and Wizard crops used as decoy hard-negatives | 5 classes; 226 Odlaw + 293 Wizard crops extracted and re-composited | CC BY 4.0 (annotation layer); art copyright Handford |

- The book pages are sliced into 640 px tiles at native resolution. Tiles with
  Waldo are positives; other tiles are background negatives.
- Odlaw and Wizard crops are pasted onto Waldo-free tiles with empty labels, so
  the model learns Waldo specifically, not any red/white or striped pattern.
  Wilma and woof from the 5-class set are excluded (Wilma is too Waldo-like;
  woof is a red/white dog with noisy labels).

## Build the training data

```bash
python scripts/build_book_dataset.py   # book pages -> data/book_yolo (640 tiles, split by book)
python scripts/build_decoys.py         # Odlaw/Wizard crops -> data/decoy_neg + data/book_decoy.yaml
```

Because the raw scans and labels are kept local, reproducing this needs the book
pages under `data/raw/book_pages/` (images + YOLO labels) and the 5-class set
under `data/raw/kaggle_mohaneddz/`.

## Leakage control

The split is grouped by book, so no book appears in more than one split:

- test: `b04` (The Great Waldo Search), `b09` (Where's Waldo?)
- validation: `b11`, `b16`
- train: the remaining eight books

This tests cross-book generalization. Decoy negatives are added to the train
split only; validation and test are real tiles from the held-out books.

## Datasets used in earlier experiments (not in the reported model)

These fed the first one-class dataset (`data/processed_v2`) and the synthetic
compositing pipeline in `synth/` (Track A). Track A did not transfer to real
pages (about 0.00 hit-rate on held-out pages), so the reported model dropped
these in favour of the real book-page tiles above. They are documented here for
history and reproducibility.

| Source | Earlier use | Published licence |
|---|---|---|
| [Hey-Waldo](https://github.com/vc1492a/Hey-Waldo) | hard-negative background patches | ODbL 1.0 (database) |
| [HereIsWally](https://github.com/tadejmagajna/HereIsWally) | real Waldo crops | MIT (repository) |
| [Wally-Finder v5](https://universe.roboflow.com/wheres-wally/wally-finder/dataset/5) | Wally polygons converted to boxes | CC BY 4.0 |
| Kaggle sets (`ffaraz`, `bakshi15`, `residentmario`, `mohaneddz`) | explored; `mohaneddz` is the current decoy source | varies; some report Unknown on Kaggle |

The earlier merge (`src/merge_datasets.py`) normalized Roboflow variants such as
`13_jpg.rf.<hash>.jpg`, Hey-Waldo patches such as `13_0_2.jpg`, and HereIsWally's
`13.jpg` to a single `page:13` group so a whole source page landed in only one
split, and wrote `data/processed_v2/manifest.csv`. That leakage-safe grouping is
kept for the earlier processed dataset; the reported model instead groups by
book, which is simpler and stricter.
