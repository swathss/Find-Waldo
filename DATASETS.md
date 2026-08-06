# Dataset inventory and use

The project uses three public dataset releases. Raw images are downloaded to
`data/` and are intentionally excluded from Git.

| Source | Local inventory | Project use | Published licence |
|---|---:|---|---|
| [Hey-Waldo](https://github.com/vc1492a/Hey-Waldo) | 317 colour 256 px patches (31 positive, 286 negative) | Hard-negative backgrounds; positive classification patches are not treated as detection boxes | ODbL 1.0 for the database |
| [HereIsWally](https://github.com/tadejmagajna/HereIsWally) | 36 labelled scenes, 43 boxes | Real Waldo crops for training, validation, and testing | MIT repository licence |
| [Wally-Finder v5](https://universe.roboflow.com/wheres-wally/wally-finder/dataset/5) | 249 images, 107 character polygons; 49 are Wally | Wally polygons converted to boxes, plus other-character-only hard negatives | CC BY 4.0 |

The datasets contain copyrighted *Where's Wally/Waldo* illustrations. Their
database or repository licences do not necessarily grant copyright in every
underlying illustration. Keep the raw images out of Git and use them only in
accordance with the source terms and the project's academic context.

## Reproduce the local collection

```bash
python src/download_dataset.py
```

The command clones Hey-Waldo and HereIsWally. Download the Wally-Finder v5
YOLOv8 export from the link above and extract it to:

```text
data/roboflow/wally-finder-v5/
```

Then build the unified one-class YOLO dataset:

```bash
python src/merge_datasets.py
```

This produces `data/processed_v2/data.yaml` and a `manifest.csv` recording the
source, page group, split, output path, and number of Waldo boxes for every
sample.

## Leakage control

Roboflow file names such as `13_jpg.rf.<hash>.jpg` are variants of one source
page. Hey-Waldo patches such as `13_0_2.jpg` and HereIsWally's `13.jpg` may come
from that same page too. `src/merge_datasets.py` normalizes all of them to
`page:13` and assigns the complete group to exactly one of train, validation,
or test. Non-numeric Roboflow variants are grouped by their pre-export name.
Decoded pixel hashes also link differently named groups when their image
content is identical; this catches duplicate source-page numbering errors.

The published Wally-Finder split is not used directly because 34 normalized
page groups occur in more than one of its train/valid/test folders.

## Sources not enabled

The repository previously mentioned `ffaraz/waldo-yolo` and
`bakshi15/waldo-dataset-v3`. Kaggle currently reports their licence as
`Unknown`, so the collection script does not download or merge them. Add them
only after the team has confirmed appropriate terms and checked for duplicate
source pages.
