# Data

Raw images are deliberately excluded from Git because Where's Wally/Waldo
artwork may be protected independently of the dataset annotations.

## Primary dataset

The main experiments use **Wally-Finder v5** from Roboflow Universe:

- URL: https://universe.roboflow.com/wheres-wally/wally-finder/dataset/5
- Task: object detection
- Published classes: `Wally`, `Wanda`, `Wizard`, `Yllaw`
- Published size: 249 images (166 train, 41 validation, 42 test)
- Published license: CC BY 4.0
- Published preprocessing: auto-orient and stretch to 512 x 512

Download the YOLOv8 export and extract it to:

```text
data/roboflow/wally-finder-v5/
```

The extracted directory should contain `data.yaml` plus `train`, `valid`, and
`test` image/label directories. Run the audit and source-group re-split before
training:

```bash
python scripts/audit_dataset.py --dataset data/roboflow/wally-finder-v5
python scripts/create_grouped_split.py \
  --dataset data/roboflow/wally-finder-v5 \
  --output data/processed/wally-finder-grouped
python scripts/audit_dataset.py --dataset data/processed/wally-finder-grouped
```

The published Roboflow split contains differently exported images sharing the
same source filename stem across train, validation and test. The grouped split
keeps every `.rf.` source ID in exactly one split and is the only split used for
reported performance.

## Secondary dataset

The Hey-Waldo dataset is used only for external qualitative testing and the
classical visual-search baseline. It contains 19 original images and patch-level
labels, but it is not treated as precise object-detection ground truth because
its author states that some look-alikes were labelled as Waldo.

Source: https://github.com/vc1492a/Hey-Waldo

Local location:

```text
data/raw/hey-waldo/
```

## Data governance

- Do not commit raw images, generated tiles, or downloaded archives.
- Do not publish screenshots from the artwork without attribution and a clear
  academic-purpose justification.
- Commit scripts, manifests, aggregate statistics, and reproducible download
  instructions instead.
- Record the exact dataset version and download date in experiment metadata.
