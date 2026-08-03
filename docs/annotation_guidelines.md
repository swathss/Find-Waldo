# Annotation and review guidelines

The primary Roboflow labels must be visually reviewed rather than assumed to be
correct.

- Boxes should tightly contain the visible character, including hat and body.
- Use the published four class names consistently.
- Do not relabel a striped background object as Wally.
- Occluded characters should be boxed around the visible extent, consistently
  across the dataset.
- Flag uncertain labels in a separate review CSV; do not silently edit them.
- Corrections must retain the source image ID and original annotation.
- Review all test labels and at least 20% of train/validation labels.

Any manually corrected annotation file is a derived database and must retain
dataset attribution and applicable share-alike terms.

