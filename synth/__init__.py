"""Synthetic Waldo scene engine.

The core idea of this project: a detector can only find Waldo in *unseen*,
arbitrarily-styled "Where's Waldo" images if it has been trained on a large,
diverse variety of Waldo appearances and clutter. The ~250 real labelled
images cannot supply that variety, so we manufacture it.

Pipeline:
    foregrounds.py  mine real Waldo boxes -> alpha (RGBA) cut-outs
    compositor.py   paste cut-outs onto crowd backgrounds with heavy
                    augmentation + red/white distractors -> image + YOLO box
    generate.py     produce an arbitrary number of labelled synthetic tiles
"""
