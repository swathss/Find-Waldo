import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "synth" / "data.yaml"


def pick_device():
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "waldo_synth" / "weights" / "best.pt"))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()

    model = YOLO(args.weights)
    m = model.val(data=str(DATA), split=args.split, device=pick_device(),
                  name="waldo_synth_eval", project=str(ROOT / "models"))
    print(f"\n{args.split} split")
    print(f"mAP50      {m.box.map50:.4f}")
    print(f"mAP50-95   {m.box.map:.4f}")
    print(f"precision  {m.box.mp:.4f}")
    print(f"recall     {m.box.mr:.4f}")


if __name__ == "__main__":
    main()
