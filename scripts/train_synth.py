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
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--name", default="waldo_synth")
    ap.add_argument("--close-mosaic", type=int, default=10)
    ap.add_argument("--resume", action="store_true", help="resume from the run's last.pt")
    args = ap.parse_args()

    device = pick_device()

    if args.resume:
        last = ROOT / "models" / args.name / "weights" / "last.pt"
        if not last.exists():
            raise FileNotFoundError(f"{last} missing - nothing to resume")
        print(f"resuming {last} on {device}")
        YOLO(str(last)).train(resume=True)
        print("best:", ROOT / "models" / args.name / "weights" / "best.pt")
        return

    data = Path(args.data)
    if not data.exists():
        raise FileNotFoundError(f"{data} missing")

    print(f"training {args.model} on {device}, {args.epochs} epochs")

    model = YOLO(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name=args.name,
        project=str(ROOT / "models"),
        patience=20,
        hsv_h=0.02, hsv_s=0.5, hsv_v=0.4,
        degrees=15, translate=0.15, scale=0.6, fliplr=0.5,
        # turned mixup off, it was blending the small waldo targets together
        mosaic=1.0, mixup=0.0, copy_paste=0.2,
        close_mosaic=args.close_mosaic,
        exist_ok=True,
    )
    print("best:", ROOT / "models" / args.name / "weights" / "best.pt")


if __name__ == "__main__":
    main()
