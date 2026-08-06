"""
Step 1: Download datasets.

Sources (both free, no login):
  A. Hey-Waldo by Valentino Constantinou
     https://github.com/vc1492a/Hey-Waldo
     64×64 positive patches (Waldo) and negative patches (not-Waldo).

  B. HereIsWally by Tadej Magajna
     https://github.com/tadejmagajna/HereIsWally
     Full Where's Waldo scene images + Pascal-VOC XML bounding boxes.
"""

import os
import subprocess
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

HEY_WALDO_URL = "https://github.com/vc1492a/Hey-Waldo.git"
HERE_IS_WALLY_URL = "https://github.com/tadejmagajna/HereIsWally.git"


def clone_repo(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest.name} already exists")
        return
    print(f"[clone] {url} → {dest}")
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)


def main():
    clone_repo(HEY_WALDO_URL, RAW / "Hey-Waldo")
    clone_repo(HERE_IS_WALLY_URL, RAW / "HereIsWally")

    # Quick inventory
    waldo_patches = list((RAW / "Hey-Waldo" / "64" / "Waldo").glob("*.jpg")) if (RAW / "Hey-Waldo" / "64" / "Waldo").exists() else []
    non_waldo_patches = list((RAW / "Hey-Waldo" / "64" / "NotWaldo").glob("*.jpg")) if (RAW / "Hey-Waldo" / "64" / "NotWaldo").exists() else []
    scenes = list((RAW / "HereIsWally").rglob("*.jpg"))

    print(f"\nDataset inventory:")
    print(f"  Waldo patches   : {len(waldo_patches)}")
    print(f"  Non-Waldo patches: {len(non_waldo_patches)}")
    print(f"  Full scenes     : {len(scenes)}")
    print("\nNext: run  python src/prepare_dataset.py")


if __name__ == "__main__":
    main()
