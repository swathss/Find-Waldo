from pathlib import Path

from PIL import Image

from waldo_ai.yolo_io import read_yolo_labels


def test_polygon_label_is_converted_to_bounding_box(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    image_path = image_dir / "sample.jpg"
    Image.new("RGB", (100, 200)).save(image_path)
    (label_dir / "sample.txt").write_text("0 0.1 0.2 0.5 0.1 0.6 0.8 0.2 0.9\n")
    box = read_yolo_labels(image_path)[0]
    assert (box.x1, box.y1, box.x2, box.y2) == (10.0, 20.0, 60.0, 180.0)

