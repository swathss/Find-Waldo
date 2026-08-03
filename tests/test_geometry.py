from waldo_ai.geometry import Box, iou, non_max_suppression, tile_origins


def test_iou_identical_boxes() -> None:
    box = Box("a.jpg", 0, 10, 10, 30, 30)
    assert iou(box, box) == 1.0


def test_iou_disjoint_boxes() -> None:
    first = Box("a.jpg", 0, 0, 0, 10, 10)
    second = Box("a.jpg", 0, 20, 20, 30, 30)
    assert iou(first, second) == 0.0


def test_nms_is_class_aware() -> None:
    boxes = [
        Box("a.jpg", 0, 0, 0, 10, 10, 0.9),
        Box("a.jpg", 0, 1, 1, 11, 11, 0.8),
        Box("a.jpg", 1, 1, 1, 11, 11, 0.7),
    ]
    kept = non_max_suppression(boxes, 0.5)
    assert len(kept) == 2
    assert {box.class_id for box in kept} == {0, 1}


def test_tile_origins_cover_right_edge() -> None:
    origins = tile_origins(512, 256, 64)
    assert origins == [0, 192, 256]
    assert origins[-1] + 256 == 512

