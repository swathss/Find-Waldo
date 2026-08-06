from pathlib import Path

from PIL import Image

from src.merge_datasets import (
    Sample,
    link_duplicate_groups,
    normalized_group,
    remap_yolo_labels,
    split_group_names,
)


def test_normalized_group_links_the_same_numeric_page_across_sources():
    assert normalized_group("13.jpg", "HereIsWally") == "page:13"
    assert normalized_group("13_0_2.jpg", "Hey-Waldo") == "page:13"
    assert (
        normalized_group(
            "13_jpeg.rf.80a85a886888763509ffc622dc8b825c.jpg",
            "Wally-Finder-v5",
        )
        == "page:13"
    )


def test_normalized_group_keeps_exported_variants_together():
    first = normalized_group(
        "Group-9_png.rf.0c3e88919903fcd06a353b5472e69712.jpg",
        "Wally-Finder-v5",
    )
    second = normalized_group(
        "Group-9_png.rf.ffffffffffffffffffffffffffffffff.jpg",
        "Wally-Finder-v5",
    )
    assert first == second


def test_remap_yolo_labels_keeps_only_wally(tmp_path: Path):
    labels = tmp_path / "sample.txt"
    labels.write_text(
        "0 0.5 0.5 0.1 0.2\n"
        "1 0.4 0.4 0.2 0.2\n"
        "0 0.2 0.3 0.1 0.1\n",
        encoding="utf-8",
    )
    assert remap_yolo_labels(labels, waldo_class_id=0) == (
        "0 0.500000 0.500000 0.100000 0.200000",
        "0 0.200000 0.300000 0.100000 0.100000",
    )


def test_remap_yolo_polygon_to_bounding_box(tmp_path: Path):
    labels = tmp_path / "polygon.txt"
    labels.write_text(
        "0 0.2 0.1 0.6 0.1 0.6 0.5 0.2 0.5\n",
        encoding="utf-8",
    )
    assert remap_yolo_labels(labels) == (
        "0 0.400000 0.300000 0.400000 0.400000",
    )


def test_split_assignment_never_splits_a_group():
    placeholder = Path("unused.jpg")
    groups = {
        f"page:{index}": [
            Sample(
                image=placeholder,
                labels=("0 0.5 0.5 0.1 0.1",) if index % 2 else (),
                source="test",
                group=f"page:{index}",
            ),
            Sample(
                image=placeholder,
                labels=(),
                source="test-variant",
                group=f"page:{index}",
            ),
        ]
        for index in range(20)
    }
    assignment = split_group_names(groups, seed=42)

    assert set(assignment) == set(groups)
    assert set(assignment.values()) == {"train", "val", "test"}
    assert all(assignment[sample.group] == assignment[group] for group, samples in groups.items() for sample in samples)


def test_pixel_duplicates_link_differently_named_groups(tmp_path: Path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    image = Image.new("RGB", (8, 8), (255, 0, 0))
    image.save(first)
    image.save(second, compress_level=9)
    samples = [
        Sample(first, (), "source-a", "page:3"),
        Sample(second, (), "source-b", "page:6"),
    ]
    aliases = link_duplicate_groups(samples)
    assert aliases["page:3"] == aliases["page:6"]
