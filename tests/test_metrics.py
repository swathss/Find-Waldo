from waldo_ai.geometry import Box
from waldo_ai.metrics import evaluate


def test_perfect_prediction() -> None:
    truth = [Box("page.jpg", 0, 10, 10, 30, 30)]
    predictions = [Box("page.jpg", 0, 10, 10, 30, 30, 0.9)]
    per_class, errors, summary = evaluate(truth, predictions, {0: "Wally"})
    assert per_class.loc[0, "precision"] == 1.0
    assert per_class.loc[0, "recall"] == 1.0
    assert summary["page_success_rate"] == 1.0
    assert errors.loc[0, "outcome"] == "tp"


def test_wrong_class_is_false_positive_and_false_negative() -> None:
    truth = [Box("page.jpg", 0, 10, 10, 30, 30)]
    predictions = [Box("page.jpg", 1, 10, 10, 30, 30, 0.9)]
    _, errors, summary = evaluate(truth, predictions, {0: "Wally", 1: "Wanda"})
    assert set(errors["outcome"]) == {"fp", "fn"}
    assert summary["page_success_rate"] == 0.0

