import json

import pytest

from run_experiments import Config, mean_std, run_phase1

TINY = Config(n_train=64, n_test=40, n_gen=20, epochs=3, seeds=(0, 1))


def test_mean_std_formats_over_seeds():
    assert mean_std([0.5, 0.7]) == (pytest.approx(0.6), pytest.approx(0.1))


def test_phase1_writes_json_tables_and_the_object_count_plot(tmp_path):
    results = run_phase1(TINY, tmp_path)
    saved = json.loads((tmp_path / "phase1.json").read_text())
    assert saved["config"]["epochs"] == 3
    for model in ("ours", "baseline"):
        runs = saved["runs"][model]
        assert len(runs) == 2
        assert set(runs[0]["test"]) == {"overall", "simple", "complex"}
        assert set(runs[0]["generalisation"]) == {str(n) for n in range(3, 11)}
        assert set(runs[0]["test"]["overall"]) == {"iou", "iou_m", "subject_acc", "placement_acc"}
    assert "ours_no_curriculum" in saved["runs"]
    assert results["runs"]["ours"][0]["test"]["overall"]["iou"] == saved["runs"]["ours"][0]["test"]["overall"]["iou"]
    assert (tmp_path / "fig4a_objects.png").stat().st_size > 1000
    assert "| Ours" in (tmp_path / "results_phase1.md").read_text()
