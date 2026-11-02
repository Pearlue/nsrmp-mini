import json

import pytest

from run_experiments import Config, mean_std, run_noise, run_phase1, run_phase2

TINY = Config(n_train=64, n_test=40, n_gen=20, epochs=3, seeds=(0, 1))


def test_mean_std_uses_the_sample_std_over_seeds():
    assert mean_std([0.5, 0.7]) == (pytest.approx(0.6), pytest.approx(0.1414, abs=1e-4))


def test_phase1_writes_json_tables_and_the_object_count_plot(tmp_path):
    results = run_phase1(TINY, tmp_path)
    saved = json.loads((tmp_path / "phase1.json").read_text())
    assert saved["config"]["epochs"] == 3
    for model in ("ours", "baseline", "parsed_baseline"):
        runs = saved["runs"][model]
        assert len(runs) == 2
        assert set(runs[0]["test"]) == {"overall", "simple", "complex"}
        assert set(runs[0]["generalisation"]) == {str(n) for n in range(3, 11)}
        assert set(runs[0]["test"]["overall"]) == {"iou", "iou_m", "subject_acc", "placement_acc"}
    assert "ours_no_curriculum" in saved["runs"]
    assert results["runs"]["ours"][0]["test"]["overall"]["iou"] == saved["runs"]["ours"][0]["test"]["overall"]["iou"]
    assert (tmp_path / "fig4a_objects.png").stat().st_size > 1000
    table = (tmp_path / "results_phase1.md").read_text()
    assert "| Ours" in table and "| Parse-matched neural baseline (NMN+-like)" in table
    assert "| End-to-end GRU baseline" in table


def test_phase2_trains_on_one_or_two_steps_and_evaluates_one_to_five(tmp_path):
    run_phase2(TINY, tmp_path)
    saved = json.loads((tmp_path / "phase2.json").read_text())
    assert set(saved["runs"]) == {"ours", "baseline", "parsed_baseline"}
    assert set(saved["runs"]["ours"][0]["generalisation"]) == {"1", "2", "3", "4", "5"}
    assert (tmp_path / "fig4b_steps.png").stat().st_size > 1000
    assert "steps=5" in (tmp_path / "results_phase2.md").read_text()


def test_noise_sweep_trains_both_parse_given_models_at_every_noise_level(tmp_path):
    run_noise(TINY, tmp_path, levels=(0.1, 0.5))
    saved = json.loads((tmp_path / "noise.json").read_text())
    assert set(saved["runs"]) == {"ours", "parsed_baseline"}
    for model in saved["runs"].values():
        assert set(model) == {"0.1", "0.5"}
        assert [r["seed"] for r in model["0.5"]] == [0, 1]
        assert set(model["0.5"][0]["test"]) == {"overall", "simple", "complex"}
    assert "σ=0.5" in (tmp_path / "results_noise.md").read_text()
