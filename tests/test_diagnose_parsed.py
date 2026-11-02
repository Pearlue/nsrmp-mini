from diagnose_parsed_baseline import ROWS, Settings, run_all


def test_every_diagnostics_row_runs_and_reports_its_metrics():
    tiny = Settings(n_train=48, n_test=24, epochs_short=1, epochs_long=1, seeds=(0,))
    results = run_all(tiny)
    assert [r["row"] for r in results["rows"]] == [name for name, *_ in ROWS]
    for r in results["rows"]:
        assert set(r["per_seed"][0]) >= {"seed", "iou_m_simple", "iou_m_complex", "subject_acc_simple", "subject_acc_complex"}
    assert set(results["mechanism"]) == {"entropy_by_epoch", "gold_weight_by_epoch", "same_object_fraction",
                                         "gold_gradient_favoured_fraction"}
