import pandas as pd

from scaling_piml.analysis import build_pilot_summary


def test_build_pilot_summary_marks_ready_when_trends_are_monotonic():
    grouped_df = pd.DataFrame(
        [
            {
                "model_name": "plain",
                "is_physics_informed": False,
                "capacity_name": "tiny",
                "hidden_widths": "32,32",
                "parameter_count": 100,
                "dataset_size": 64,
                "test_rel_l2_mean": 0.40,
            },
            {
                "model_name": "plain",
                "is_physics_informed": False,
                "capacity_name": "tiny",
                "hidden_widths": "32,32",
                "parameter_count": 100,
                "dataset_size": 128,
                "test_rel_l2_mean": 0.30,
            },
            {
                "model_name": "plain",
                "is_physics_informed": False,
                "capacity_name": "small",
                "hidden_widths": "64,64",
                "parameter_count": 200,
                "dataset_size": 64,
                "test_rel_l2_mean": 0.35,
            },
            {
                "model_name": "plain",
                "is_physics_informed": False,
                "capacity_name": "small",
                "hidden_widths": "64,64",
                "parameter_count": 200,
                "dataset_size": 128,
                "test_rel_l2_mean": 0.25,
            },
        ]
    )

    aggregate_df = pd.DataFrame(
        [
            {"status": "success", "diverged": False, "nan_detected": False, "best_epoch": 10},
            {"status": "success", "diverged": False, "nan_detected": False, "best_epoch": 12},
            {"status": "success", "diverged": False, "nan_detected": False, "best_epoch": 9},
            {"status": "success", "diverged": False, "nan_detected": False, "best_epoch": 11},
        ]
    )

    summary = build_pilot_summary(grouped_df, aggregate_df)

    assert summary["gate"]["ready_for_full_sweep"] is True
    assert summary["gate"]["all_error_vs_D_checks_pass"] is True
    assert summary["gate"]["all_error_vs_N_checks_pass"] is True


def test_build_pilot_summary_detects_capacity_reversal():
    grouped_df = pd.DataFrame(
        [
            {
                "model_name": "plain",
                "is_physics_informed": False,
                "capacity_name": "tiny",
                "hidden_widths": "32,32",
                "parameter_count": 100,
                "dataset_size": 64,
                "test_rel_l2_mean": 0.30,
            },
            {
                "model_name": "plain",
                "is_physics_informed": False,
                "capacity_name": "small",
                "hidden_widths": "64,64",
                "parameter_count": 200,
                "dataset_size": 64,
                "test_rel_l2_mean": 0.40,
            },
            {
                "model_name": "plain",
                "is_physics_informed": False,
                "capacity_name": "tiny",
                "hidden_widths": "32,32",
                "parameter_count": 100,
                "dataset_size": 128,
                "test_rel_l2_mean": 0.25,
            },
            {
                "model_name": "plain",
                "is_physics_informed": False,
                "capacity_name": "small",
                "hidden_widths": "64,64",
                "parameter_count": 200,
                "dataset_size": 128,
                "test_rel_l2_mean": 0.35,
            },
        ]
    )

    aggregate_df = pd.DataFrame(
        [
            {"status": "success", "diverged": False, "nan_detected": False, "best_epoch": 10},
            {"status": "success", "diverged": False, "nan_detected": False, "best_epoch": 12},
            {"status": "success", "diverged": False, "nan_detected": False, "best_epoch": 9},
            {"status": "success", "diverged": False, "nan_detected": False, "best_epoch": 11},
        ]
    )

    summary = build_pilot_summary(grouped_df, aggregate_df, relative_tolerance=0.01)

    assert summary["gate"]["ready_for_full_sweep"] is False
    assert summary["gate"]["all_error_vs_N_checks_pass"] is False