"""check_model_metrics.py - the refresh's model regression gate."""

from __future__ import annotations

from check_model_metrics import LOG_LOSS_RISE, VALUE_R2_DROP, compare


def _m(r2, pure, hybrid=0.974):
    return {"value": {"r2_log": r2},
            "outcome": {"pure": {"log_loss": pure}, "hybrid": {"log_loss": hybrid}}}


def test_small_moves_pass():
    assert compare(_m(0.887, 0.977), _m(0.885, 0.979)) == []


def test_better_models_pass():
    assert compare(_m(0.887, 0.977), _m(0.91, 0.960)) == []


def test_value_r2_drop_past_tolerance_fails():
    fails = compare(_m(0.887, 0.977), _m(0.887 - VALUE_R2_DROP - 0.001, 0.977))
    assert len(fails) == 1 and "value" in fails[0]


def test_log_loss_rise_past_tolerance_fails():
    fails = compare(_m(0.887, 0.977), _m(0.887, 0.977 + LOG_LOSS_RISE + 0.001, 0.990))
    assert {f.split()[1] for f in fails} == {"pure", "hybrid"}


def test_missing_section_is_skipped_not_failed():
    assert compare({"value": {"r2_log": 0.887}}, _m(0.887, 0.977)) == []
