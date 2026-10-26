"""Smoke test of the judge study (task 17): one small pair with a fast stand-in engine, and the analysis."""

import pytest

from research import judge_study as js
from tests.test_critic_study import LogisticEngine


def test_pairs_and_seeds():
    tuning, reporting = js.study_pairs(js.TUNING_SEEDS), js.study_pairs(js.REPORTING_SEEDS)
    assert len(tuning) == len(reporting) == len(js.PAIRS) * 6
    assert {p.split for p in tuning} == {"tuning"} and {p.split for p in reporting} == {"reporting"}
    assert len({p.generator_seed for p in tuning + reporting}) == len(tuning) + len(reporting)


@pytest.mark.slow
def test_one_pair_gives_a_row_per_setting(tmp_path):
    pair = next(p for p in js.study_pairs([0]) if p.kind == "drop_two_informative")
    rows = js.run_pair(pair, work_dir=tmp_path, engine=LogisticEngine(), n_jobs=1, fresh_rows=500)
    assert len(rows) == len(js.REPEATS) * len(js.ALPHAS)
    assert {(r["repeats"], r["alpha"]) for r in rows} == {(r, a) for r in js.REPEATS for a in js.ALPHAS}
    # One true difference for the pair, measured on the fresh sample (whatever its sign).
    assert len({r["true_difference"] for r in rows}) == 1
    assert rows[0]["true_difference"] == pytest.approx(rows[0]["fresh_f1_b"] - rows[0]["fresh_f1_a"])
    assert all(r["verdict"] in ("keep", "keep_simpler", "discard") for r in rows)
    # More repeats use more folds: the 5-repeat Judge sees all 25 folds' differences.
    assert {r["repeats"]: r for r in rows}[5]["std_difference"] is not None


def fake_rows(split, verdicts_by_setting, true_differences):
    rows = []
    for (repeats, alpha), verdicts in verdicts_by_setting.items():
        for n, (verdict, true) in enumerate(zip(verdicts, true_differences)):
            rows.append({"pair_id": f"{split}-{n}", "split": split, "repeats": repeats, "alpha": alpha,
                         "failed": False, "verdict": verdict, "true_difference": true})
    return rows


def test_rates_and_the_choice_of_defaults():
    truths = [-0.02, -0.02, -0.02, -0.02, 0.03, 0.03, 0.0]  # 4 harmful, 2 helpful, 1 neutral
    settings = {(r, a): ["discard"] * 7 for r in js.REPEATS for a in js.ALPHAS}
    settings[(1, 0.20)] = ["keep", "discard", "discard", "discard", "keep", "keep", "keep"]  # 1 of 4 false keeps
    settings[(3, 0.10)] = ["discard", "discard", "discard", "discard", "keep", "keep", "discard"]  # perfect
    settings[(5, 0.10)] = ["discard", "discard", "discard", "discard", "keep", "keep", "discard"]  # perfect, dearer
    rates = {(r["repeats"], r["alpha"]): r for r in js.rates(fake_rows("tuning", settings, truths))}
    assert rates[(1, 0.20)]["false_keep_rate"] == 0.25 and rates[(1, 0.20)]["power"] == 1.0
    assert rates[(3, 0.10)]["false_keep_rate"] == 0.0 and rates[(3, 0.10)]["power"] == 1.0
    assert rates[(1, 0.05)]["power"] == 0.0 and rates[(1, 0.05)]["harmful"] == 4
    chosen = js.choose(list(rates.values()))
    assert (chosen["repeats"], chosen["alpha"]) == (3, 0.10)  # fewer repeats win the tie with 5
    summary = js.summary(fake_rows("tuning", settings, truths) + fake_rows("reporting", settings, truths))
    assert (summary["chosen_on_reporting"]["repeats"], summary["chosen_on_reporting"]["alpha"]) == (3, 0.10)


def test_when_no_setting_is_safe_enough_the_safest_is_chosen():
    truths = [-0.02, -0.02, 0.03]
    settings = {(r, a): ["keep", "keep", "keep"] for r in js.REPEATS for a in js.ALPHAS}
    settings[(5, 0.05)] = ["keep", "discard", "discard"]  # false-keep rate 0.5, the lowest
    chosen = js.choose(js.rates(fake_rows("tuning", settings, truths)))
    assert (chosen["repeats"], chosen["alpha"]) == (5, 0.05) and chosen["false_keep_rate"] == pytest.approx(0.5)
