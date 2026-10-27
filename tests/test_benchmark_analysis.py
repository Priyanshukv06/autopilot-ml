"""Tests for the benchmark analysis (spec task 26): which rows are scored, complete blocks, the families of
comparisons, time-to-target and the written tables and charts, on hand-made rows."""

import pandas as pd
import pytest

from research import benchmark, benchmark_analysis as ba
from research import stat_tests as st

DATASETS = ["d1", "d2", "d3", "d4"]
QUALITY = {"manual": 0.60, "flaml": 0.70, "autogluon": 0.72, "autopilot_rules": 0.71, "autopilot_full": 0.74,
           "random_choice": 0.69, "no_critic": 0.68, "no_resource_manager": 0.73}


def row(system, dataset, budget, seed, score, status="completed", metric="f1", wall=None):
    return {"cell_id": benchmark.Cell(system, dataset, budget, seed).cell_id, "system": system, "dataset": dataset,
            "metric": metric, "budget_seconds": budget, "seed": seed, "status": status,
            "test_score": score if status == "completed" else None,
            "wall_seconds": wall if wall is not None else float(budget), "n_experiments": 3, "error": None}


def grid_rows():
    """Every system of the default grid on 4 datasets and 2 seeds; scores grow with the budget."""
    rows = []
    for cell in benchmark.grid(DATASETS, seeds=[1, 2]):
        bonus = 0.02 if cell.budget_seconds == 300 else 0.0
        noise = 0.001 * (DATASETS.index(cell.dataset) + cell.seed)
        rows.append(row(cell.system, cell.dataset, cell.budget_seconds, cell.seed,
                        QUALITY[cell.system] + bonus + noise))
    return rows


def test_only_the_last_completed_row_of_each_cell_is_scored():
    rows = [row("flaml", "d1", 120, 1, None, status="failed"), row("flaml", "d1", 120, 1, 0.7),
            row("manual", "d1", 120, 1, 0.5), row("manual", "d1", 120, 1, None, status="failed"),
            row("autogluon", "d1", 120, 1, 2.5, metric="rmse")]
    frame = ba.scored(rows).set_index("system")
    assert set(frame.index) == {"flaml", "autogluon"}  # manual's last row failed
    assert frame.loc["autogluon", "signed"] == -2.5 and frame.loc["flaml", "signed"] == 0.7
    assert list(ba.failures(rows)["cell_id"]) == ["manual-d1-120s-s1"]
    assert ba.scored([]).empty


def test_blocks_keep_only_pairs_where_every_system_completed():
    rows = [row(s, d, 120, seed, 0.5 + 0.01 * seed) for s in ("flaml", "manual") for d in ("d1", "d2")
            for seed in (1, 2)]
    rows[-1] = row("manual", "d2", 120, 2, None, status="failed")
    table = ba.blocks(ba.scored(rows), 120, ["flaml", "manual"])
    assert list(table.index) == [("d1", 1), ("d1", 2), ("d2", 1)]
    means = ba.dataset_means(table)
    assert means.loc["d1", "flaml"] == pytest.approx(0.515) and means.loc["d2", "manual"] == pytest.approx(0.51)


def test_the_families_of_comparisons():
    result = ba.analyse(grid_rows())
    tests = result["tests"].set_index(["family", "budget_seconds"])
    assert list(tests.index) == [("main", 120), ("main", 300), ("ablations", 300)]
    assert tests.loc[("main", 120), "datasets"] == 4 and tests.loc[("main", 120), "blocks"] == 8
    assert tests.loc[("ablations", 300), "systems"] == 5
    ranks = result["ranks"].query("family == 'main' and budget_seconds == 120").set_index("system")["average_rank"]
    assert ranks.to_dict() == {"autopilot_full": 1.0, "autogluon": 2.0, "autopilot_rules": 3.0, "flaml": 4.0,
                               "manual": 5.0}
    assert tests.loc[("main", 120), "nemenyi_cd"] == pytest.approx(st.nemenyi_cd(5, 4))

    pairs = result["pairs"]
    main = pairs.query("family == 'main' and budget_seconds == 120")
    assert list(zip(main["system_a"], main["system_b"])) == ba.FAMILIES["main"]["pairs"]
    first = main.iloc[0]
    assert (first["wins"], first["losses"], first["block_wins"]) == (4, 0, 8)
    assert first["p_value"] == pytest.approx(2 / 2 ** 4) and first["p_holm"] == pytest.approx(min(1, 5 * 2 / 16))
    ablation = pairs.query("family == 'ablations'")
    assert set(ablation["system_b"]) == {"autopilot_rules", *benchmark.ABLATIONS}
    assert ablation.set_index("system_b").loc["no_resource_manager", "median_difference"] == pytest.approx(0.01)


def test_time_to_target_of_the_main_systems():
    result = ba.analyse(grid_rows())
    summary = result["time_to_target_summary"].set_index("system")
    # the best score of each pair is autopilot_full's at 300 s (0.76 + noise); 99% of it is about 0.752
    assert summary.loc["autopilot_full", "reached_by_120s"] == 0 and summary.loc["autopilot_full", "reached_by_300s"] == 8
    assert summary.loc["manual", "never"] == 8 and pd.isna(summary.loc["manual", "median_time_to_target"])
    assert list(summary.index) == list(benchmark.MAIN_SYSTEMS)
    assert set(result["time_to_target"]["system"]) == set(benchmark.MAIN_SYSTEMS)  # no ablation


def test_the_tables_and_charts_are_written(tmp_path):
    result = ba.analyse(grid_rows())
    paths = ba.write(result, tmp_path)
    names = {p.name for p in paths}
    assert {"analysis_tests.csv", "analysis_pairs.csv", "analysis_per_dataset.csv", "score_against_budget.png",
            "cd_main_120s.png", "cd_main_300s.png", "cd_ablations_300s.png"} <= names
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
    per_dataset = pd.read_csv(tmp_path / "analysis_per_dataset.csv")
    assert set(per_dataset["seeds"]) == {2} and len(per_dataset) == 4 * (5 * 2 + 3)


def test_a_smoke_sized_analysis_does_not_fail():
    rows = [row(c.system, c.dataset, c.budget_seconds, c.seed, QUALITY[c.system]) for c in benchmark.smoke_grid()]
    result = ba.analyse(rows)
    tests = result["tests"].set_index(["family", "budget_seconds"])
    assert tests.loc[("main", 120), "datasets"] == 1 and pd.isna(tests.loc[("main", 120), "friedman_p"])
    assert tests.loc[("ablations", 300), "datasets"] == 0  # the smoke run has no 300 s cells
    assert len(result["pairs"].query("family == 'main'")) == 5
