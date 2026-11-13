# Problem definition and objectives

## Problem statement

The problem is stated as follows. Given a tabular dataset D with a target column y, a task T (binary classification, multiclass classification or regression), a primary metric m and a compute budget B (a time limit, a largest number of Experiments and a number of CPU cores), the system must do two things.

1. Find a pipeline p, meaning the treatment of the data, the features, the model and its hyperparameters, with the best possible score m(p) on test data it has never used for any decision.
2. Reach that pipeline using as little of the budget B as possible.

A conventional AutoML run solves the first part with one search [@wang2021flaml]. AutoPilot-ML works on both parts with a sequence of Experiments e1, e2, ..., ek, where each next Experiment is chosen from the Diagnoses of the earlier ones and from a memory of Experiments on other datasets. The research question of the project is whether an autonomous experimentation layer can improve the efficiency and effectiveness of conventional AutoML by learning from previous experiments, diagnosing model weaknesses, selecting appropriate improvement strategies and allocating computational resources adaptively.

## Objectives

Table @tab:objectives lists the seven objectives set at the start of the work and their status at the end of the project.

::table tab:objectives | Objectives of the project and their status | widths=0.8,6,6 | align=cjj
| No. | Objective | Status at the end of the project |
|---|---|---|
| O1 | A baseline: profiling, task detection, FLAML search, held-out evaluation, explanations, experiment tracking and a dashboard | Built and tested in Phase 1 |
| O2 | A Critic: automatic, structured Diagnoses of each Experiment, with their evidence | Built: 14 kinds; the 12 original kinds measured on data with known answers |
| O3 | A Strategy_Engine: map Diagnoses to candidate changes and rank them by rules, bandits and memory | Built: 12 Strategies, a rule table and four ways to order them; measured in the benchmark |
| O4 | A planner and autonomous loop with a fair comparison of every change over several splits | Built: the Session loop, the Assessor on 15 folds and the Judge; the Judge measured in two studies |
| O5 | An Experiment_Memory that reuses what worked on similar datasets | Built: records written by every Session and read by the learned selectors; effect measured in the benchmark |
| O6 | A Resource_Manager that shares the time and CPU budget and decides when to stop | Built: budgets, a reserve, six stop rules and the utility of each step recorded |
| O7 | An evaluation against a manual baseline, FLAML and AutoGluon, with ablations, on public datasets and several seeds | Done: 390 benchmark runs on 6 OpenML datasets, 5 seeds and 2 budgets |
::end

## Hypotheses and success criteria

The hypotheses were written in October 2026, and the exact criteria for accepting or rejecting them, shown in Table @tab:hypotheses, were written on 28 October 2026, before the benchmark runs that decide them [@demsar2006]. A difference is the per-dataset difference of test scores between two systems, each dataset's score being the mean over its seeds, signed so that a positive value favours the first system. The interval is the 95% bootstrap interval of the median difference over the 6 evaluation datasets [@efron1993bootstrap]. The margin of -0.005 is the same non-inferiority margin the Judge uses.

::table tab:hypotheses | Hypotheses and the criteria fixed before the benchmark | widths=3.6,4.2,4.0,4.0 | align=cccc
| Hypothesis | Comparison | Supported if | Contradicted if |
|---|---|---|---|
| H1: the full system reaches the target score sooner than FLAML | Time-to-target of the full system against FLAML alone | It reaches the target at an earlier budget in more (dataset, seed) pairs than the other way round, and its median time is not later | The other way round |
| H2: without the memory, the loop does worse | Full system against the Session with the rule order, at 300 s | The interval lies above 0 | The interval lies below 0 |
| H3: without strategy selection, the loop does worse | Full system against a random order of the same changes, at 300 s | The interval lies above 0 | The interval lies below 0 |
| H4: the full system is not worse than FLAML at the same budget | Full system against FLAML alone, at 120 s and at 300 s | The interval lies above -0.005 at both budgets | The interval lies below -0.005 at either budget |
| RQ3 and the Critic: adaptive budgets and Diagnoses help | Full system against the runs without the Resource_Manager and without the Critic, at 300 s | The interval lies above 0 | The interval lies below 0 |
::end

In every other case the result is "not shown". The Wilcoxon tests with Holm's correction are reported as well, but with 6 datasets they cannot fall below 0.05 for any pair once two or more pairs are compared, so no claim rests on them alone [@wilcoxon1945; @holm1979].

## Scope and limits

The work has the following scope.

- In scope: single-table tabular data in CSV files; binary classification, multiclass classification and regression; a laptop CPU; fully offline operation.
- Out of scope: images, free text documents, time series and clustering; language models as decision makers; cloud or paid services.
- Limits of the current code: a time budget of 5 to 3,600 seconds for one Experiment, at least 20 labelled rows, and a test share of 0.05 to 0.5. The user chooses the target column, helped by ranked suggestions.
