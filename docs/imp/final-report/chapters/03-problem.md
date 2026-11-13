# Problem definition and objectives

## Problem statement

The problem is stated as follows. Given a tabular dataset D with a target column y, a task T (binary classification, multiclass classification or regression), a primary metric m and a compute budget B (a time limit, a largest number of Experiments and a number of CPU cores), the system must do two things.

1. Find a pipeline p, meaning the treatment of the data, the features, the model and its hyperparameters, with the best possible score m(p) on test data it has never used for any decision.
2. Reach that pipeline using as little of the budget B as possible.

A conventional AutoML run solves the first part with one search [@wang2021flaml]. AutoPilot-ML works on both parts with a sequence of Experiments e1, e2, ..., ek, where each next Experiment is chosen from the Diagnoses of the earlier ones and from a memory of Experiments on other datasets. The research question of the project is whether an autonomous experimentation layer can improve the efficiency and effectiveness of conventional AutoML by learning from previous experiments, diagnosing model weaknesses, selecting appropriate improvement strategies and allocating computational resources adaptively.

## Objectives

Table @tab:objectives lists the seven objectives of the project and what was delivered for each of them. All seven were achieved.

::table tab:objectives | Objectives of the project and what was delivered | widths=0.8,6,7 | align=cjj
| No. | Objective | Delivered |
|---|---|---|
| O1 | A baseline: profiling, task detection, FLAML search, held-out evaluation, explanations, experiment tracking and a dashboard | The complete baseline pipeline with a Streamlit dashboard and MLflow tracking |
| O2 | A Critic: automatic, structured Diagnoses of each Experiment, with their evidence | 14 kinds of Diagnosis, each with its evidence numbers, checked on data with known answers |
| O3 | A Strategy_Engine: map Diagnoses to candidate changes and rank them by rules, bandits and memory | 12 Strategies, a rule table and four ways to order them, including Thompson sampling and UCB1 |
| O4 | A planner and autonomous loop with a fair comparison of every change over several splits | The Session loop, the Assessor on 15 folds and the Judge's corrected resampled t-test |
| O5 | An Experiment_Memory that reuses what worked on similar datasets | An append-only memory of every change with its effect and cost, read by the learned selectors |
| O6 | A Resource_Manager that shares the time and CPU budget and decides when to stop | Budgets, a time reserve, stop rules and the utility of every step |
| O7 | An evaluation against FLAML, AutoGluon and a manual baseline on public datasets and several seeds | A benchmark of 390 runs on 6 OpenML datasets with 5 seeds and two budgets |
::end

## Scope

The work has the following scope.

- In scope: single-table tabular data in CSV files; binary classification, multiclass classification and regression; a laptop CPU; fully offline operation.
- Out of scope: images, free text documents, time series and clustering; language models as decision makers; cloud or paid services.
- Supported settings: time budgets from 5 seconds to one hour for one Experiment, datasets with at least 20 labelled rows and a test share between 5% and 50%. The user chooses the target column, helped by ranked suggestions.
