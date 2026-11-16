# Monthly evaluation reports E1 to E4

Content for the department's form "Project - Monthly Evaluation Report - E1 / E2 / E3 / E4". The project is divided into four phases in the order in which the work was done (`development-roadmap.md`). Each description of work done is at most 50 words. Fill in the date of each evaluation and the group number by hand; the supervisor completes the evaluation part.

## Fields that are the same on all four forms

| Field | Entry |
|---|---|
| Department of | Computer Science and Engineering |
| Group Number | (fill in) |
| Number of Students in the Group | 2 |
| Sch. No. and Name of the Students | 23U02016 Priyanshu Verma; 23U02003 Salil Kushwah |
| Project Title | AutoPilot-ML - An Autonomous, Self-Optimizing Machine Learning Experimentation Framework |
| Project Supervisor | Dr. Gagan Vishwakarma |
| Department Coordinator | Dr. Rahul Chaurasia |

## E1: Phase 1, study and baseline AutoML pipeline

Studied AutoML systems and FLAML, then built the baseline pipeline: CSV upload, dataset profiling, automatic task detection, FLAML model search, held-out test evaluation against trivial baselines, permutation importance, MLflow experiment tracking and a Streamlit dashboard, with automated tests. Presented the working baseline at the Phase 1 review.

## E2: Phase 2, autonomous experimentation loop

Built the autonomous loop around FLAML: Critic (diagnoses weaknesses), Strategy Engine (proposes changes), Judge (accepts a change only through a corrected resampled t-test on 15 folds), Resource Manager and Experiment Memory. Validated Critic and Judge on synthetic data; the Titanic session raised test F1 from 0.684 to 0.724.

## E3: Phase 3, learning from past experiments and benchmark set-up

Added learning from past experiments: Thompson sampling and UCB1 order the changes using Experiment Memory records from similar datasets. Selected 12 OpenML datasets by a fixed rule, set up FLAML alone, AutoGluon and a LightGBM baseline as comparison systems, and built a resumable benchmark harness with statistical analysis.

## E4: Phase 4, extensions, full benchmark and final documentation

Added SMOTE, target encoding, word features, group-aware folds, SHAP explanations, a leakage check, a local web API and a Docker image (586 tests). Ran the full 390-run benchmark: at 5 minutes, FLAML-level scores with 29% less time, second of five systems, best on pc4. Prepared the final report and presentation.
