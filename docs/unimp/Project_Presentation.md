# AutoPilot-ML

## Autonomous Machine Learning Experimentation and Self-Optimization Framework

B.Tech Computer Science Major Project, Phase 1 review

> Archived: the text presented at the Phase 1 review (October 2026). It is no longer updated.
>
> Correction to point 2 in section 11: the shared-ticket explanation does not hold up. On that split, test rows shared a ticket with a training row more often (37%) than validation rows did with their training folds (29%), so shared tickets would help the test score at least as much. Validation preferring the worse setup happened on that one split only. Details: `docs/imp/report-notes.md`, chapter 9.

**Contents**

- Part 1. The project
- Part 2. Demo
- Part 3. Summary and the work ahead

---

# Part 1. The project

## 1. What is AutoPilot-ML?

AutoPilot-ML is a framework that aims to run machine learning experiments on its own.

The user gives it a tabular dataset (a CSV file), chooses the column to predict, and sets a budget, for example ten minutes on a laptop. The finished system will train a model, work out what is wrong with it, decide what to try next, run that experiment, and keep going until the budget is used up or nothing is improving any more. At the end it returns the best model it found, together with a log of every experiment and the reason behind each step.

In this first phase we have built and tested the baseline that the rest of the system will sit on: dataset profiling, problem-type detection, model training with FLAML, honest evaluation, basic explainability, and tracking of every experiment.

## 2. The problem

Building a good model for a new dataset is a loop. You train a model, look at the results, find a weakness, change something, and train again.

Take the Titanic dataset. In our first experiment the model reached a test F1 of 0.67, but its F1 on the training data was 0.93. The model is far better on data it has already seen, so it is overfitting. Looking at the columns explains part of it: `Ticket` has 681 different values in 891 rows, and 63% of the passengers in the test set have a ticket number the model never saw during training. A person would now try removing `Ticket` and `Cabin`, train again, compare the two runs, and decide whether the change was worth it.

AutoML tools automate one step of this loop very well: choosing the model and its hyperparameters. Everything around that step is still done by hand: spotting the problem, choosing the fix, comparing runs and deciding when to stop. And what was learned on one dataset ("removing high-cardinality columns helped here") is usually lost by the time the next dataset comes along.

AutoPilot-ML is about automating that outer loop.

## 3. Our goal

The finished system takes:

- a CSV file and the target column,
- a budget: maximum time (for example 10 minutes), maximum number of experiments (for example 15), and the number of CPU cores (for example 2).

It returns:

- the best pipeline it found (data treatment, model and hyperparameters),
- an experiment log: what it tried, what it diagnosed, why it chose each next step, and what each step cost.

To do this, the system needs four abilities that ordinary AutoML does not have:

| Ability | Planned module | Example |
|---|---|---|
| Diagnose | Critic | "Train F1 0.93, test F1 0.67: overfitting." "Majority class is 92%: severe imbalance." |
| Decide | Strategy engine | "Overfitting together with a high-cardinality column: try removing the column first, then a more regularised search." |
| Remember | Experiment memory | "On datasets that looked like this one, class weighting improved F1." |
| Budget | Resource manager | "The last two experiments gained almost nothing for their cost: stop, or try something different." |

The research question of the project:

> Can an autonomous experimentation layer improve the efficiency and effectiveness of conventional AutoML by learning from previous experiments, diagnosing model weaknesses, selecting appropriate improvement strategies, and allocating computational resources adaptively?

In simple terms: on the same laptop and with the same time limit, can our system reach a good model faster, or reach a better one, than running FLAML for the whole budget? We will answer this by measurement.

## 4. Where it started: FLAML

This project started from working with FLAML, the open-source AutoML library from Microsoft Research (Wang, Wu, Weimer and Zhu, "FLAML: A Fast and Lightweight AutoML Library", MLSys 2021, MIT licence).

FLAML finds good models quickly on ordinary hardware because its search methods (CFO and BlendSearch) take the cost of each trial into account. They start with cheap configurations and only move to expensive ones when that looks worthwhile. That is what makes a fully local project like ours realistic: no GPU and no paid service are needed.

While using FLAML, we noticed that it answers one question very well: which model and which hyperparameters? The questions that came after every run (why is this model weak, what should change, was the extra time worth it) were still left to us. That gap became this project.

FLAML is therefore the search engine of AutoPilot-ML. We use it unchanged, as a normal dependency, and build the layers around it. We do not modify FLAML and we do not claim to beat it. Comparing against FLAML on its own is part of our evaluation plan.

## 5. What we have built in Phase 1

- **Data loading:** upload any CSV in the dashboard, or choose one of four public sample datasets (Titanic, Breast Cancer Wisconsin, Palmer Penguins, California Housing).
- **Dataset profiling:** rows, columns, data types, missing values, duplicate rows and unique values for every column, with automatic flags for identifier columns, constant columns, high-cardinality columns, mostly-empty columns and class imbalance.
- **Problem detection:** after the user picks the target column, the system detects binary classification, multiclass classification or regression, and explains why. The user can override the choice, and impossible overrides are rejected with a clear message.
- **Training with FLAML:** a stratified 80/20 train/test split, then a FLAML search within the time budget, using the metric and random seed the user chose.
- **Honest evaluation:** every metric is calculated on the 20% test set that FLAML never saw, next to the training score and a trivial baseline. The system warns when accuracy would be misleading because of class imbalance.
- **Explainability:** permutation feature importance, the model's own feature importance, a confusion matrix, a ROC curve, and residual plots for regression.
- **Experiment tracking:** every experiment is recorded in MLflow with its settings, results, model and files, and also saved as a small JSON record. The dashboard shows the full experiment history.
- **Quality:** 47 automated tests. Everything runs offline on a laptop CPU, with no paid APIs and no cloud services.

## 6. How one experiment runs

```text
CSV ─► Profiler ─► Task detection ─► Train/test split ─► FLAML search ─► Evaluation ─► Explanation ─► Tracking ─► Dashboard
```

1. **Check the setup:** the target exists, the time budget is valid, there are enough rows, and the metric suits the task.
2. **Prepare the data:** rows without a target value are dropped, and identifier columns are excluded (for Titanic, `PassengerId` and `Name`).
3. **Split:** 80% for training (712 rows on Titanic) and 20% for testing (179 rows). The ratio of survivors is kept the same in both parts.
4. **Search:** within the time budget, FLAML tries several model families (LightGBM, XGBoost, random forest, extra trees, linear models) with different hyperparameters. On Titanic each trial is scored with 5-fold cross-validation on the training part. On larger datasets FLAML switches to a single holdout split to save time.
5. **Retrain:** the best configuration is trained again on all training rows.
6. **Evaluate:** the final model is scored once on the untouched test set and compared with a trivial baseline (always predict the majority class).
7. **Explain:** each column of the test set is shuffled in turn, and the drop in score shows how much the model relies on that column.
8. **Record:** the model, the results and all files are saved and logged to MLflow, and the experiment appears in the history.

## 7. Architecture

The code is split into three layers:

```text
Dashboard (Streamlit)    app/main.py, app/components/     input and display only
Services                 app/services/                    all the logic
Core                     app/core/                        settings and logging
Storage                  experiments/, models/, mlruns/   local files, no server
```

| Module | Role |
|---|---|
| `profiler.py` | Builds the dataset profile and the data-quality flags |
| `task_detector.py` | Detects the problem type and checks user overrides |
| `automl_runner.py` | Runs a complete experiment; FLAML sits behind a small engine interface |
| `evaluator.py` | Metrics, trivial baselines, imbalance warnings, feature importance |
| `experiment_tracker.py` | Experiment IDs, MLflow logging, JSON records, history |

The services do not depend on the dashboard, so the same experiment can be run from a script or a notebook. FLAML is only used inside one class, so a second engine (AutoGluon, for comparison) can be added without changing anything else.

The complete planned system builds on this baseline:

```text
Dataset ─► Profiler + task detection ─► Experiment planner* ─► FLAML ─► Evaluation
                                               ▲                           │
                                               │                           ▼
                                        Strategy engine* ◄───────────── Critic*
                                               ▲                           │
                                               │                           ▼
                                        Resource manager*         Experiment memory*

* planned for the next phase
```

## 8. What the baseline adds to a plain FLAML run

| Area | Plain FLAML run | AutoPilot-ML baseline |
|---|---|---|
| Understanding the data | Not part of the run | Full profile with data-quality flags |
| Problem type and metric | Given by the user | Detected and validated, with an imbalance warning for accuracy |
| Evaluation | FLAML's own validation score, which is the number the search optimised | Separate untouched test set, trivial baseline, train vs test comparison |
| Unseen values at prediction time | A warning printed on every prediction | One clear note per column, such as "Ticket: 63% of test rows have categories never seen in training" |
| Explanation | Importance on FLAML's internal features | Importance on the original columns, confusion matrix, ROC curve, residual plots |
| Search visibility | A log file | Leaderboard per model family and a chart of every trial |
| Record keeping | Nothing kept between runs | Every experiment in MLflow and in a JSON record, with history in the dashboard |
| Usability | Python code | Dashboard: upload a CSV and run |

The training-versus-test numbers, the data-quality flags and the experiment records are also the raw material that the planned critic and experiment memory will work from.

## 9. Technology

Python 3.11, pandas, NumPy, scikit-learn, FLAML (with LightGBM and XGBoost), MLflow, Streamlit, Plotly and pytest. All of it is open source and runs locally.

---

# Part 2. Demo

## Step 1. Open the project

```bash
conda activate automl
cd C:\Users\priya\Desktop\project\AutoML\autopilot-ml
```

## Step 2. Run the automated tests

```bash
python -m pytest -q
```

## Step 3. Start the dashboard

```bash
streamlit run app/main.py --server.address localhost --server.headless true --browser.gatherUsageStats false
```

Open http://localhost:8501

## Step 4. Start the experiment tracker (second window)

```bash
conda activate automl
cd C:\Users\priya\Desktop\project\AutoML\autopilot-ml
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000 --workers 1
```

Open http://127.0.0.1:5000

## Step 5. Run an experiment from Python, without the dashboard (third window)

```bash
conda activate automl
cd C:\Users\priya\Desktop\project\AutoML\autopilot-ml
python
```

```python
import pandas as pd
from app.services.automl_runner import AutoMLRunner
df = pd.read_csv("data/sample/titanic.csv")
result = AutoMLRunner().run(dataframe=df, target_column="Survived", time_budget=30, excluded_columns=["PassengerId", "Name"], dataset_name="titanic")
print(result.experiment_id, result.best_model_name, round(result.test_score, 3))
exit()
```

## Step 6. Compare two setups over several random splits

```bash
python scripts/compare_seeds.py --dataset data/sample/titanic.csv --target Survived --exclude-a PassengerId,Name --exclude-b PassengerId,Name,Ticket,Cabin --seeds 1 2 3 --budget 10
```

---

# Part 3. Summary and the work ahead

## 10. Summary

In Phase 1 we built a working, tested and fully local baseline for automated machine learning experiments. A CSV file goes through profiling, problem detection, a FLAML search, honest evaluation on held-out data, explanation and tracking, and every experiment is kept so that experiments can be compared. FLAML is used unchanged as the search engine. Our work is the layer around it, and that layer is designed so that the autonomous parts can be added next without rebuilding anything.

## 11. What the first experiments showed

We ran Titanic twice with a 60-second budget and the same seed. The only difference was that the second run also excluded `Ticket` and `Cabin`, which the profiler had flagged as high-cardinality.

| | EXP-001: identifiers excluded | EXP-002: `Ticket` and `Cabin` also excluded |
|---|---|---|
| Test F1 | 0.673 | 0.709 |
| Test ROC-AUC | 0.792 | 0.823 |
| Test accuracy | 0.793 | 0.793 |
| Share of survivors found (recall) | 55% | 65% |
| Train F1 | 0.934 | 0.873 |
| Gap between train and test F1 | 0.26 | 0.16 |
| FLAML validation F1 | 0.802 | 0.782 |
| Trials in 60 seconds | 290 | 501 |

What we learned from this:

1. **The change helped in the right way on this split.** The model overfits less (the gap fell from 0.26 to 0.16) and finds 65% of the survivors instead of 55%, at the same accuracy.
2. **The validation score pointed the wrong way.** FLAML's own validation score preferred the first setup, but the untouched test set showed the second one was better. A plain AutoML run, which picks by validation score, would have kept the worse setup. A likely reason is that families share ticket numbers, so cross-validation sees the same ticket in its training and validation parts and looks better than it really is.
3. **The search became cheaper.** Without the two messy text columns, FLAML ran 501 trials in a minute instead of 290.
4. **One split is not enough to judge a change.** We repeated both setups on five other random splits, with a 30-second budget each (Step 6 of the demo runs a shorter version of the same comparison). On average, test F1 went from 0.772 to 0.784, and the change helped on three of the five splits. Across all six splits, the test F1 of the same setup varied from 0.67 to 0.81, only because different passengers ended up in the test set. That variation is larger than the effect of the change.

The short version in Step 6 of the demo uses only three splits and 10 seconds per run, so its numbers move around even more, and it can come out slightly in favour of either setup. That is the same point again. The one result that holds every time is that the second setup completes more trials in the same time.

So removing the two columns is a small and cheap improvement, and a single run could not have told us that. A system that changes experiments on its own must measure changes over several splits and weigh them against their cost, otherwise it will chase noise. This is exactly what the critic and the experiment memory are designed to handle.

## 12. Improvements we will work on

### 12.1 Critic

The critic will read every finished experiment and produce structured diagnoses, each with its evidence. It will check for overfitting (a large gap between train and test scores), underfitting (low scores on both), class imbalance, missing values, high-cardinality or redundant columns, unseen categories at prediction time, unreliable validation (validation and test disagreeing), and wasted compute. For example: `{"overfitting": "high", "evidence": "train F1 0.93, test F1 0.67"}`.

We will test it on synthetic datasets where we plant known problems on purpose, so we can measure how often it finds them.

### 12.2 Strategy engine

The strategy engine will turn diagnoses into candidate fixes and rank them:

| Diagnosis | Candidate fixes |
|---|---|
| Overfitting | Remove or encode high-cardinality columns, use a more regularised search space, select features |
| Underfitting | More budget, more expressive model families, new features |
| Class imbalance | Class weights, SMOTE resampling, switching the metric to F1 or ROC-AUC, threshold tuning |
| Missing values | Missing-value indicator columns, or dropping columns that are mostly empty |
| Wasted compute | Restrict the list of models, reduce the budget, or stop |

The ranking will be developed in three steps: first a fixed rule table, then a multi-armed bandit that learns which fixes give the most improvement per second of compute, and finally priors taken from the experiment memory.

### 12.3 Experiment planner and the automatic loop

The planner will turn the chosen fix into the next experiment and run it. The current runner already accepts a complete experiment configuration, so the planner plugs straight in. On Titanic, one cycle of the loop will look like this:

```text
EXP-001   baseline run                  test F1 0.67, train F1 0.93
Critic    overfitting: high; 'Ticket' high cardinality, 63% of test rows unseen
Strategy  remove 'Ticket' and 'Cabin' (the cheapest fix that matches both diagnoses)
EXP-002   run with the change           compare over several splits, keep it only if it helps
Memory    record the diagnosis, the fix, its real effect and its cost
Next      the next fix, or stop when the budget or the gains run out
```

In this phase we performed this cycle by hand. The next phase automates it.

### 12.4 Experiment memory

The memory will store, for every experiment, a description of the dataset (its size, share of missing values, imbalance, number of categorical columns and similar measures), the diagnosis, the fix applied, the model, the result and the cost. For a new dataset it will find the most similar past datasets and try first the fixes that worked there. This is how the system is meant to improve with use.

### 12.5 Resource manager

The resource manager will enforce the user's limits on time, number of experiments and CPU cores. It will decide how to split the budget between experiments and when to stop, using a simple measure of value:

```text
Utility = improvement in score / computational cost
```

### 12.6 Better evaluation inside the baseline

- Repeating each comparison over several random splits by default, as in our first experiments, so that decisions are not based on noise.
- Group-aware validation where rows are related, for example keeping passengers with the same ticket in the same fold, so that the validation score stays honest.

### 12.7 New strategies and features

- Feature engineering, for example extracting the passenger's title (Mr, Mrs, Miss, Master) from `Name` and computing family size on Titanic.
- Class weighting and SMOTE for imbalanced targets.
- Encoding of high-cardinality columns instead of only removing them.
- SHAP explanations for the whole model and for single predictions.
- Automatic target detection and data-leakage checks.

### 12.8 Comparison and research evaluation

To answer the research question, we will compare six systems on the same public datasets (from OpenML), time budgets and random seeds:

1. A manual baseline (one sensible model, no search)
2. FLAML alone, with the whole budget
3. AutoGluon, another strong AutoML tool
4. AutoPilot-ML without the experiment memory
5. AutoPilot-ML without strategy selection
6. The full AutoPilot-ML

We will measure the final test score, the time needed to reach a target score, the number of experiments, CPU time, how the best score improves over time, and how consistent the results are across datasets. Systems 4 and 5 form the ablation study, which shows which part of the system actually helps. Differences will be checked with standard statistical tests over datasets and seeds. If the full system does not beat FLAML, we will report that result as well.

### 12.9 Later

- An optional local language model (for example through Ollama) to explain diagnoses in plain language. Decisions will stay measurable and will not depend on it.
- A REST API (FastAPI), a Docker image, and a database backend for multiple users.

## 13. Timeline

| Weeks | Work |
|---|---|
| 1-2 | Baseline (done) |
| 3-4 | Critic, tested on synthetic datasets with planted problems |
| 5-6 | Strategy catalogue and rule-based strategy engine |
| 7-8 | Planner loop, experiment memory and resource manager |
| 9-10 | Learned strategy selection (bandits, memory-based) and the AutoGluon engine |
| 11-12 | Benchmark on public datasets |
| 13-14 | Ablation study, analysis and final report |

## 14. Closing

AutoML has made model selection easy. The harder part of machine learning work is the loop around it: understanding why a model is weak, choosing what to change, judging whether a change really helped, and deciding where to spend limited compute. Phase 1 gives us a reliable and honest base for that loop, and our first experiments already showed why careful measurement matters. The next phase builds the critic, the strategy engine, the memory and the resource manager on top of it, and tests whether they make AutoML faster and more effective.

## References

1. Wang, C., Wu, Q., Weimer, M., and Zhu, E. (2021). FLAML: A Fast and Lightweight AutoML Library. MLSys.
2. Wu, Q., Wang, C., and Huang, S. (2021). Frugal Optimization for Cost-related Hyperparameters. AAAI.
3. Wang, C., Wu, Q., Huang, S., and Saied, A. (2021). Economical Hyperparameter Optimization with Blended Search Strategy. ICLR.
4. Feurer, M., Klein, A., Eggensperger, K., Springenberg, J., Blum, M., and Hutter, F. (2015). Efficient and Robust Automated Machine Learning. NeurIPS.
5. Erickson, N., Mueller, J., Shirkov, A., Zhang, H., Larroy, P., Li, M., and Smola, A. (2020). AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data. arXiv:2003.06505.
6. Chawla, N. V., Bowyer, K. W., Hall, L. O., and Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16.
7. Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. Journal of Machine Learning Research, 7.
8. Vanschoren, J., van Rijn, J. N., Bischl, B., and Torgo, L. (2013). OpenML: Networked Science in Machine Learning. SIGKDD Explorations, 15(2).
