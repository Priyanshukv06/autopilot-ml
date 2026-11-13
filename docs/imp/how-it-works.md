# How AutoPilot-ML works today, and how FLAML works inside it

This guide explains the current system in plain language. Sections 1 to 8 cover the single experiment (the Phase 1 baseline): what each part does, how the parts fit together, what FLAML does, step by step, when we ask it to find a model, and the limits of one experiment. Section 9 covers the autonomous layer built on top of it in Phase 2, which answers those limits with a loop.

Everything here has been checked. The numbers come from our own runs, mostly EXP-001 (Titanic, 60 second budget, seed 42). The descriptions of FLAML come from reading the source code of FLAML 2.7.0, the version we use (Appendix C lists the files).

- To see the diagrams, open this file in Kiro and press Ctrl+Shift+V (Markdown preview). The diagrams are drawn by Mermaid; the charts are PNG files in the `images` folder next to this file.
- The technical reference for the same code is `architecture.md`. The notes for the final report are in `report-notes.md`.
- Phase 3 so far: choosing changes from the memory and the switches that turn single parts of the loop off are in section 9.3. The comparison systems and the benchmark harness are research tools outside the app; they are described in `architecture.md` and in the report notes (section 4.10). The first benchmark run was stopped early, and a second, full run with the final code finished on 13 November 2026; their results and those of the judge agreement study are in the report notes (sections 9.16 to 9.18 and 9.20), not here.
- Optional extensions (tasks 29 to 35, built from 1 to 12 November 2026): the new Strategies and Diagnoses are in the tables of section 9.3, together with SMOTE, group-aware splits and the new dashboard options; target suggestion, the leakage check, SHAP, the plain-language summaries, the HTTP API and the Docker image are in section 9.8. The first benchmark run was made before them; the second, full run includes them.

Last updated: 13 November 2026 (the second benchmark run: the notes above and section 9.7; before that, 12 November 2026, the optional extensions: sections 1, 3.2, 3.3, 9.3, 9.7, 9.8 and Appendices A and B; before that, 31 October 2026, changes made after reading the Phase 3 results: sections 9.3 and 9.7)

## Contents

1. The big picture
2. Words used in this guide
3. Our code, part by part
4. One experiment, step by step
5. How the data is split, and the three scores
6. How FLAML works inside
7. One real run from start to finish: EXP-001
8. What a single experiment cannot do
9. The autonomous layer (Phase 2)
- Appendix A. Where each step lives in the code
- Appendix B. Commands
- Appendix C. Sources

---

## 1. The big picture

You give the app a table (a CSV file) and choose the column to predict (the app suggests likely ones, with its reasons). The app then:

1. describes the table: its columns, missing values and problem columns,
2. works out what kind of question it is: one of two answers, one of several classes, or a number,
3. locks away 20% of the rows as a final exam,
4. asks FLAML to find a good model with the other 80%, within your time limit,
5. gives the final exam to the model FLAML found, and shows which columns the model relies on,
6. saves everything, so that experiments can be compared later.

```mermaid
flowchart LR
    U["You"] --> D["Dashboard in the browser<br/>(Streamlit)"]
    D --> S["AutoPilot-ML services<br/>(our code)"]
    S --> F["FLAML<br/>(searches for a model)"]
    F --> S
    S --> R[("Saved results<br/>experiments, models, MLflow")]
    R --> D
    S --> D
```

FLAML is the only part that searches for a model. It is an open-source library from Microsoft Research, and we use it unchanged. Everything around it is our code:

| Part | Written by | Job |
|---|---|---|
| Dashboard | us, using the Streamlit library | The web page: collects your choices and shows the results |
| Services | us | All the logic: describe the data, detect the task, run the experiment, score it, save it |
| FLAML | Microsoft Research (open source) | Finds the best kind of model and its settings |
| scikit-learn | open source | Splits data, computes scores, measures how much the model relies on each column |
| MLflow | open source | Stores every experiment so it can be browsed later |
| Plotly | open source | Draws the charts in the dashboard |

```mermaid
flowchart TB
    subgraph OURS["Written by us"]
        UI["Dashboard<br/>app/main.py, app/components/"]
        SV["Services<br/>app/services/"]
        CO["Settings and logging<br/>app/core/"]
    end
    subgraph LIBS["Libraries we use unchanged"]
        ST["Streamlit and Plotly"]
        FL["FLAML, with LightGBM and XGBoost"]
        SK["scikit-learn"]
        ML["MLflow"]
        PD["pandas and NumPy"]
    end
    subgraph DISK["Folders on the laptop"]
        EX[("experiments/")]
        MO[("models/")]
        MR[("mlruns/")]
    end
    UI --> SV
    UI --> ST
    CO --> UI
    CO --> SV
    SV --> FL
    SV --> SK
    SV --> ML
    SV --> PD
    SV --> EX
    SV --> MO
    ML --> MR
```

Everything runs on the laptop: no internet service, no paid API, no GPU.

---

## 2. Words used in this guide

| Word | Meaning | Example from our project |
|---|---|---|
| Dataset | A table of examples: one row per example, one column per piece of information | Titanic: 891 passengers, 12 columns |
| Target | The column we want to predict | `Survived`: 1 survived, 0 died |
| Feature | Any other column the model may use | `Sex`, `Age`, `Pclass` (ticket class), `Fare` |
| Model | A rule learned from examples that turns features into a prediction | Something like "women in 1st or 2nd class survived", but learned automatically and far more detailed |
| Training | Learning a model from examples whose answer is known | |
| Task | The kind of question. Binary classification: one of two answers. Multiclass: one of several classes. Regression: a number | Titanic is binary, penguins multiclass, house prices regression |
| Decision tree | A model that asks a chain of yes/no questions ("female?", "younger than 10?") and ends in an answer | |
| Random forest, extra trees | Many decision trees, each built on a random part of the data, voting together | |
| Gradient boosting | Trees built one after another, each one fixing the mistakes of the trees before it. LightGBM and XGBoost are two fast versions | The winner of EXP-001 is XGBoost |
| Linear model | Adds up the features, each multiplied by a learned weight. Fast, but often weaker on tables like ours | SGD and L1 logistic regression |
| Learner, learner family | One kind of model that FLAML can try | FLAML's default list for classification has 7 |
| Hyperparameter, setting | A choice made before training that shapes the model | Number of trees, leaves per tree, learning rate |
| Configuration | One complete set of settings for one learner | XGBoost with 304 trees and at most 11 leaves per tree, plus 7 other settings |
| Trial | Training and scoring one configuration once | EXP-001 ran 290 trials in 60 seconds |
| Search | Trying many configurations to find a good one | |
| Time budget | How long FLAML may search, in seconds | 60 s in EXP-001 |
| Cost | How long a trial takes. Bigger models cost more | A trial with a 4-tree model took about 0.1 s |
| Training part, test part | Our outer split. FLAML only ever sees the training part (80%). The test part (20%) is used once, at the end | 712 and 179 passengers |
| Validation | Scoring during the search, on rows held back from training, to compare configurations | |
| Cross-validation, fold | Cut the training part into 5 pieces called folds. Train on 4, score on the 5th, do this 5 times so every fold is scored once, and average the 5 scores | Section 5 |
| Holdout | A quicker kind of validation: keep back one piece (FLAML uses 10% of the training part) and score on it | Used for larger datasets |
| Overfitting | The model learned the training rows too closely, including their noise, so it is much better on rows it has seen than on new rows | EXP-001: F1 0.93 on training rows, 0.67 on test rows |
| Metric, score | A number that says how good the predictions are | Explained below this table |
| Loss | A score where lower is better. FLAML always minimises a loss; for F1 it uses 1 - F1 | |
| Trivial baseline | The score of a "model" that learns nothing, such as "always predict died". A useful model must beat it | Titanic test part: F1 0.000, accuracy 0.615 |
| Categorical column | A column of labels instead of numbers | `Sex`, `Embarked`, `Ticket` |
| High cardinality | A categorical column with very many different labels | `Ticket`: 681 labels for 891 passengers |
| Unseen label | A label in the test part that never appears in the training part, so the model knows nothing about it | 63% of test tickets in EXP-001 |
| Missing value | An empty cell | `Age` is empty for 177 passengers |
| Median | The middle value when the values are sorted | Used to fill missing ages |
| Seed | A number that fixes "random" choices so they can be repeated | 42 |
| Stratified split | A split that keeps the share of each class the same in every part | About 38% survivors in both parts |
| Permutation importance | Shuffle one column and see how much the score drops. A big drop means the model relies on that column | `Sex` is the most important column on Titanic |
| MLflow run, artifact | One experiment's record in MLflow, and a file saved with it | The model file, `profile.json` |
| Pickle | Python's way of saving an object, such as a trained model, to a file | `models/EXP-001/model.pkl` |
| CFO | Cost-Frugal Optimization: FLAML's way of choosing settings. "Frugal" means careful with cost: start small, grow only when it pays off | Section 6.7 |
| Local search, step, converged | Searching close to the best point found so far. A step is one small change of the settings. Converged means the steps have become so small that nothing new is being found | Section 6.7 |

The metrics, in short:

- Accuracy: the share of correct answers.
- Precision: when the model says "survived", how often it is right.
- Recall: of the passengers who really survived, how many the model found.
- F1: one number that balances precision and recall, from 0 to 1, higher is better. It is our default for classification, because accuracy can look good while the smaller class is ignored.
- ROC-AUC: how well the model ranks survivors above non-survivors. 0.5 is a coin flip, 1 is perfect.
- For numbers (regression): R² (1 is perfect, 0 is no better than always guessing the average), and RMSE and MAE (the typical size of the error, lower is better).

---

## 3. Our code, part by part

### 3.1 Code map

```mermaid
flowchart TD
    MAIN["main.py<br/>page layout and sidebar"]
    UP["dataset_uploader.py<br/>Section 1: load a CSV"]
    DP["dataset_profiler.py, dashboard part<br/>Sections 1 and 2"]
    EV["experiment_viewer.py<br/>Sections 3, 4 and 7"]
    RV["results_viewer.py<br/>Sections 5 and 6"]
    PROF["profiler.py<br/>DatasetProfiler"]
    TASK["task_detector.py<br/>TaskDetector"]
    RUN["automl_runner.py<br/>AutoMLRunner, FLAMLEngine"]
    EVAL["evaluator.py<br/>Evaluator"]
    TRK["experiment_tracker.py<br/>ExperimentTracker"]
    FLAML["FLAML library"]
    MAIN --> UP
    MAIN --> DP
    MAIN --> EV
    MAIN --> RV
    DP --> PROF
    DP --> TASK
    EV --> RUN
    EV --> TRK
    RV --> EVAL
    RUN --> PROF
    RUN --> TASK
    RUN --> EVAL
    RUN --> TRK
    RUN --> FLAML
```

The code has three layers:

- `app/main.py` and `app/components/` are the dashboard. They only collect your choices and show the results.
- `app/services/` holds all the logic. None of it uses Streamlit, so the same code also runs from a script, a notebook or a test.
- `app/core/` holds the settings (`config.py`) and the logging setup (`logging_config.py`).

Two rules keep the parts independent: FLAML is used in exactly one class (`FLAMLEngine`), and MLflow in exactly one class (`ExperimentTracker`). Replacing either of them later only touches that one class.

### 3.2 The five services

**Dataset profiler** (`app/services/profiler.py`). Describes the table before any training: the number of rows and columns, the type of each column, missing values, duplicate rows, and how many different values each column has. It also flags problem columns:

| Flag | Rule | Titanic |
|---|---|---|
| Identifier-like | Every value is different, nothing is missing, and the column is not decimal or a date. Or the name looks like an ID and more than 95% of the values are different | `PassengerId`, `Name` |
| Constant | Only one value | none |
| High cardinality | A text column with more than 50 different values | `Ticket` (681), `Cabin` (147) |
| Mostly missing | At least 40% empty | `Cabin` (77% empty) |
| Class imbalance | The biggest class divided by the smallest: under 1.5 none, 1.5 to 3 mild, 3 to 9 moderate, 9 or more severe | 549 died, 342 survived: 1.61, mild |

Identifier-like and constant columns are pre-selected for exclusion, because a column that is different in every row cannot teach the model anything general. The profiler also computes "meta-features": a few numbers that describe the whole dataset (its size, the share of missing values, the imbalance and so on). The experiment memory of the autonomous layer uses them to find similar datasets (section 9.3). Since November 2026 it also lists the frequent words of each text column (for the word features of section 9.3) and, when asked, scores each column alone as a predictor of the target (the leakage check of section 9.8).

**Task detector** (`app/services/task_detector.py`). Looks at the target column:

- text or yes/no values with 2 different values: binary; with more: multiclass,
- numbers with exactly 2 different values: binary,
- whole numbers with at most 20 different values: multiclass, with a note that it might really be a count,
- anything else: regression.

You can override its choice. Overrides that cannot work are refused with a clear message, for example "Multiclass classification needs at least 3 classes".

It also suggests which column is the target: up to three likely columns, each with its reasons, and the first is the default choice (section 9.8). You confirm it or choose another.

**AutoML runner** (`app/services/automl_runner.py`). Runs one experiment from start to finish (section 4). It contains `FLAMLEngine`, the only place that talks to FLAML. FLAML sits behind a small interface, so another engine can be added without changing anything else. (The benchmark of Phase 3 runs AutoGluon as a separate comparison system instead, in its own environment.)

**Evaluator** (`app/services/evaluator.py`). Scores any model: every metric on the test part and on the training part, the trivial baseline, the confusion matrix (a table of right and wrong answers per class) and permutation importance. It only needs the model's predictions, so it does not depend on FLAML.

**Experiment tracker** (`app/services/experiment_tracker.py`). Gives each experiment its ID (EXP-001, EXP-002 and so on), writes the MLflow run, writes the small `summary.json` record, and builds the history table. An ID is reserved by creating its folder, so two experiments started at the same time can never get the same ID.

### 3.3 The dashboard

| Section | What you see |
|---|---|
| 1. Dataset | Upload or pick a sample; cards with rows, columns, missing values and duplicates; tabs for the columns, a preview, data-quality warnings and target candidates (with the likely targets and their reasons) |
| 2. Target | Choose the target; the detected problem with its reason; override; a chart of the classes; a warning when one column alone predicts the target almost perfectly; the columns to exclude |
| 3. Experiment configuration | Time budget, primary metric, random seed, test split |
| 4. Run AutoML | The Run Experiment button and a progress bar |
| 5. Results | Best model, test score, validation score, training time; notes; every test metric against the baseline; train vs test; the leaderboard; a chart of all trials |
| 6. Explainability | Permutation importance, the model's own importance, the confusion matrix, a ROC curve (binary) or residual plots (regression), the winning settings |
| 7. Experiment history | All past experiments in one table, with a chart |

The experiment runs in the background. Streamlit runs the whole page script again after every click, so a long experiment started directly by the page would block it or get lost. Instead the page starts a background worker and checks on it twice a second:

```mermaid
sequenceDiagram
    participant You
    participant Page as Dashboard page
    participant Worker as Background worker
    participant Runner as AutoMLRunner
    You->>Page: click Run Experiment
    Page->>Worker: start the experiment
    Worker->>Runner: run_experiment(data, settings)
    loop twice a second
        Page->>Worker: which stage are you at?
        Worker-->>Page: stage and message
        Page-->>You: move the progress bar
    end
    Runner-->>Worker: ExperimentResult
    Worker-->>Page: finished
    Page-->>You: results, charts and history
```

FLAML reports nothing while it searches, so during the search the progress bar is estimated from the elapsed time against the budget.

---

## 4. One experiment, step by step

```mermaid
flowchart TD
    A["You click Run Experiment"] --> B["Step 1: check the settings"]
    B --> C{"Settings valid?"}
    C -->|No| ERR["Show the error, nothing is run"]
    C -->|Yes| D["Step 2: drop rows without a target<br/>and the excluded columns"]
    D --> E["Step 3: confirm the task and the metric"]
    E --> F["Step 4: split into an 80% training part<br/>and a 20% test part"]
    F --> G["Step 5: note test labels that never appear<br/>in the training part"]
    G --> H["Step 6: reserve the next experiment ID"]
    H --> I["Step 7: FLAML searches on the training part only"]
    I --> J["Step 8: score the model on the test part<br/>and the training part, plus the trivial baseline"]
    J --> K["Step 9: permutation importance on the test part"]
    K --> L["Step 10: save the model and the result files,<br/>log the MLflow run, write summary.json"]
    L --> M["Step 11: show the results and update the history"]
```

What each step did in EXP-001:

| Step | What happens | EXP-001 |
|---|---|---|
| 1. Check | The target and the excluded columns exist, the budget is 5 to 3600 seconds, the test share is 0.05 to 0.5 | Passed |
| 2. Prepare | Rows with an empty target are dropped and the excluded columns removed; at least 20 rows and one feature column must remain | No empty targets; `PassengerId` and `Name` removed, 9 feature columns left |
| 3. Task and metric | Detected, or checked if you overrode it | Binary, F1 |
| 4. Split | Stratified 80/20 split with your seed | 712 training rows (273 survivors, 38.3%), 179 test rows (69 survivors, 38.5%) |
| 5. Unseen labels | For each text column, the share of test rows whose label never appears in the training part | `Ticket` 63%, `Cabin` 11% |
| 6. ID | Creates the folder `experiments/EXP-001/` | |
| 7. Search | FLAML, section 6 | 290 trials in 60 s; winner XGBoost |
| 8. Score | Every metric on both parts, and the trivial baseline on the test part | Test F1 0.673, train F1 0.934, baseline F1 0.000 |
| 9. Importance | Each column of the test part is shuffled 5 times; the average drop in F1 is its importance | `Sex` 0.219, `Pclass` 0.098, `Ticket` 0.043 |
| 10. Save | See the file list below | |
| 11. Show | Sections 5 to 7 of the dashboard | |

If anything fails after step 6, the runner writes `error.json` instead of `summary.json`, and the failed experiment does not appear in the history.

Files written by one experiment:

```text
experiments/EXP-001/
    summary.json                   the record the history table reads
    profile.json                   the dataset profile, including the target
    evaluation.json                test metrics, confusion matrix, per-class scores, train and baseline metrics
    leaderboard.csv                the best validation score of each learner family
    search_history.csv             every FLAML trial
    feature_importance.csv         permutation importance
    native_feature_importance.csv  the model's own importance
    predictions.csv                the test rows' true answers, predictions and probabilities
    flaml_search.log               FLAML's raw log, one line per trial
models/EXP-001/model.pkl           the trained model, the feature columns, the target, the task and the classes
mlruns/                            the MLflow store (mlflow.db), with copies of the model and the files above
```

---

## 5. How the data is split, and the three scores

There are two splits, one inside the other.

```mermaid
flowchart TD
    ALL["Titanic: 891 passengers"] --> TR["Training part: 712 passengers (80%)<br/>the only rows FLAML ever sees"]
    ALL --> TE["Test part: 179 passengers (20%)<br/>locked away until the end"]
    TR --> F1["Fold 1"]
    TR --> F2["Fold 2"]
    TR --> F3["Fold 3"]
    TR --> F4["Fold 4"]
    TR --> F5["Fold 5"]
    TE --> EXAM["Scored once, after FLAML has finished"]
```

1. Our outer split (step 4) makes the training part and the test part. We do it before FLAML starts, so FLAML cannot see the test part, not even by accident.
2. Inside the training part, FLAML validates each trial. On Titanic it uses 5-fold cross-validation, with about 142 or 143 passengers in each fold.

How one trial is scored with 5-fold cross-validation:

| Round | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---|---|---|---|---|
| 1 | score | learn | learn | learn | learn |
| 2 | learn | score | learn | learn | learn |
| 3 | learn | learn | score | learn | learn |
| 4 | learn | learn | learn | score | learn |
| 5 | learn | learn | learn | learn | score |

So one trial trains 5 models, each on 4 folds, and scores each one on the fold it did not learn from. The trial's validation score is the average of the 5 scores. The 5 models are then thrown away: FLAML only keeps the score and the settings.

The app shows three scores for the final model. They answer different questions:

| Score | Measured on | EXP-001 (F1) | How far to trust it |
|---|---|---|---|
| Train score | The 712 rows the final model learned from | 0.934 | Too optimistic: the model has seen these answers |
| Validation score | FLAML's 5-fold average for the winning settings | 0.802 | Better, but it is the number FLAML used to pick the winner out of 290 trials, so it is still a little optimistic |
| Test score | The 179 rows nobody touched | 0.673 | The honest number |

A big gap between the train score and the other two means overfitting.

Why is the validation score above the test score here? Two reasons. First, the best of 290 trials usually looks a little better than it really is, because part of its lead is luck. Second, this particular split (seed 42) put harder passengers into the test part: on five other splits the same setup scored 0.73 to 0.81 on its test part. The second reason is measured in `report-notes.md`, sections 9.6 and 9.7.

---

## 6. How FLAML works inside

### 6.1 The idea

FLAML answers one question: which learner family, with which settings, gives the best validation score within the time limit?

It treats time as money. Think of a cook who has one minute to find the best recipe. The cook first makes small, quick tasting portions of several recipes, notices which ones are promising, and spends most of the remaining time improving those. A big, slow dish is only made when a small step in that direction already tasted better. FLAML does the same with models: the first trials are tiny models that train in a fraction of a second, and the models only grow when growing improves the score.

Two decisions are repeated in every round of the search:

1. which learner family to try next (section 6.6),
2. which settings to try for it (section 6.7, CFO).

### 6.2 What we ask FLAML to do

`FLAMLEngine.fit` calls FLAML with these settings:

| Setting | Our value | Meaning |
|---|---|---|
| `task` | `"classification"` or `"regression"` | FLAML works out binary or multiclass from the labels |
| `metric` | `"f1"`, `"roc_auc"`, `"accuracy"`, `"log_loss"`, `"r2"`, `"rmse"`, `"mae"` or `"mse"`; for multiclass, F1 becomes `"macro_f1"` and ROC-AUC becomes `"roc_auc_ovr"` | What to optimise. "Macro" means the average over the classes |
| `time_budget` | your budget, for example 60 | Seconds for the search |
| `seed` | your seed, for example 42 | Used for CFO's random steps (section 6.10) |
| `n_jobs` | -1 | Each model may use every CPU core. Trials still run one at a time |
| `estimator_list` | `"auto"` | FLAML's default learner families (section 6.6) |
| `eval_method` | `"auto"` | FLAML chooses cross-validation or holdout (section 6.5) |
| `retrain_full` | `True` | Retrain the winner on the whole training part at the end |
| `early_stop` | `True` | FLAML may stop before the budget runs out if every search has settled (section 6.8) |
| `log_type` | `"all"` | Write every trial to the log, not only the improvements |
| `mlflow_logging` | `False` | We write to MLflow ourselves |
| `verbose` | 0 | Print nothing |

### 6.3 Inside one FLAML call

```mermaid
flowchart TD
    IN["Training part from our runner<br/>EXP-001: 712 rows, 9 columns"] --> P1["Step 1: prepare the data<br/>labels into categories, fill empty cells"]
    P1 --> P2["Step 2: choose how to score a trial<br/>5-fold cross-validation or holdout"]
    P2 --> P3["Step 3: search loop, one trial per round,<br/>until the time is up"]
    P3 --> P4["Step 4: retrain the best settings<br/>on the whole training part"]
    P4 --> OUT["Step 5: hand back the model, its settings,<br/>the best scores and the log"]
```

### 6.4 Step 1: FLAML prepares the data

Before searching, FLAML cleans the training part in its own way (its `DataTransformer` class). We checked these rules on the EXP-001 split:

| Rule | Titanic example |
|---|---|
| A text column with only one value, or with a different value in every filled row, is dropped | `Name` (every name is different). We exclude it ourselves, but FLAML would drop it anyway |
| A number column with fewer than 2 different values is dropped. Other number columns are kept, even when every value is different | `PassengerId` would be kept and used as if it meant something. This is why our own exclusion matters |
| Other text columns become categories. An empty text cell becomes a category called `__NAN__` | `Sex`, `Ticket`, `Cabin`, `Embarked`. Empty `Cabin` and `Embarked` cells become `__NAN__` |
| Empty number cells are filled with the median of the training part | The 137 missing ages in the training part become 28.5 |
| Date columns are split into year, month, day, weekday and so on | none on Titanic |
| The target's labels become 0, 1, 2 and so on, and are turned back into the original labels when predicting | |
| A class with fewer than 20 rows is copied until it has at least 20 | does not happen on our datasets |

Then each learner family sees the categories in its own way:

- LightGBM understands categories directly.
- XGBoost, random forest, extra trees and the linear models need numbers, so each category is replaced by its position in an alphabetical list: `female` is 0 and `male` is 1, and the 571 tickets of the training part become 0 to 570.
- A label that never appeared in the training part, such as a new ticket number, becomes `__NAN__` when predicting. For XGBoost that is position 571, so the model treats every unseen ticket as if it came just after "WE/P 5735", the alphabetically last ticket it knows. That is meaningless, and it applied to 112 of the 179 test passengers in EXP-001, whose winner was XGBoost. This is one concrete way a high-cardinality column can hurt.

### 6.5 Step 2: FLAML chooses how to score each trial

FLAML picks 5-fold cross-validation when the data is small compared with the time budget, and a holdout otherwise. The exact rule in FLAML 2.7.0:

```text
use cross-validation if   rows x columns / 0.9   is below   10,000,000 x budget in seconds / 3600
                    and   rows is below 100,000
otherwise use a holdout of 10% of the training part
```

| Dataset (training part) | rows x columns / 0.9 | Limit at 10 s | Limit at 60 s | Result |
|---|---|---|---|---|
| Titanic (712 x 9) | 7,120 | 27,778 | 166,667 | Cross-validation at every budget the app allows (even at 5 s the limit is 13,889) |
| California housing (16,512 x 8) | 146,773 | 27,778 | 166,667 | Holdout up to 52 s; cross-validation from 53 s (limit 147,222) |

Two details:

- The folds and the holdout are made with FLAML's own fixed seed (1), not with ours. For a given training part they are always the same.
- A trial's validation score is the average of its fold scores, or simply the holdout score.

### 6.6 Step 3: the search loop

```mermaid
flowchart TD
    A["Start"] --> B["Pick a learner family<br/>(section 6.6)"]
    B --> C["Its CFO search proposes settings<br/>(section 6.7)"]
    C --> D["Train and score the trial:<br/>5 models, one per fold"]
    D --> E["Trial score = average of the 5 fold scores"]
    E --> F{"Best of all trials so far?"}
    F -->|Yes| G["Remember it as the overall best"]
    F -->|No| H["Keep the old best"]
    G --> I["Update this family's records:<br/>time used, time since its last improvement"]
    H --> I
    I --> J{"Time left, and not stopped early?"}
    J -->|Yes| B
    J -->|No| K["End of the search"]
```

Each round runs exactly one trial. EXP-001 went round this loop 290 times in 60 seconds, and spent 59.7 of those seconds training and scoring models. FLAML's own decision making takes very little time.

#### The learner families

| Name in the logs | Name in the app | Kind | Starting cost guess | Settings tuned | First tried in EXP-001 |
|---|---|---|---|---|---|
| `lgbm` | LightGBM | gradient boosting | 1 | 8 | trial 0, at 0.1 s |
| `sgd` | SGD Linear Model | linear | 1 | 9 | trial 3, at 0.4 s |
| `xgboost` | XGBoost | gradient boosting | 1.6 | 9 | trial 12, at 1.4 s |
| `extra_tree` | Extra Trees | many random trees | 1.9 | 4 | trial 14, at 1.7 s |
| `rf` | Random Forest | many random trees | 2 | 4 | trial 21, at 2.8 s |
| `xgb_limitdepth` | XGBoost (depth-limited) | gradient boosting | 64 | 9 | trial 258, at 52.4 s |
| `lrl1` | Logistic Regression (L1) | linear | 160 | 1 | trial 289, at 60.0 s |

The starting cost guess is built into FLAML: how expensive a first trial of this family is expected to be, compared with LightGBM. EXP-001 tried the families for the first time in exactly this order, cheapest first. For regression the list is the same without logistic regression. CatBoost would be added if it were installed; it is not.

The two linear models did badly on Titanic (best F1 0.531 and 0.454). FLAML keeps them in the list because they are cheap and sometimes win on other data. The "did not converge" warnings we hid in Phase 1 came from these linear models.

#### How FLAML picks the next learner family

For every family it has already tried, FLAML keeps two clocks: how much of that family's time its last improvement took, and how much of its time has passed since that improvement. The larger of the two is the family's expected cost of its next improvement. A family that is behind the leader also gets a penalty: roughly the time it would need to catch up at its recent pace, doubled. A family that has not been tried yet uses its starting cost guess (multiplied by the time of the very first trial).

Then:

1. If a family that has not been tried yet looks cheapest, it gets its first turn.
2. Otherwise FLAML draws lots among the tried families. Each family's chance is proportional to 1 divided by its expected cost, so a family that is improving cheaply is picked often, and a stuck one rarely.
3. A family whose best trial took longer than the time that is left is skipped.

The tried families' expected costs grow while they stop improving, because their "time since the last improvement" keeps growing. Sooner or later an untried family looks cheapest and gets its turn. That is why XGBoost (depth-limited), with a starting cost guess 64 times LightGBM's, waited until 52 seconds in EXP-001. In EXP-002 it got its first turn at 20 seconds, and the best trial of that whole run came from it 3 seconds later.

In EXP-001 most of the time went to the family that won:

![Time per learner family in EXP-001](images/exp001_time_per_learner.png)

XGBoost got 34.6 of the 59.7 seconds of trials (58%) and 114 of the 290 trials. The draw favours families that improve cheaply, and XGBoost kept improving until 42.9 seconds.

### 6.7 How FLAML picks the settings: CFO

Each learner family has a list of settings, each with an allowed range. For XGBoost on Titanic, for example: the number of trees from 4 to 712 and the maximum leaves per tree from 4 to 712 (FLAML caps both at the number of training rows), the learning rate from about 0.001 to 1, and 6 more.

CFO (Cost-Frugal Optimization; Wu, Wang and Huang, AAAI 2021) chooses the settings for each family separately:

1. **Start small.** The settings that make training slower as they grow (the number of trees, the leaves) start at small values; for most families at the smallest allowed. The others start at sensible defaults, such as a learning rate of 0.1. The first XGBoost trial in EXP-001 had 4 trees with at most 4 leaves each.
2. **Step in a random direction.** Picture every setting as a slider scaled from 0 to 1 (wide ranges, like 0.001 to 1, are spread out evenly on a log scale). From the best point so far, CFO moves all the sliders a little, in a random direction.
3. **Try the opposite direction.** If that step was not better, it tries the same step backwards.
4. **Move when better.** As soon as a step improves the score, that point becomes the new best, and the next steps start from there.
5. **Shrink the step.** After many failed steps in a row, the step gets smaller.
6. **Restart.** When the step becomes tiny, the search has converged: it is stuck around one point. CFO then starts again from near the small starting point, with some randomness, to look elsewhere.

```mermaid
flowchart TD
    S["Start at small settings<br/>for example 4 trees with 4 leaves"] --> P["From the best point so far,<br/>take a small step in a random direction"]
    P --> T1{"Better than the best so far?"}
    T1 -->|Yes| M["Move there: new best point"]
    T1 -->|No| O["Take the same step backwards"]
    O --> T2{"Better?"}
    T2 -->|Yes| M
    T2 -->|No| C["One more failed try"]
    M --> P
    C --> Q{"Many failed tries in a row?"}
    Q -->|No| P
    Q -->|Yes| SH["Make the step smaller"]
    SH --> Z{"Step tiny?"}
    Z -->|No| P
    Z -->|Yes| R["Converged: restart near the small start,<br/>with some randomness"]
    R --> P
```

Because every family starts small and only grows when growing helps, the first trials take only 0.1 to 0.2 seconds, and slow, large models are trained only after smaller ones have shown that size pays off. That is the "frugal" part.

XGBoost in EXP-001, every time it set a new best of its own (its first trial counts as the first):

| XGBoost trial | Time (s) | Trees | Max leaves per tree | Validation F1 |
|---|---|---|---|---|
| 1 | 1.35 | 4 | 4 | 0.6313 |
| 3 | 2.55 | 4 | 7 | 0.7127 |
| 5 | 3.54 | 5 | 8 | 0.7486 |
| 7 | 3.73 | 9 | 13 | 0.7518 |
| 11 | 8.45 | 19 | 8 | 0.7739 |
| 17 | 9.40 | 18 | 26 | 0.7835 |
| 18 | 9.61 | 57 | 15 | 0.7853 |
| 53 | 24.29 | 39 | 43 | 0.7861 |
| 60 | 30.20 | 63 | 14 | 0.7915 |
| 75 | 34.16 | 123 | 10 | 0.8001 |
| 95 | 42.90 | 304 | 11 | 0.8017 |

The number of trees grew from 4 to 304 over 43 seconds. The other 103 XGBoost trials were steps that did not help.

![XGBoost settings in EXP-001](images/exp001_best_learner_settings.png)

The exact numbers behind steps 2 to 6, from FLAML's source (useful for the report):

- The first step has a length of 0.1 x the square root of the number of settings, in the 0 to 1 slider space: 0.3 for XGBoost's 9 settings.
- The step only starts to shrink after 2 to the power of the number of settings failed tries in a row, at most 512: that is 512 for XGBoost and 256 for LightGBM. In EXP-001, XGBoost never had more than 34 failed tries in a row, so its step never shrank.
- Our seed (42) drives the random directions.

### 6.8 When the search stops

The search stops when the time budget is used up. With `early_stop`, it may stop sooner, but only if both of these are true:

1. every family's CFO search has converged at least once, and
2. the search has already run more than 10 times as long as it took to find the current best.

In EXP-001 neither was true: XGBoost's search never converged, and the best was found at 42.9 seconds, so stopping early would have needed 429 seconds anyway. In EXP-002 the best came at 23.2 seconds, so the second condition alone would have needed 232 seconds. So on small datasets like ours, the search normally uses the whole budget, even after it stops improving: EXP-001 found nothing better in its last 17 seconds, and EXP-002 nothing in its last 37 seconds.

### 6.9 Steps 4 and 5: retrain the winner and hand back the result

During cross-validation, every model is trained on only 4 of the 5 folds. So after the search, FLAML trains the winning settings once more, on all 712 training rows. This happens after the budget, with no time limit. That is why an experiment takes a little longer than its budget: EXP-001's FLAML call took 60.1 seconds, and California housing with a 10 second budget took 16.9 seconds, mostly because the final model was retrained on 16,512 rows.

FLAML then hands back:

- the trained model, ready to predict (it applies the same data preparation to new rows, and turns the 0/1 answers back into the original labels),
- the name of the winning family, its settings and its validation loss,
- the best loss and the best settings of every family it tried,
- the log file, with one line per trial.

One real line of EXP-001's log (trial 0, with the numbers shortened and two fields left out):

```json
{"record_id": 0, "learner": "lgbm", "trial_time": 0.1086, "wall_clock_time": 0.1232,
 "validation_loss": 0.3687, "sample_size": 712,
 "config": {"n_estimators": 4, "num_leaves": 4, "min_child_samples": 20, "learning_rate": 0.1,
            "log_max_bin": 8, "colsample_bytree": 1.0, "reg_alpha": 0.00098, "reg_lambda": 1.0}}
```

A validation loss of 0.3687 means an F1 of 1 - 0.3687 = 0.631.

`FLAMLEngine` turns all this into the result we keep: losses become scores again (1 - loss for F1, ROC-AUC, accuracy and R²; the loss itself for error measures such as RMSE), the log becomes the trial history and the "Search progress" chart, and the best of each family becomes the leaderboard.

### 6.10 Randomness: what our seed controls

| What | Controlled by |
|---|---|
| Our 80/20 split into the training and test parts | our seed |
| The random directions of each CFO search | our seed, passed to FLAML |
| FLAML's shuffling of the training part, its folds and its holdout | FLAML's own fixed seed (1) |
| FLAML's draw between learner families | FLAML's own fixed seed (1) |
| Randomness inside each model (for example which rows each random-forest tree sees) | fixed numbers built into FLAML |
| How many trials fit into the budget | the speed of the laptop at that moment |

So two runs with the same seed make the same choices in the same order, as long as the timings are the same. But the search stops on time, and the choice of family uses measured times, so a run on a busier laptop can finish a different number of trials, and occasionally pick a different winner. When we ran the same 30 second experiment twice (Titanic, seed 1), the trial counts were 213 and 210, and the test F1 was the same.

### 6.11 Things FLAML can do that our runs do not use

- **Sampling.** For big data (a holdout and more than 40,000 training rows), FLAML first trains on a sample of about 10,000 rows and grows the sample for promising settings. None of our datasets is that big, so every trial used the whole training part.
- **Parallel trials and BlendSearch.** With the Ray library, FLAML can run several trials at once and use BlendSearch, which mixes CFO with a broader search. We run one trial at a time with CFO.
- **Ensembles.** FLAML can combine several good models into one ("stacking"). It is off by default and we do not use it.

---

## 7. One real run from start to finish: EXP-001

Titanic, target `Survived`, 60 second budget, seed 42, run on 7 October 2026.

| Stage | What happened |
|---|---|
| Data | 891 passengers, 12 columns, 866 empty cells (`Age` 177, `Cabin` 687, `Embarked` 2), no duplicate rows |
| Profile flags | `PassengerId` and `Name` identifier-like; `Ticket` (681 values) and `Cabin` (147) high cardinality; `Cabin` 77% empty; mild imbalance (549 died, 342 survived) |
| Task | Binary classification, metric F1 |
| Excluded | `PassengerId`, `Name` |
| Split | 712 training rows, 179 test rows, 38% survivors in each |
| Unseen labels | `Ticket`: 63% of test rows; `Cabin`: 11% |
| FLAML preparation | 4 category columns (`Sex`, `Ticket`, `Cabin`, `Embarked`) and 5 number columns; 137 missing ages filled with 28.5 |
| Validation | 5-fold cross-validation |
| Search | 290 trials, 7 families, 59.7 s of training and scoring |
| Improvements | 12 new overall bests: 5 from LightGBM in the first 0.8 s, then 7 from XGBoost between 8.5 s and 42.9 s |
| Failed trials | 6 trials scored F1 0 (they found no survivors at all); 19% of all trials scored below 0.70 |
| Winner | XGBoost: 304 trees, at most 11 leaves per tree, learning rate 0.03, and 6 more settings; validation F1 0.802, found at 42.9 s |
| Retrain | The winning settings were trained again on all 712 rows; the whole FLAML call took 60.1 s |
| Test part | F1 0.673, accuracy 0.793, precision 0.864, recall 0.551, ROC-AUC 0.792. It found 38 of the 69 survivors, and wrongly flagged 6 passengers who died |
| Training part | F1 0.934: a large gap, so the model overfits |
| Trivial baseline | "Everyone died": F1 0.000, accuracy 0.615 |
| Importance | `Sex` 0.219, `Pclass` 0.098, `Ticket` 0.043, `Fare` 0.027, `SibSp` 0.021 |
| Saved | `experiments/EXP-001/`, `models/EXP-001/model.pkl`, one MLflow run |

The whole search on one chart:

![All trials of EXP-001](images/exp001_search_timeline.png)

Each dot is one trial, coloured by family; the black line is the best validation F1 so far. Three things stand out:

1. **Most of the gain came almost at once.** The best F1 was 0.769 after 0.8 seconds (8 trials) and 0.785 after 10 seconds. The remaining 50 seconds added about 0.016.
2. **Cheap families first, then the promising one.** LightGBM led in the first second; XGBoost took over at 8.5 seconds and got most of the remaining time.
3. **The last 17 seconds found nothing.** The budget was used up anyway (section 6.8).

The charts were drawn with `python scripts/plot_search_history.py --experiment EXP-001` from the saved trial history. The same command works for any experiment.

---

## 8. What a single experiment cannot do

The baseline works, and its numbers are honest. These are the limits of one experiment, each one seen in our own runs; section 9 shows how the autonomous layer answers them:

| Limit | What we saw |
|---|---|
| FLAML only changes the learner family and its settings. It cannot drop a column, encode one differently, add a feature, rebalance the classes or change the metric | In EXP-001 the winner leaned on `Ticket` (third most important), although 63% of the test tickets were unseen and all became position 571 (section 6.4). Dropping `Ticket` and `Cabin` was our own manual decision, in EXP-002 |
| The whole budget goes into one search, whether or not more time helps | EXP-001 reached 0.785 in 10 seconds, then spent 50 more seconds to gain about 0.016; EXP-002 found nothing new after 23 seconds |
| The validation score is the number FLAML optimised, and one split is noisy | EXP-001: validation 0.802, test 0.673. Changing only the split moved this setup's test F1 between 0.67 and 0.81 (`report-notes.md`, sections 9.6 and 9.7) |
| Every run starts from zero | Nothing learned in EXP-001 was used in EXP-002 |
| Nothing explains why a model is weak | The app shows the train vs test gap and the data flags, but a person has to read them and decide what to do |
| One click runs one experiment | The comparison of EXP-001 and EXP-002 was done by hand |

---

## 9. The autonomous layer (Phase 2)

Phase 2 puts a loop around the single experiment of sections 1 to 8. One run of the loop is a **Session**. It does what we did by hand between EXP-001 and EXP-002: look at the result, find out what is wrong, change one thing, run again, and keep the change only if it really helps. Every part below is built and tested; the numbers that set its thresholds come from two studies on synthetic data (section 9.6). The parts added by the optional tasks of November 2026 were not part of those studies (sections 9.3 and 9.8).

### 9.1 The loop in one picture

```mermaid
flowchart TD
    START["Session starts:<br/>training part 80%, test part 20%<br/>15 fixed folds of the training part"] --> FIRST["Planner: first Setup<br/>(ID-like and constant columns left out)"]
    FIRST --> SEARCH["FLAML search on the training part<br/>(test rows never used)"]
    SEARCH --> ASSESS["Assessor: refit the best configuration<br/>on each of the 15 folds"]
    ASSESS --> JUDGE{"Judge: candidate against<br/>the Accepted Setup"}
    JUDGE -->|keep or keep_simpler| ACCEPT["The candidate becomes<br/>the Accepted Setup"]
    JUDGE -->|discard| STAY["The Accepted Setup stays"]
    ACCEPT --> MEM[("Experiment memory:<br/>one record per Verdict")]
    STAY --> MEM
    MEM --> CRITIC["Critic: Diagnoses of the Accepted Setup"]
    CRITIC --> STRAT["Strategy engine: ranked Proposals<br/>(changes not tried yet)"]
    STRAT --> RES{"Resource manager:<br/>stop, or the next search budget"}
    RES -->|go on| CAND["Planner: candidate Setup =<br/>Accepted Setup + one change"]
    CAND --> SEARCH
    RES -->|stop| FINAL["Final evaluation:<br/>first and final model on the test part, once"]
```

The first Setup skips the Judge: after its Assessment it becomes the Accepted Setup and goes straight to the Critic.

### 9.2 Words used in the loop

| Word | Meaning |
|---|---|
| Session | One run of the loop on one dataset, with one budget. Its record is `experiments/SES-###/session.json` |
| Setup | Everything that defines one Experiment except the search result: the columns used, the data changes, class weights, limits on the search space, the learner families, the search metric, the time budget |
| Accepted Setup | The best Setup so far. A candidate replaces it only if the Judge says so |
| Session folds | 3 repeats of 5-fold cross-validation on the training part, made once per Session: 15 folds, the same for every Setup |
| Assessment | The best configuration of a Setup's search, refitted on each of the 15 folds (no new search), with its score on each fold |
| Diagnosis | A problem the Critic finds, with a kind, a severity (low, moderate, high), the columns involved, the evidence numbers and a sentence |
| Strategy, Proposal | A Strategy is a kind of change (for example "drop a column"); a Proposal is one concrete change for this Setup (for example "drop `Ticket`"), with the Diagnoses it answers |
| Verdict | The Judge's answer: `keep` (clearly better), `keep_simpler` (simpler and not worse) or `discard` |
| Test part | The 20% of rows set aside at the start. Only the final evaluation reads it |

### 9.3 The parts, one by one

**Splits** (`app/services/splits.py`). The training and test parts are made exactly as in Phase 1, with the user's seed, so a Session and a single experiment with the same seed use the same rows. The 15 folds are made from the training part only, with seed + 1, stratified for classification.

Some data has several rows per group, for example per patient, per store or per family, and new rows will come from new groups. A model scored on rows of groups it has already seen then looks better than it will be. For such data you can name a grouping column (optional task 31). Whole groups then go to the test part (so the test share counts groups, not rows, and the test part is not stratified), and in each of the 3 repeats the groups of the training part are dealt into 5 folds, keeping the class balance where possible, so a group is never split between the training and validation rows of a fold. The grouping column is never a feature, and the choice is recorded in `session.json`. Without a grouping column nothing changes.

**Planner** (`app/services/experiment_planner.py`). The first Setup is the user's settings with the identifier-like and constant columns of the training part left out. Each later candidate is a copy of the Accepted Setup with one Strategy applied. Its search starts from the best configuration of each learner family found so far (a warm start), and it gets the search time the Resource manager gives it.

**Assessor** (`app/services/assessor.py`). FLAML's own validation score is the number it optimised, from its own folds, so it is a little optimistic and cannot be compared fairly between two searches. The Assessor takes the winning configuration of a search and trains it again on the training rows of each of the 15 folds, then scores it on that fold's validation rows. When a later Setup would refit exactly the same model (same columns, data changes, class weights, learner and settings), its fold scores are reused instead of refitted, and the record says which Experiment they come from. Data changes (below) are learned again inside each fold, from that fold's training rows only. It records every metric on the fold's training and validation rows, the recall of each class, the share of validation rows with a category the fold's training rows never had, and the trivial baseline's score. Two Setups are therefore compared on exactly the same rows. A Setup whose configuration was assessed before is not refitted again.

**Critic** (`app/services/critic.py`, thresholds in `critic_thresholds.toml`). Fourteen rules read the training-part profile, the Assessment and how the search spent its time. "m" is the primary metric (or F1, accuracy, ROC-AUC or R2 when the primary metric is a loss), and all values are means over the 15 folds. Thresholds are version `v1-critic-study`:

| Diagnosis | Fires when | Severity |
|---|---|---|
| overfitting | training m minus validation m is at least 0.11 | moderate; high from 0.20 |
| underfitting | the gap is below 0.03 and the skill over the trivial baseline is below 0.50 | moderate; high below 0.25 |
| budget_limited | the search's best score still rose by at least 0.005 in the last quarter of its time | moderate; high from 0.02 |
| wasted_budget | at least half of the search time came after the best configuration was found | moderate; high from 75% |
| class_imbalance | the largest class is at least 3 times the smallest, or (from 1.5 times) the smallest class's recall is below 0.5 | moderate; high from 9 times, or recall below 0.3 |
| missing_values | a used column is at least 30% empty | moderate; high from 70% |
| high_cardinality | a used text column has more than 50 values, or on average 30% of validation rows have a value the fold never saw | moderate; high from 60% unseen |
| identifier_column | a used column looks like an ID on the training part | high |
| constant_column | a used column has one value, or the same value in at least 98% of rows | low |
| redundant_features | two used numeric columns have a correlation of at least 0.95 | moderate; high from 0.99 |
| unreliable_validation | the validation score varies over the folds with a standard deviation of at least 0.05 | moderate; high from 0.08 |
| weak_model_families | a learner family is at least 0.10 behind the best family and used at least 5% of the search time | low |
| unused_words | a text column has frequent words (found in at least 2% of the rows and in at least two different values, but not in every row) and no encoding (frequency, grouping, target encoding or word features) has changed it | low if the model uses the column (it sees each value only as a whole); moderate if the first Setup left the column out as identifier-like |
| possible_leakage | a used column alone predicts the target almost perfectly: a score of at least 0.99 in 5-fold cross-validation on the training part (balanced accuracy, or R2 for regression) | high; a warning only |

The last two rows came with the optional tasks 30 and 33 (November 2026) and were not part of the critic study (section 9.6). unused_words uses the word rule of the word features below and reads no threshold from the file. The 0.99 of possible_leakage was set before any run and not tuned. A column the user excluded is never diagnosed by unused_words, so it is never brought back.

**Strategies and the Strategy engine** (`app/services/strategies.py`, `app/services/strategy_engine.py`). Twelve Strategies, each a small change to a copy of the Setup (the last three came with the optional tasks 29 and 30):

| Strategy | What it changes | Answers |
|---|---|---|
| drop_columns | leaves columns out | identifier, constant, high-cardinality, mostly empty (if high) and redundant columns; overfitting together with a high-cardinality column |
| frequency_encode | replaces a text column by how often each value occurs | high_cardinality |
| group_rare_categories | puts rare values of a text column into one group | high_cardinality |
| missing_indicator | adds a "was empty" column for a numeric column | missing_values |
| class_weight_balanced | weights the rows so that every class counts the same | class_imbalance |
| regularised_search_space | smaller trees: at most 32 leaves for LightGBM and XGBoost, at least 10 rows per LightGBM leaf, depth at most 4, at most 64 leaves for the forests | overfitting |
| restrict_model_families | searches only the families that were competitive | weak_model_families |
| more_search_time | the same Setup with up to twice the search time | underfitting, budget_limited |
| smoother_search_metric | lets FLAML optimise log loss while the primary metric stays the same | unreliable_validation, class_imbalance |
| smote | adds synthetic rows of the smaller classes to the training rows only, until every class is as large as the largest (SMOTE) | class_imbalance (after class weights and the smoother metric) |
| target_encode | replaces a text column by the smoothed mean target of its value, cross-fitted on the training rows | high_cardinality (after dropping, frequency encoding and grouping) |
| token_features | replaces a text column by one 0/1 column for each of its frequent words (at most 20) | unused_words |

The engine turns the Diagnoses into Proposals with a fixed rule table and ranks them: the most severe Diagnosis answered first, then the cheaper change, then the order of the table. A change already tried in the Session, one that does not apply (for example class weights in regression) and one that conflicts with the Accepted Setup are left out. smote and class weights exclude each other, because FLAML does not take a resampler together with row weights. wasted_budget has no Strategy: the Resource manager shortens the next search instead. possible_leakage has none either, because only the user knows whether the column is known before the outcome.

**Data changes** (`app/services/transforms.py`). Frequency encoding, grouping rare values, missing indicators, target encoding and word features are learned from training rows only (from each fold's training rows during an Assessment) and saved together with the model, so a saved model still takes the original columns.

Target encoding is the only data change that reads the target. Each value of the column becomes (sum of the targets of its rows + 10 x the overall mean) / (number of its rows + 10), with the target as 1 for the positive class (binary), one 0/1 column per class (multiclass) or the value itself (regression), so a rare value gets almost the overall mean. On the training rows themselves the encoding is cross-fitted: the rows are split into 5 parts and each part is encoded from the other 4, so no row's encoding uses its own label and the model cannot learn its own answer. Other rows (validation, test and new rows) use the encodings learned from all training rows. The 10 and the 5 parts are design choices, not tuned.

Word features split each value into runs of letters and runs of digits, in lower case ("Braund, Mr. Owen Harris" gives braund, mr, owen, harris). They keep the words found in at least 2% of the training rows and in at least two different values, but not in every row, at most the 20 most frequent, and replace the column by one 0/1 column per kept word. For a column the first Setup left out as identifier-like, such as Titanic's `Name`, the Strategy brings it back with only its words. On Titanic's training part (seed-42 split) the words of `Name` begin with mr, miss, mrs, william, john, master, and `Cabin` gives the deck letters c, b, d, e. There is no Titanic-specific code: the same rule runs on every text column.

**Resampling** (`app/services/resampling.py`). The smote Strategy gives FLAML a resampler. FLAML applies a fresh copy of it to the training rows of each of its search folds and to the rows of the final model, never to validation rows, and the Assessor does the same in every refit, so only a fold's training rows are resampled and its validation rows are scored as they are. FLAML has already turned text columns into category columns at that point, which plain SMOTE refuses, so `FoldSafeSmote` uses SMOTE for numeric columns only, SMOTE-NC for mixed columns and SMOTE-N for text columns only. The original rows come back first and unchanged, followed by the new ones.

**Judge** (`app/services/judge.py`). For each of the 15 folds, d = candidate score minus Accepted score, signed so that positive is better. The folds share most of their training rows, so their scores are not independent; the corrected resampled t-test (Nadeau and Bengio, 2003) allows for this by using (1/15 + 1/4) times the variance of d instead of 1/15 of it. Then:

- `keep` if the mean of d is above 0 and p(better) is at most 0.20;
- otherwise `keep_simpler` if the candidate is simpler (for example it uses fewer columns), the mean of d is at least -0.005 and p(worse) is above 0.20;
- otherwise `discard`.

**Resource manager** (`app/services/resource_manager.py`). The first search gets 35% of the total time. Every later search gets the time left divided by the Experiments left, where "time left" is measured, not counted from the budgets handed out, and keeps back 1.2 times the previous step's Assessment and other time for the next Assessment. After a wasted_budget Diagnosis the next search gets at most 1.5 times the time the last search needed to find its best. The Session stops, in this order, when Stop is pressed, after an error, when the time left is under 5 s, after the maximum number of Experiments, when no Proposal is left, or after three discards in a row.

**Experiment memory** (`app/services/experiment_memory.py`). After every Verdict one line is added to `experiments/memory/memory.jsonl`: a description of the dataset (its meta-features), the Diagnoses answered, the Strategy, the Verdict, the Judge's mean difference and its spread, and the cost in seconds. It can already find the most similar past datasets and summarise how each Strategy did on them. Since task 21 (Phase 3) it can also choose the next change: with the memory switched on, the `thompson` and `ucb1` selectors re-rank the Proposals by how each pair of Strategy and diagnosis kind did on the 5 most similar other datasets (Thompson sampling: a sampled keep probability times the mean gain when kept, per second of cost; UCB1: the keep rate plus an exploration bonus). A pair with fewer than 3 past records, or with records from only one dataset, keeps its rule-order place, and without the memory the order is the rule order. A step that was kept but changed no fold score (in practice: the search found the same model) counts as not kept, since it shows nothing about whether the change helps.

**Switches for the ablations** (`SessionSettings`, task 22, Phase 3). To measure what each part adds, a Session can run with one part turned off: `selector="random"` tries the applicable changes in a random order; `use_critic=False` makes no Diagnoses and picks random valid changes instead; `use_resource_manager=False` gives every search an equal share of the time and stops only on time or on the number of Experiments; and without `use_memory` the learned choice keeps the rule order. The random choices use the Session's seed, so such a run can be repeated, and every switch is recorded in `session.json`.

**Records** (`app/services/experiment_tracker.py`, `app/services/autopilot.py`). `session.json` is rewritten after every step, so a crash keeps the finished steps. Each Experiment of a Session keeps its `summary.json` and also `assessment.json`, `diagnoses.json` and `decision.json`, so every Diagnosis and Verdict can be worked out again from the files. In MLflow each Session is a parent run with its Experiments as child runs.

**Dashboard and command line** (`app/components/autopilot_viewer.py`, `scripts/run_autopilot.py`). In the sidebar choose **Autonomous run**: set the time, the maximum number of Experiments and the CPU cores, press Start, and watch the steps, the score chart and the Diagnoses; Stop ends the Session after the current step. At the end the first and final model are shown side by side on the test part. The same Session runs from the command line with `scripts/run_autopilot.py`. Since the optional tasks, the advanced settings have a group column option (`--group-column` on the command line), and a finished Session has two buttons: "Explain the final model" (SHAP) and "Write a summary" (a local language model). Both are described in section 9.8.

### 9.4 One Session, step by step

1. Check the settings, drop rows without a target, and make the training and test parts and the 15 folds. Profile the training part only.
2. Reserve `SES-###` and write `session.json` with the status "running".
3. First Experiment: search the first Setup with 35% of the time, assess it on the 15 folds and diagnose it. It becomes the Accepted Setup. If this fails, the Session ends with an error record.
4. Ask the Strategy engine for Proposals and the Resource manager whether to go on and with how much time. Build the candidate, search it, assess it and let the Judge compare it with the Accepted Setup. A kept candidate becomes the Accepted Setup and is diagnosed. A failed candidate counts as a discard and the Session goes on. Write one memory record per Verdict and rewrite `session.json`.
5. Repeat step 4 until a stop rule applies.
6. Final evaluation, the only time the test part is read: score the first and the final model once each.

### 9.5 Keeping the test part out of every decision

Inside a Session, experiments run with `evaluate_test=False`: no test score, test predictions, importance on test rows or test notes are made, and the profile behind the Diagnoses is made from the training part only. The Session checks every Experiment result for this, and a test feeds the Session marked rows to show that no test row reaches FLAML, the Assessor or the Critic.

### 9.6 Where the thresholds and settings come from

- **Critic study** (`research/critic_study.py`): 270 synthetic datasets with one planted problem each (or none) and a fresh sample of 20,000 rows that gives the true answer. Thresholds were tuned on five seeds and checked on five others. The data-level Diagnoses had no false alarm; overfitting reached precision 0.88 and recall 0.91; unreliable validation was weak (precision 0.27). Details: `report-notes.md`, section 9.12.
- **Judge study** (`research/judge_study.py`): 84 pairs of Setups whose true difference was measured on a fresh sample. Alpha 0.20 had the most power without more harmful changes kept, so it became the default; 15 folds were kept, because 25 accepted the same changes at five thirds of the cost. On the reporting seeds: 0 of 18 harmful changes kept, 10 of 12 helpful ones kept as better. Details: section 9.13 of the report notes.

### 9.7 What the autonomous layer cannot do yet

| Limit | Why it matters |
|---|---|
| By default the order of changes is a fixed rule table; the learned order needs records of similar datasets in the memory | Without such records, a change that never helps on this kind of data is still tried; the first benchmark run was stopped too early to show whether the learned order helps, and in the second, full run its difference to the rule order was within the intervals over 6 datasets |
| One change at a time, each tried at most once | Two changes that only help together are not found |
| Each Assessment refits a configuration 15 times | It takes a share of the budget, more on bigger data |
| Both scores the Judge compares come from configurations FLAML chose on the same training part | Both are a little optimistic; pairing removes much of this, not all. Searching again on 5 inner splits gave the same verdict for only 8 of 20 real decisions (report notes, section 9.18) |
| FLAML picks learner families using measured times | Two runs with the same seed can take different paths and end with different models |
| The unreliable-validation Diagnosis is weak in the critic study | Its Proposal is cheap and still has to pass the Judge |
| The newest Strategies (smote, target_encode, token_features) and Diagnoses (unused_words, possible_leakage) were not part of the studies or the first benchmark run | They were checked with short runs; in the second benchmark run token_features was kept 3 of 26 times, smote 2 of 4 and target_encode 0 of 7, and their effect on the test score is not measured on its own |
| possible_leakage looks at one column at a time and only warns | A column that gives the answer away only together with other columns is not found, and a flagged column is fine if its value is known before the outcome; only the user can tell |

The Phase 2 measurement against FLAML alone is in `report-notes.md`, section 9.14.

### 9.8 Parts around the loop (optional tasks, November 2026)

**Target suggestion** (`TaskDetector.suggest_targets`, task 33). Among the columns that can be a target (not constant, not identifier-like, not text with very many values), a column gets points for a generic target name such as target, label, class, y, outcome or response (3), or such a word inside its name (2), for being the last column (1), for holding only 0 and 1 or true and false (1), and for having no empty cells (0.5); ties go to the later column. The best three are shown with their reasons in the target candidates tab, and the first is the default choice in the Target section. You always confirm or change it. The points are design choices, not measured. On 15 of the 16 registered datasets the registered target came first; on penguins it came third, after `sex` (the last column) and `island`.

**Leakage check** (`single_column_scores` in `app/services/profiler.py`, task 33). Target leakage is a column that gives the answer away, for example one filled in after the outcome: a model built on it looks excellent and fails on new data. The check scores each feature column alone in 5-fold cross-validation (at most 5,000 rows): a numeric column with a small decision tree (and, for regression, also a monotone fit, the better of the two counting), any other column by the most common target of each value (the mean for regression). The score is balanced accuracy, so a constant guess gets 1 / classes whatever the class balance, or R2 for regression. A score of at least 0.99 gives a warning in the Target section before any run, and in a Session a used column with such a score on the training part gets the possible_leakage Diagnosis. Nothing is changed automatically. Run on the training part of the seed-42 split of all 16 registered datasets (12 November 2026, `scripts/check_single_columns.py`, `results/leakage_check.csv`), no column reached 0.99, so there was no false alarm; the highest scores were 0.905 (energy_efficiency, R2) and 0.900 (breast_cancer, balanced accuracy).

**SHAP explanations of the final model** (`app/services/explainer.py`, task 32). SHAP (Lundberg and Lee, 2017) splits one prediction into one share per input column; the shares add up to the prediction minus the average prediction. The explainer rebuilds the Session's training part from the dataset and the recorded settings and refuses the data if the rebuilt split's fingerprint differs from the record, so the test part is never used. It explains at most 200 rows, passed along the final model's own path (its data changes, then FLAML's preparation), so the columns it shows are the learner's own inputs, for example `Name__tok_mr` after word features. Tree learners (LightGBM, XGBoost, random forest, extra trees) get shap's exact tree method: on Titanic, penguins and a 1,500-row sample of California housing it took at most 1.9 s for 200 rows (12 November 2026, seed 1, 4 cores; the slowest was the first call, which includes loading shap). Other learners get a slower permutation estimate on 50 rows against 25 background rows. The dashboard shows the mean size of the shares of the top 15 columns and the shares for one chosen row; `scripts/explain_session.py` writes `experiments/SES-###/shap_summary.json`.

**Summaries in plain words** (`app/services/narrator.py`, task 34). The language model explains; it never decides. Facts are taken from the Session record only (the data and target, the budget, the first Setup's Diagnoses, each step's change and its outcome in words, the changes kept, the stop reason, the first and final test scores; no data rows) and sent with fixed instructions to `qwen2.5:7b-instruct`, running on this machine through Ollama, at temperature 0 with a fixed seed. Any address other than this machine is refused. The numbers in the text that are not in the facts are listed next to it. The text is saved in `experiments/SES-###/summary_generated.json`, apart from the record, with a label saying that it is generated text that no person checked and no decision used, and no module of the loop reads it (a test checks this). In the first live check (12 November 2026) the model misread a verdict code, so outcomes are now given in words; the second summary described the kept change correctly and had no number outside the facts, but still added judgements of its own.

**Local HTTP API** (`app/api.py`, `scripts/run_api.py`, task 35). The same Session code over HTTP, with FastAPI: `GET /health`, `GET /sessions`, `GET /sessions/{id}` (the whole `session.json`), `POST /sessions` (starts one Session in the background and returns its ID; one at a time) and `POST /sessions/{id}/stop`; the endpoints are documented at `/docs`. The service has no login, so it listens on 127.0.0.1 by default, a dataset must be a CSV file inside the project's `data/` folder (checked before anything runs), session IDs must look like `SES-001`, so no path can be passed as an ID, and the request fields have ranges (for example 10 s to 24 h of budget).

**Docker image** (`Dockerfile`, `scripts/docker_entry.sh`, task 35). Python 3.11 slim with the code, the bundled samples and the pinned packages, but no results. It runs the API on port 8000 and the dashboard on 8501 as a non-root user, and keeps `experiments/`, `models/` and `mlruns/` in volumes. Publish its ports on 127.0.0.1 only. The local summaries are not available inside it, and its records have no commit hash, because the image holds no git history. Built and checked on 12 November 2026 (2.12 GB).

---

## Appendix A. Where each step lives in the code

| Step | File | Function or class |
|---|---|---|
| Read the CSV | `app/components/dataset_uploader.py` | `read_csv_bytes`, `render_dataset_uploader` |
| Profile | `app/services/profiler.py` | `DatasetProfiler.profile`, `analyze_class_balance`, `DatasetProfile.meta_features` |
| Detect the task | `app/services/task_detector.py` | `TaskDetector.detect`, `TaskDetector.validate` |
| Suggest the target | `app/services/task_detector.py` | `TaskDetector.suggest_targets`, `suggest_target` |
| Leakage check | `app/services/profiler.py` | `single_column_scores` |
| Dashboard sections | `app/components/` | `render_dataset_profile`, `render_target_selection`, `render_experiment_config`, `render_run_section`, `render_results`, `render_explainability`, `render_experiment_history` |
| Background worker | `app/components/experiment_viewer.py` | `_start_run`, `_wait_for_run` |
| Run an experiment | `app/services/automl_runner.py` | `AutoMLRunner.run`, `run_experiment`, `_prepare`, `_split`, `_execute` |
| Unseen labels | `app/services/automl_runner.py` | `unseen_category_warnings` |
| Call FLAML | `app/services/automl_runner.py` | `FLAMLEngine.fit`, `_read_history`, `_leaderboard` |
| Score | `app/services/evaluator.py` | `Evaluator.evaluate`, `compute_metrics`, `baseline_metrics`, `feature_importance` |
| Save the model and files | `app/services/automl_runner.py` | `_save_model`, `_write_artifacts` |
| Track | `app/services/experiment_tracker.py` | `allocate_experiment_id`, `log_run`, `save_record`, `history_frame` |
| Settings | `app/core/config.py` | `Settings`, `get_settings` |
| Session loop and final evaluation | `app/services/autopilot.py` | `AutopilotSession.run`, `SessionSettings`, `_final_evaluation` |
| Training and test parts, folds | `app/services/splits.py` | `make_outer_split`, `make_session_folds`, `group_codes` |
| First and candidate Setups | `app/services/experiment_planner.py` | `Planner.first_setup`, `Planner.candidate` |
| Assessment | `app/services/assessor.py` | `Assessor.assess` |
| How a search spent its time | `app/services/search_summary.py` | `summarize_search` |
| Diagnoses | `app/services/critic.py` | `Critic.diagnose`, `CriticThresholds.load` |
| Strategies and their ranking | `app/services/strategies.py`, `app/services/strategy_engine.py` | `CATALOGUE`, `propose`, `RulesSelector.rank` |
| Data changes | `app/services/transforms.py` | `DataPipeline`, `PipelineModel`, `frequent_tokens` |
| SMOTE | `app/services/resampling.py` | `FoldSafeSmote`, `make_resampler` |
| Verdicts | `app/services/judge.py` | `Judge.compare`, `corrected_resampled_t` |
| Budgets and stop rules | `app/services/resource_manager.py` | `ResourceManager.first_budget`, `next_budget`, `check_stop` |
| Memory | `app/services/experiment_memory.py` | `ExperimentMemory.add`, `similar`, `strategy_stats` |
| Autonomous mode in the dashboard | `app/components/autopilot_viewer.py` | `render_session_settings`, `render_session_run`, `render_memory_records`, `_render_shap`, `_render_summary` |
| SHAP of the final model | `app/services/explainer.py` | `explain_session`, `explain`, `ShapExplanation` |
| Summary in plain words | `app/services/narrator.py` | `session_facts`, `narrate`, `unsupported_numbers`, `save` |
| HTTP API | `app/api.py` | `create_app`, `SessionRequest`, `dataset_path` |

## Appendix B. Commands

Run these from the `autopilot-ml` folder with the `automl` environment active (`conda activate automl`).

```bash
# dashboard, then open http://localhost:8501
streamlit run app/main.py --server.address localhost --browser.gatherUsageStats false

# MLflow, then open http://127.0.0.1:5000
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000 --workers 1

# tests
python -m pytest -q

# compare two setups over several splits
python scripts/compare_seeds.py --dataset data/sample/titanic.csv --target Survived --exclude-a PassengerId,Name --exclude-b PassengerId,Name,Ticket,Cabin --seeds 1 2 3 4 5 --budget 30

# charts of one experiment's search (written to docs/imp/images)
python scripts/plot_search_history.py --experiment EXP-001

# one Session with whole groups kept together (here: passengers sharing a ticket)
python scripts/run_autopilot.py --dataset data/sample/titanic.csv --target Survived --seconds 300 --group-column Ticket

# SHAP of a finished Session's final model (the CSV file the Session ran on)
python scripts/explain_session.py SES-001 --dataset data/sample/titanic.csv

# plain-language summary of a finished Session (needs Ollama running, with: ollama pull qwen2.5:7b-instruct)
python scripts/summarise_session.py SES-001

# leakage check and target suggestion on every registered dataset (written to results/leakage_check.csv)
python scripts/check_single_columns.py

# local HTTP API, then open http://127.0.0.1:8000/docs
python scripts/run_api.py

# Docker image with the dashboard and the API (publish the ports on 127.0.0.1 only)
docker build -t autopilot-ml:0.1.0 .
docker run --rm -p 127.0.0.1:8501:8501 -p 127.0.0.1:8000:8000 \
    -v autopilot-experiments:/app/experiments -v autopilot-models:/app/models -v autopilot-mlruns:/app/mlruns \
    autopilot-ml:0.1.0
```

## Appendix C. Sources

FLAML 2.7.0 source files read for section 6 (in the `automl` environment, under `Lib/site-packages/flaml/`):

| File | What it showed |
|---|---|
| `config.py` | Fixed values: 5 folds, FLAML's own seed 1, the holdout share of 0.1, the cross-validation and sampling thresholds |
| `automl/automl.py` | The search loop, how a learner family is picked, early stopping, the cross-validation rule, retraining after the budget |
| `automl/state.py` | Each family's records and its expected cost of the next improvement |
| `automl/data.py` | Data preparation (`DataTransformer`) |
| `automl/task/generic_task.py` | The default learner list, the folds and the holdout, the averaging of fold scores |
| `automl/model.py` | Each learner's settings, ranges, starting values and starting cost guess; how categories become numbers |
| `tune/searcher/flow2.py`, `tune/searcher/blendsearch.py` | CFO: steps, opposite steps, shrinking, restarts, and the use of our seed |

Papers:

- Wang, C., Wu, Q., Weimer, M., and Zhu, E. (2021). FLAML: A Fast and Lightweight AutoML Library. MLSys.
- Wu, Q., Wang, C., and Huang, S. (2021). Frugal Optimization for Cost-related Hyperparameters. AAAI. (The paper behind CFO.)

Our own data: the trial logs and summaries in `experiments/EXP-001/` and `experiments/EXP-002/`, and checks of FLAML's data preparation on the EXP-001 split, run on 22 October 2026.
