# AutoPilot-ML: project walkthrough and demo guide

This file explains the whole project in one place: the idea, the final goal, where FLAML fits in, what is built so far, what comes next, and exactly how to run the demo.

Part A is the explanation. Part B is the demo script and the checklist for the day.

---

# Part A: The idea

## 1. The project in one minute

AutoPilot-ML is a system that runs machine learning experiments on its own.

You give it a table of data (a CSV file), tell it which column to predict, and give it a budget such as "10 minutes on my laptop". The finished system will:

1. train a first model,
2. look at what went wrong,
3. decide what to try next,
4. try it,
5. repeat until the budget runs out or nothing is improving any more,

and then hand back the best model together with a record of everything it tried and why.

The first stage is working today: a baseline that profiles the data, works out what kind of problem it is, uses FLAML to search for a good model, evaluates that model properly, explains it, and records every experiment. The "decide what to try next" part is the research part, and that is what we build next.

## 2. The problem we are solving

Think about how a person actually works on a dataset like Titanic.

- They train a model and get a test F1 of about 0.67.
- They notice the training F1 is about 0.93. The model does much better on data it has seen than on new data, so it is overfitting.
- They look at the columns and see that `Ticket` has 681 different values in 891 rows. The model is probably memorising ticket numbers.
- They remove `Ticket` and `Cabin`, train again, and compare the two runs.
- They decide whether the change helped, and whether more training time is worth it.

AutoML tools like FLAML automate one part of this very well: choosing the model and its hyperparameters. Everything around it (spotting the problem, choosing the fix, comparing runs, deciding when to stop) is still done by hand. And when the project ends, everything learned ("dropping high-cardinality columns helped on this kind of data") is forgotten. The next dataset starts from zero.

That outer loop is what AutoPilot-ML automates.

## 3. The final goal

The finished system takes:

- a CSV file and the target column,
- a budget: maximum time (for example 10 minutes), maximum number of experiments (for example 15), and CPU cores (for example 2).

It gives back:

- the best pipeline it found (data treatment + model + hyperparameters),
- an experiment log: what it tried, what it diagnosed, why it chose each next step, and what each step cost.

To do that it needs four abilities that plain AutoML does not have:

| Ability | Module (planned) | Example |
|---|---|---|
| Diagnose | Critic | "Train F1 0.91, test F1 0.69: overfitting." "Majority class is 92%: severe imbalance." |
| Decide | Strategy engine | For overfitting plus a high-cardinality column, try dropping the column first, then a more regularised search. |
| Remember | Experiment memory | "On datasets that looked like this, class weighting improved F1." Use that to choose faster next time. |
| Budget | Resource manager | "The last two experiments gained almost nothing for their cost. Stop, or try something different." |

The research question behind all of this:

> Can an autonomous experimentation layer improve the efficiency and effectiveness of conventional AutoML by learning from previous experiments, diagnosing model weaknesses, selecting appropriate improvement strategies, and allocating computational resources adaptively?

In plain words: with the same laptop and the same time limit, can we reach a good model faster (or reach a better one) than just running FLAML for the whole budget? We will measure this, not assume it.

## 4. Where this started: FLAML

This project started from my work with FLAML, the open-source AutoML library from Microsoft Research (Wang, Wu, Weimer and Zhu, "FLAML: A Fast and Lightweight AutoML Library", MLSys 2021; MIT licence).

What I liked about FLAML is that it finds good models quickly on an ordinary laptop. It does this by paying attention to the cost of each trial: its search methods (CFO and BlendSearch) start with cheap configurations and only move to expensive ones when that looks worthwhile. No GPU and no paid service are needed.

While using it, I noticed that FLAML answers one question very well: *which model and hyperparameters?* The questions I kept asking after every run (why is this model weak, what should I change, was the extra time worth it) were still mine to answer. That gap became this project.

So FLAML is the engine of AutoPilot-ML. We use it unchanged, as a normal dependency. We have not modified FLAML, and we do not claim to beat it. Comparing against FLAML on its own is part of the evaluation plan.

## 5. What the baseline adds on top of a plain FLAML run

A plain FLAML run takes `X` and `y`, searches for a while, and returns the best model and its internal validation score. Around that, the baseline adds what is needed to judge experiments fairly and to build the autonomous layer later:

| Area | Plain FLAML run | AutoPilot-ML baseline |
|---|---|---|
| Understanding the data | Not part of the run | Profile of every column: types, missing values, duplicates, unique counts. Flags for identifier columns, constant columns, high-cardinality columns, heavy missingness, class imbalance |
| Problem type | You pass it in | Detected from the target column with a reason and confidence. You can override it, and impossible overrides are rejected with a clear message |
| Metric | You pass it in | Chosen per task (F1, ROC-AUC, R², RMSE...). Warns when accuracy would be misleading because of class imbalance |
| Honest evaluation | Internal validation score, which is the same number the search was optimising | A separate 20% test set that FLAML never sees, scored once at the end. Plus a trivial baseline (always predict the majority class, or the mean) for comparison, and train vs test scores to show overfitting |
| Data problems at prediction time | A warning printed on every prediction | One clear note per column, for example "Ticket: 63% of test rows have categories never seen in training" |
| Explanation | Importances of the final model on FLAML's internal features | Permutation importance on the original columns, confusion matrix, ROC curve, residual plots |
| Search visibility | A log file | Leaderboard per model family and a chart of every trial over time |
| Record keeping | Nothing kept between runs | Every experiment logged to MLflow (parameters, metrics, model, files) and saved as a small JSON record. Experiment history in the app |
| Ease of use | Python code | A dashboard. Upload a CSV and click Run |
| Engine | FLAML only | FLAML sits behind a small engine interface, so AutoGluon can be plugged in later for comparison |

The JSON records and the train/test numbers are not just for display. They are the raw material the critic and the experiment memory will work from.

## 6. How the baseline works, step by step

```text
CSV ─► Profiler ─► Task detection ─► Split 80/20 ─► FLAML search ─► Evaluation ─► Explanation ─► Save + track ─► Dashboard
```

When you click Run on Titanic:

1. **Check the setup.** The target exists, the budget is between 5 and 3600 seconds, there are at least 20 labelled rows, and the metric suits the task.
2. **Prepare the data.** Rows with a missing target are dropped and the excluded columns are removed (by default `PassengerId` and `Name`, because they are identifiers).
3. **Split.** 80% for training (712 rows), 20% for testing (179 rows). The split keeps the same survived/died ratio in both parts.
4. **Search with FLAML.** For the time budget, FLAML tries different model families (LightGBM, XGBoost, random forest, extra trees, linear models) and hyperparameters. Each trial is scored by 5-fold cross-validation on the training part. Larger datasets use a holdout split instead, and FLAML chooses which.
5. **Retrain the best one.** The best configuration is retrained on all the training rows.
6. **Evaluate once on the test set.** All metrics, the trivial baseline, train vs test scores, the confusion matrix.
7. **Explain.** Permutation importance: shuffle one column in the test set and measure how much F1 drops. A big drop means the model relies on that column.
8. **Save and record.** The model goes to `models/EXP-001/model.pkl`. MLflow gets a run with all parameters, metrics and files. `experiments/EXP-001/summary.json` gets the summary, and the history table reads from it.

## 7. Concepts you should be ready to explain

- **Time budget.** How long FLAML may search. A longer budget means more trials, but the gains get smaller.
- **Primary metric.** The score FLAML optimises and the headline number. For binary problems the default is F1.
- **Why not accuracy?** If 92% of rows are "no", a model that always says "no" gets 0.92 accuracy and is useless. F1 and ROC-AUC do not have this problem. The app warns when accuracy would be misleading.
- **F1.** The balance between precision (when we predict "survived", how often are we right) and recall (of the people who survived, how many did we catch).
- **ROC-AUC.** How well the model ranks positives above negatives. 0.5 is a coin flip, 1.0 is perfect.
- **Cross-validation vs holdout.** CV splits the training data into 5 parts and rotates which part is used for validation. That is more reliable on small data. Holdout uses one fixed validation part, which is faster on big data.
- **Validation score vs test score.** The validation score is what FLAML optimised, so it is slightly optimistic. The test score comes from rows that were never used in the search. That is the honest number.
- **Trivial baseline.** For Titanic, "everyone died" gets 0.62 accuracy and F1 = 0. Any useful model has to beat this, and the app shows the difference.
- **Train vs test gap.** A large gap means overfitting. Detecting this automatically is the first job of the critic.
- **High cardinality.** A text column with very many different values, like `Ticket`. The model cannot learn from values it has only seen once.
- **Permutation importance.** Explained in step 7 above. It works for any model.
- **MLflow.** A standard open-source tool for recording ML experiments. We run it locally with a SQLite file, so no server is needed.
- **Meta-features.** Numbers that describe a dataset (rows, columns, share of missing values, imbalance ratio...). The memory will use them to find "similar datasets".

## 8. Where we are going next

The planned modules, in the order we will build them:

1. **Critic.** Reads a finished experiment and produces structured diagnoses: overfitting, underfitting, class imbalance, missing values, high-cardinality or redundant columns, wasted time. Every diagnosis comes with its evidence. Example: `{"overfitting": "high", "evidence": "train F1 0.91, test F1 0.69"}`.
2. **Strategy engine.** Turns diagnoses into candidate fixes and ranks them. First as a fixed rule table, later learned. Example fixes: drop or encode a high-cardinality column, use class weights or SMOTE for imbalance, restrict to more regularised models for overfitting, give more budget for underfitting.
3. **Experiment planner.** Turns the chosen fix into the next experiment configuration and runs it. The runner already accepts a configuration object, so this plugs straight in.
4. **Experiment memory.** Stores dataset meta-features + diagnosis + strategy + model + result + cost. On a new dataset it finds similar past datasets and suggests the strategies that worked there first.
5. **Resource manager.** Splits the total budget between experiments, sets CPU cores, and stops when the expected gain per second gets too small. The idea is to score each experiment as *improvement divided by cost*.

How one cycle will look on Titanic once this is built:

```text
EXP-001  baseline run              test F1 0.67, train F1 0.93
Critic   overfitting: high; 'Ticket' high cardinality, 63% of test rows unseen
Strategy drop 'Ticket' and 'Cabin' (cheapest fix that matches both diagnoses)
EXP-002  run with the change       compare with EXP-001, keep it only if it helps
Memory   record: "high-cardinality + overfitting -> drop column: effect, cost"
Next     next strategy, or stop if the budget or the gains run out
```

In today's demo I do EXP-001 and EXP-002 by hand. That manual step is exactly what the system will automate.

Rough timeline (weeks from the start of the project):

| Weeks | Work |
|---|---|
| 1-2 | Baseline (done) |
| 3-4 | Critic, tested on synthetic datasets with problems we inject on purpose |
| 5-6 | Strategy catalogue and rule-based strategy engine |
| 7-8 | Planner loop, experiment memory, resource manager |
| 9-10 | Learned strategy selection (multi-armed bandits, memory-based), AutoGluon engine |
| 11-12 | Benchmark on public datasets |
| 13-14 | Ablation study, analysis, report |

## 9. How we will know if it works

We compare six systems on the same datasets, time budgets and random seeds:

1. A manual baseline (one sensible model, no search)
2. FLAML alone, given the whole budget
3. AutoGluon, another strong AutoML tool
4. AutoPilot-ML without experiment memory
5. AutoPilot-ML without strategy selection (strategies in random or fixed order)
6. The full AutoPilot-ML

We measure:

- the final test score,
- time to reach a target score,
- number of experiments,
- CPU time,
- how the best score improves over time,
- how consistent the results are across datasets.

Systems 4 and 5 are the ablation study: they show which part of the system actually helps. Datasets will come from OpenML, a public benchmark collection, and the datasets used to fill the memory will be kept separate from the ones used for testing, so the memory cannot cheat. If the full system does not beat FLAML, that is still a valid result and we will report it.

## 10. What is not done yet

- The critic, strategy engine, planner, memory and resource manager are not built. Today, one click means one experiment.
- The user chooses the target column. There is no automatic target detection yet.
- No feature engineering, SMOTE or class weighting yet.
- No SHAP explanations yet. Permutation importance is used instead.
- CSV and tabular data only.
- FLAML's search is time-based, so two runs with the same seed can give slightly different numbers.

---

# Part B: The demo

## 11. Suggested flow of the presentation (about 12 minutes)

| Time | What | Where |
|---|---|---|
| 2 min | The idea and the problem | Sections 1 and 2 |
| 1 min | The final goal and the research question | Section 3 |
| 1 min | FLAML credit and what the baseline adds | Sections 4 and 5 |
| 5 min | Live demo | Section 13 |
| 2 min | Future plan and how we will evaluate it | Sections 8 and 9 |
| Rest | Questions | Section 15 |

## 12. Thirty minutes before the demo

1. Plug in the charger and close heavy programs. FLAML uses all CPU cores during a run.
2. Open **Anaconda Prompt** (or PowerShell, if conda works there) and run:

   ```bash
   conda activate automl
   cd C:\Users\priya\Desktop\project\AutoML\autopilot-ml
   python -m pytest -q
   ```

   Expected result: `47 passed` in about 30 seconds. If you want to show that the code is tested, this is the screen to show.

3. Start the dashboard in the same window:

   ```bash
   streamlit run app/main.py --server.address localhost --server.headless true --browser.gatherUsageStats false
   ```

   Open **http://localhost:8501** in Chrome or Edge. Leave this window open. Closing it stops the app.

4. **Do one full dry run** (steps 2 to 8 of section 13) with a 30-second budget. This checks that everything works on the day and creates the MLflow database. Also try the second experiment (step 8) during the dry run, so you know how the comparison comes out on your laptop before showing it live.
5. Open a **second** Anaconda Prompt for MLflow:

   ```bash
   conda activate automl
   cd C:\Users\priya\Desktop\project\AutoML\autopilot-ml
   mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000 --workers 1
   ```

   Open **http://127.0.0.1:5000**, click the experiment **autopilot-ml-baseline**, and check that the dry-run experiments are listed.
6. Take a few screenshots during the dry run (results, confusion matrix, history, MLflow). They are your backup if anything fails live.
7. Zoom the browser to about 80% (Ctrl and minus) so the cards and charts fit on the projector.
8. Keep both browser tabs open: the dashboard and MLflow.

The dry-run experiments stay in the history. That is fine: they are real experiments, and they make the history table more interesting.

## 13. Live demo, step by step

**Step 1: Introduce the screen.** Point at the left sidebar: it lists what is implemented and what is planned. This shows straight away that we are honest about the current stage.

**Step 2: Section 1 · Dataset.**
- Select **Sample dataset**, then **Titanic passenger survival · Binary classification**.
- Point at the cards: 891 rows, 12 columns, 866 missing values, 0 duplicate rows.
- Open the **Data quality (5)** tab. Read out the flags: `PassengerId` and `Name` look like identifiers, `Ticket` (681 values) and `Cabin` (147 values) are high-cardinality, and `Cabin` is 77% missing.
- Say: "All of this is computed by our profiler before any model is trained. Later, the critic will use these flags."

**Step 3: Section 2 · Target.**
- **Target column** is `Survived`. Under **Detected problem** it says **Binary Classification**, with the reason (a numeric target with exactly 2 values).
- Point at the class distribution chart: 549 died, 342 survived. The imbalance is mild, so accuracy is not badly misleading here.
- Show the validation: change **Problem type** to **Multiclass Classification**. The app refuses with "Multiclass classification needs at least 3 classes". Change it back to **Binary Classification**.
- Point at **Columns to exclude from training**: `PassengerId` and `Name` were pre-selected because they are identifiers.

Do this override check *before* setting the budget. When the error shows, the configuration section is hidden and its values go back to the defaults.

**Step 4: Section 3 · Experiment Configuration.**
- Set **Time budget** to **30** seconds, keep **Primary metric** = F1, **Random seed** = 42, **Test split** = 0.2.

**Step 5: Section 4 · Run AutoML.**
- Click **Run Experiment**. The run takes about 35 seconds.
- While it runs, explain: "FLAML is now trying LightGBM, XGBoost, random forests and linear models with different settings, scoring each one by 5-fold cross-validation on the 80% training part. The 20% test part is locked away until the end."
- Do not click anything else while it runs. Just talk.

**Step 6: Section 5 · Results.**
- The top cards show the best model, the test F1, the validation F1 and the training time. Explain why the test score is the honest one.
- **Test-set metrics:** each card shows "vs baseline". The trivial "everyone died" model gets F1 = 0, so the gain is real.
- Open **Notes and warnings**: "Ticket: 63% of test rows have categories never seen in training". This is the setup for experiment 2.
- **Train vs test** table: point at the gap. In my 60-second run (EXP-001), train F1 was 0.93 and test F1 was 0.67. "This is overfitting. Detecting it automatically is the critic's first job."
- **Leaderboard** and **Search progress:** "FLAML tried this many model families. Each dot is one trial, and the line is the best score so far."

**Step 7: Section 6 · Explainability.**
- **Feature importance:** `Sex` and `Pclass` come out on top, which matches what we know about the Titanic. In my run `Ticket` came third, even though 63% of the test tickets never appear in training. That fits the overfitting story: the model is leaning on ticket numbers it memorised.
- **Confusion matrix:** where the model makes mistakes.
- **ROC curve**, and **Best configuration** (the exact hyperparameters FLAML chose).

**Step 8: The second experiment (the manual version of the future loop).**
- Scroll up to Section 2 and add `Ticket` and `Cabin` to **Columns to exclude from training**.
- Click **Run Experiment** again.
- Compare the two runs. The history table (Section 7) shows test and train F1 side by side. For ROC-AUC and the other metrics, tick both runs in MLflow and press Compare (or open each run and look at its Metrics).

Safer option: you already have EXP-001 and EXP-002 from the dry run (60-second budget, seed 42). Instead of waiting for two more live runs, run one experiment live to show the process, then compare EXP-001 and EXP-002, which are already in the history and in MLflow:

| | EXP-001: defaults | EXP-002: `Ticket` and `Cabin` also excluded |
|---|---|---|
| Test F1 | 0.673 | 0.709 |
| Test ROC-AUC | 0.792 | 0.823 |
| Test accuracy | 0.793 | 0.793 |
| Recall (share of survivors found) | 0.55 | 0.65 |
| Train F1 | 0.934 | 0.873 |
| Train − test gap | 0.26 | 0.16 |
| Validation F1 (FLAML's own score) | 0.802 | 0.782 |
| Trials in 60 seconds | 290 | 501 |

The headline gain is small (+0.036 F1). Don't apologise for it, and don't oversell it. The useful part is how you read it:

1. **On this split, the change helped in the right way.** Overfitting went down (the gap fell from 0.26 to 0.16), and the model now finds 65% of the survivors instead of 55%, at the same accuracy.
2. **FLAML's validation score pointed the wrong way.** EXP-001 had the higher validation F1 (0.802 against 0.782) but the lower test F1. A plain AutoML run keeps whichever setup has the best validation score, so here it would have kept the worse one. A likely reason: families share ticket numbers, so in cross-validation the same ticket turns up in both the training and the validation folds, and the model looks better than it is.
3. **The search got cheaper.** Without the two messy text columns, FLAML managed 501 trials in a minute instead of 290.
4. **One split is not enough to call it.** I repeated both setups on 5 other random splits (seeds 1 to 5, 30-second budget, using `scripts/compare_seeds.py`, which calls the same runner). The average test F1 went from 0.772 to 0.784 (+0.012), and the change helped on 3 of the 5 splits. ROC-AUC went from 0.870 to 0.877 and also improved on 3 of 5. Setup B ran more trials on 4 of 5 splits. The train/test gap did not shrink consistently. Across the six splits I tried (seed 42 plus seeds 1 to 5), the default setup's test F1 ranged from 0.67 to 0.81 just because different passengers landed in the test set. That is far more than the effect of dropping the columns.

So the honest conclusion: dropping `Ticket` and `Cabin` is a small, cheap improvement, not a big one, and a single run cannot tell you that. A system that tries changes on its own has to measure them over several splits and weigh them against their cost. If it doesn't, it will chase noise. Handling this is the job of the critic and the experiment memory, and it is why the research plan uses several seeds and paired statistical tests.

- Say: "I diagnosed the problem and chose this fix myself, and even then the result needed careful reading. The critic and strategy engine will do this automatically, and the memory will record how much a change really helped and what it cost."

**Step 9: Section 7 · Experiment History.**
- The table lists every experiment with model, metric, test score, validation score, budget and time. The chart compares them.
- Say: "These records are the start of the experiment memory."

**Step 10: The MLflow tab.**
- Refresh http://127.0.0.1:5000 and open the latest run. Show the parameters (`hp.*` are the hyperparameters, `meta.*` are the dataset meta-features), the metrics, and under Artifacts the saved model, `profile.json` and `predictions.csv`.

**Optional, if there is time or the professor asks:**
- **Multiclass:** sample dataset **Palmer penguins** (about 35 seconds with a 30-second budget). Shows a 3×3 confusion matrix.
- **Regression:** **California housing** (20,640 rows, about 40 to 45 seconds). Shows predicted vs actual and the residual plots.
- **Upload:** choose **Upload CSV** and pick any CSV, for example `data/sample/breast_cancer.csv`.
- **From code:** the README shows how to run an experiment from Python without the UI.

## 14. If something goes wrong

| Problem | Fix |
|---|---|
| `streamlit` or `mlflow` "not recognized" | The environment is not active. Run `conda activate automl` |
| "Port 8501 is already in use" | Another Streamlit is running. Close it, or add `--server.port 8502` and open http://localhost:8502 |
| A Streamlit "Welcome... Email" prompt in the terminal | Press Enter. With `--server.headless true` it should not appear |
| The page looks stuck or shows an old state | Press F5. Finished experiments are already saved and will show in the history |
| MLflow shows no experiments | The database is created by the first experiment. Run one in the app, then refresh MLflow |
| MLflow crashes with "WinError 10022" | It was started without `--workers 1` |
| A run takes longer than the budget | Normal. Retraining the best model and the evaluation add a few seconds, more on California housing |
| Anything else during the live demo | Switch to the dry-run results in the history and MLflow, or to your screenshots, and keep talking |

Things to avoid during the demo:
- Do not change the dataset or other settings while a run is in progress.
- Do not close the terminal windows.
- Do not use a very long budget live. 30 seconds is enough to show everything.

## 15. Questions I expect, and short answers

**"Isn't this just a wrapper around FLAML?"**
The baseline is built around FLAML on purpose. FLAML is the search engine, and we do not want to rebuild that. What we add in phase 1 is the measurement setup: honest test evaluation, data profiling, tracking and history. The research contribution is the decision layer on top (critic, strategy engine, memory, resource manager), and that is the next phase.

**"Why not just give FLAML more time?"**
FLAML only changes the model and its hyperparameters. It cannot decide to drop a column, rebalance the classes, change the metric, or stop because more search is not paying off. It also starts from scratch on every dataset. "FLAML with the full budget" is one of our comparison baselines, so we will measure whether our layer actually does better.

**"What is new compared to existing work?"**
Auto-sklearn uses meta-learning to choose starting configurations, FLAML is cost-aware inside its search, and recent LLM agents try to run experiments end to end. Our angle is the outer loop: interventions on data, features and model family, chosen from explicit diagnoses, using a memory across datasets and a cost-aware utility. It runs locally, without paid APIs, and is evaluated with ablations.

**"Why no LLM? Everyone is building AI agents."**
The decisions need to be measurable, reproducible and free to run. An LLM is not needed to detect a train/test gap or to choose class weights for an imbalanced target. A local LLM may come later, only to explain diagnoses in plain language, and it will be evaluated separately.

**"How do you know the test score is honest?"**
The 20% test split is made before FLAML sees any data and is used exactly once, after the search. FLAML's own validation score is shown separately and labelled as such.

**"Why is the Titanic F1 only about 0.7?"**
F1 here is measured for the "survived" class, which is the harder class to predict, and the test set has only 179 rows. Accuracy is around 0.79 to 0.80, which is normal for this dataset with no feature engineering. The point of the demo is the process, not the leaderboard score.

**"Removing the columns only improved F1 by 0.036. Is the project still worth it?"**
Yes. Titanic is small and well studied, and most of the signal sits in two columns (sex and class), so one change cannot move the score much. The project is not about squeezing another decimal out of Titanic. It is about reaching a good model with fewer experiments and less compute, and about judging changes correctly. This experiment already shows why that matters: on my first split the gain was 0.036, but over five other splits the average gain was only 0.012, it helped on three of the five, and on one split FLAML's validation score even preferred the worse setup. A system that trusts one run would learn the wrong lessons. Bigger gains should come from strategies the baseline does not have yet (feature engineering such as extracting the title from `Name` or computing family size, and class weighting), and from messier datasets with more room to improve.

**"Why is the validation F1 higher than the test F1?"**
The validation score is the number FLAML was optimising, measured on the training part with cross-validation. After trying hundreds of configurations, the best one is bound to look slightly better than it really is. The test set was never used during the search, so it gives the honest number. With only 179 test rows, it also moves around a bit from run to run.

**"What does the experiment memory store, and how is it used?"**
Dataset meta-features, the diagnosis, the strategy applied, the model, the result and the cost. For a new dataset it finds the most similar past datasets (nearest neighbours on the meta-features) and tries the strategies that helped there first.

**"How will you evaluate the strategy selection?"**
By comparing against random or fixed-order strategy selection (the ablation) on time to reach a target score and on the final score, over many datasets and seeds, with standard statistical tests (Wilcoxon signed-rank and Friedman, following Demšar 2006).

**"What did you build yourself and what did you use?"**
We used FLAML (the search), scikit-learn (metrics and splitting), MLflow (tracking), Streamlit and Plotly (the UI), and pandas. We wrote the profiler, task detection, evaluation, the explanation views, tracking and history, the engine interface, the dashboard and the tests. The planned autonomous layer is our research contribution.

## 16. Command cheat sheet

```bash
conda activate automl
cd C:\Users\priya\Desktop\project\AutoML\autopilot-ml

# tests
python -m pytest -q

# dashboard -> http://localhost:8501
streamlit run app/main.py --server.address localhost --server.headless true --browser.gatherUsageStats false

# MLflow (second window) -> http://127.0.0.1:5000
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000 --workers 1
```

Stop either server with Ctrl+C in its window.
