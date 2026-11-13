# Proposed methodology and work description

## Overall approach

The work followed one rule: build and measure a reliable baseline first, then add one autonomous part at a time, each with its own tests and a way to measure it before any claim depends on it. Mature open-source libraries were used instead of being rebuilt: FLAML for the search [@wang2021flaml], scikit-learn for splits, metrics and importance [@pedregosa2011sklearn], MLflow for tracking [@zaharia2018mlflow] and Streamlit for the dashboard [@streamlit]. Other fixed rules were: no language model in any decision, no paid service and no cloud, business logic kept apart from the user interface, and no result reported unless it was measured, with anything not built marked as planned.

From the second phase on, the work was planned as a written specification in the Kiro development environment [@kiro]: a requirements document (32 requirements with 160 numbered acceptance criteria written as WHEN, IF, THEN and SHALL statements), a design document (components, code shapes, data formats and design decisions) and a task list (35 tasks and 21 test and run sub-tasks, each linked to the criteria it satisfies). Each task was built, tested and committed on its own, with one logical change per commit. Table @tab:phases shows the phases of the work.

::table tab:phases | Phases of the work | widths=1.6,10,4.2 | align=cjc
| Phase | Content | When |
|---|---|---|
| 1 | Baseline: dataset profiling, task detection, FLAML search, held-out evaluation, MLflow tracking, dashboard and experiment history | Built by 7 October 2026; reviewed in October 2026 |
| 2 | Autonomous layer: Assessor, Critic, Strategy_Engine, Judge, Resource_Manager, Planner and Session loop, Experiment_Memory, autonomous mode of the dashboard; the Critic and Judge studies; a first comparison with FLAML alone | 26 and 27 October 2026 |
| 3 | Learned strategy choice, ablation switches, comparison systems, benchmark harness and statistics, frozen memory, Judge agreement study | 27 to 31 October 2026 |
| Optional | SMOTE, target encoding, word features, group-aware splits, SHAP, target suggestion and leakage warning, local summaries, web API and Docker image | 1 to 12 November 2026 |
| Final run | The whole benchmark of 390 runs with the final code | 12 and 13 November 2026 |
::end

## One Experiment: the baseline pipeline

An Experiment is one FLAML search with fixed settings. It checks its inputs, detects the task from the target column by simple rules (two values give binary classification, few integer values or text give multiclass, other numbers give regression), and profiles the columns: types, empty cells, duplicates, identifier-like and constant columns, the number of different values and the class balance. It then makes the outer split, stratified 80% training and 20% test rows, before FLAML sees any data, because the score FLAML optimises cannot also be an independent final score [@cawley2010overfit].

FLAML searches on the training part only. The best model is scored once on the test part with every metric of its task, next to a trivial baseline (the most common class, or the mean for regression) on the same rows, and permutation importance is measured on the original columns [@fisher2019importance]. Every Experiment writes a small JSON record and an MLflow run with its settings, metrics, model and files [@zaharia2018mlflow].

## The autonomous Session

A Session runs Experiments one after another within a total time budget. Each Experiment searches one Setup: the feature columns, the Data_Steps (column changes learned only from training rows, such as an encoding), class weights or a resampler, the learner families FLAML may use, limits on its search space, the search metric and the time budget. The Setup currently believed to be best is the Accepted_Setup.

A Session first makes the outer split into a Training_Part and a Test_Part, as in Phase 1, and then splits the Training_Part into 3 repeats of 5-fold cross-validation, the 15 Session_Folds, with a seed of its own. Every decision of the Session is made on these folds of the Training_Part; the Test_Part is used once, at the end, to report the first and the final model [@cawley2010overfit]. Figure @fig:split shows the split.

::figure fig:split | figures/fig_split.png | 15.5 | The outer split and the 15 Session_Folds; decisions use only the Training_Part

The first Setup is the user's settings with identifier-like and constant columns left out. It is searched with the first budget, assessed and diagnosed. Then the loop repeats: the Strategy_Engine proposes changes to the Accepted_Setup, the Resource_Manager decides whether to stop and gives the next budget, the first Proposal is applied to a copy of the Accepted_Setup, FLAML searches it starting from the Accepted_Setup's best configurations, and the Judge compares it with the Accepted_Setup on the same folds. A kept change becomes the new Accepted_Setup and is diagnosed again. Chapter 5 gives the loop as Algorithm @alg:session.

## The Assessor

FLAML's own validation score is the best of many trials on its own folds, so it is optimistic and differs between two searches. The Assessor therefore refits the configuration FLAML chose, with no search, on the training rows of each of the 15 Session_Folds and scores it on that fold's validation rows. It records every metric of the task for training and validation rows, the recall of each class, the share of validation rows with a category value not seen in the fold's training rows, and the trivial baseline's scores. Data_Steps and class weights are fitted again inside each fold, on that fold's training rows only, so no validation row ever influences its own score. When a candidate would refit exactly the same model as an earlier Setup, its fold scores are reused.

## The Critic

The Critic turns the profile of the Training_Part, the Assessment and a summary of FLAML's search into Diagnoses. Each Diagnosis has a kind, a severity (low, moderate or high), the columns concerned, its evidence numbers and a plain message. The rules run in a fixed order, and their thresholds live in a versioned settings file. They use a diagnostic metric m: the primary metric if it is bounded and higher is better, otherwise F1 or R². Two examples of evidence are the overfitting gap of Equation (@eq:gap) over the J = 15 folds, and the skill over the trivial baseline of Equation (@eq:skill), used for underfitting [@hastie2009esl].

::equation eq:gap

::equation eq:skill

Table @tab:diagnoses lists all 14 kinds with the thresholds of the version `v1-critic-study`. That version came from the Critic study of chapter 8, which changed three starting values: the overfitting gap from 0.10 to 0.11, the empty share from 0.40 to 0.30 and the near-constant share from 0.99 to 0.98. The last two kinds were added with the optional tasks; possible_leakage is a warning only, because only the user knows whether a column is available before the outcome [@kaufman2012leakage].

::table tab:diagnoses | The 14 kinds of Diagnosis and their thresholds | widths=4.2,5.8,3.0,2.8 | align=jjcc
| Kind | Evidence | Moderate | High |
|---|---|---|---|
| overfitting | Gap g of Equation (@eq:gap) | 0.11 or more | 0.20 or more |
| underfitting | Skill s of Equation (@eq:skill), only when the gap is below 0.03 | below 0.50 | below 0.25 |
| budget_limited | Gain of the search's best score in the last quarter of its time | 0.005 or more | 0.02 or more |
| wasted_budget | Share of the search time after its best configuration | 0.50 or more | 0.75 or more |
| class_imbalance | Largest over smallest class; validation recall of the smallest class | ratio 3 or more, or recall below 0.5 | ratio 9 or more, or recall below 0.3 |
| missing_values | Share of empty cells in a used column | 0.30 or more | 0.70 or more |
| high_cardinality | Different values of a used text column; share of validation rows with an unseen value | over 50 values, or 0.30 unseen | 0.60 unseen |
| identifier_column | The profiler's identifier flag | | always |
| constant_column | One value, or the same value in 98% or more of rows | low | |
| redundant_features | Absolute correlation of two used numeric columns | 0.95 or more | 0.99 or more |
| unreliable_validation | Standard deviation of validation m over the folds | 0.05 or more | 0.08 or more |
| weak_model_families | A learner family 0.10 or more behind the best one after 5% or more of the search time | low | |
| unused_words | A text column with frequent words that no encoding uses | low if used whole | moderate if left out |
| possible_leakage | Cross-validated score of one column alone | | 0.99 or more |
::end

## The Strategy_Engine

A Strategy is a named change that returns a changed copy of a Setup and applies only when it makes sense, for example class weights only for classification. The Strategy_Engine has a rule table that gives, for each kind of Diagnosis, the Strategies to try in order. It makes one Proposal per change and column, merges a change asked for by several Diagnoses into one Proposal, leaves out changes already tried in the Session or that do not apply, and ranks the rest by severity (high first), cost (cheap first), table position and finally a fixed key, so that the order is always the same. Table @tab:strategies lists the 12 Strategies.

::table tab:strategies | The 12 Strategies, what they change and what they answer | widths=4.7,5.5,3.5,2.2 | align=jjcc
| Strategy | What it changes | Answers | Complexity |
|---|---|---|---|
| drop_columns | Removes a column and its Data_Steps | identifier, constant, high cardinality, missing values (high only), redundant pair, overfitting | simpler |
| class_weight_balanced | Balanced sample weights in the search and the refits | class imbalance | same |
| regularised_search_space | Smaller trees: at most 32 leaves for LightGBM and XGBoost, depth at most 4 for depth-limited XGBoost, at most 64 leaves for the forests | overfitting | simpler |
| frequency_encode | A text value becomes the share of training rows that have it | high cardinality | same |
| group_rare_categories | Values seen in under 1% of training rows become one value | high cardinality | simpler |
| missing_indicator | Adds a 0/1 column that marks the empty cells of a numeric column | missing values | more complex |
| restrict_model_families | Searches only the learner families that were not weak | weak model families | simpler |
| more_search_time | The same Setup with at least twice the search time | underfitting, budget limited | same |
| smoother_search_metric | Searches with log loss instead of the primary metric | class imbalance, unreliable validation | same |
| smote | Synthetic rows of the smaller classes, in each fold's training rows only | class imbalance | more complex |
| target_encode | Cross-fitted, smoothed mean target of each value (Equation @eq:te) | high cardinality | same |
| token_features | 0/1 columns for up to 20 frequent words of a text column | unused words | more complex |
::end

SMOTE is applied through FLAML's resampling hook, so FLAML resamples only the training rows of each of its folds and never the validation rows [@chawla2002smote; @lemaitre2017imblearn]. Target encoding replaces a value c by the smoothed mean target of Equation (@eq:te), with m = 10, where the sum runs over the training rows that have the value c and n_{c} is their number [@micci2001target]. It is cross-fitted: the training rows are split into 5 parts, and each part is encoded with the means of the other 4, so that no row's encoding uses its own label.

::equation eq:te

## The Judge

The Judge decides whether a candidate Setup replaces the Accepted_Setup. On each fold j it takes the difference d of the candidate's and the Accepted_Setup's validation scores, with the sign flipped for error metrics so that a positive value always favours the candidate. Because the 15 folds share most of their training rows, their differences are not independent, so the Judge uses the corrected resampled t statistic of Nadeau and Bengio in the repeated form of Bouckaert and Frank, Equation (@eq:ttest), where the ratio of validation to training rows is 1/4 for 5 folds and the variance is that of Equation (@eq:var) [@nadeau2003inference; @bouckaert2004replicability].

::equation eq:ttest

::equation eq:var

From t the Judge computes p(better) and p(worse) with Student's t distribution with J - 1 degrees of freedom. It returns one of three Verdicts. **Keep** when the mean difference is above the minimum gain (0) and p(better) is at most alpha (0.20). **Keep_simpler** when the candidate is simpler, its mean difference is at least -0.005 and p(worse) is above alpha. **Discard** in every other case. When all differences are equal, that is, their standard deviation is at most the refit tolerance of 10^{−12}, the Judge decides on the mean alone, and a mean within that tolerance counts as no gain. As an example with 5 folds, the differences 0.02, 0.01, 0.03, -0.01 and 0.02 have a mean of 0.014 and give a corrected t of 1.376 with p(better) 0.120, while an uncorrected paired t-test would give t = 2.064 and p = 0.054 [@dietterich1998tests].

## The Resource_Manager

The Resource_Manager holds the total time T, the largest number of Experiments, the CPU cores and three settings: a first share of 35% of T, a minimum of 5 seconds per search and a patience of 3. The first search gets 35% of T. Before each later step it keeps back a reserve ρ, 1.2 times the Assessment and other time of the previous step, and gives the next search the budget b of Equation (@eq:budget), where t_{e} is the time used so far and s the number of Experiments still allowed.

::equation eq:budget

A more_search_time step gets at least twice the Accepted_Setup's budget, as far as the time allows, and after a wasted_budget Diagnosis the next search is shortened to 1.5 times the time the Accepted_Setup's search needed to find its best. The Session stops, in this order of checks, on a user request, an error, no time left, no Experiments left, no Proposal left, or three discards in a row. Every step records its budget, search time, Assessment time, overrun and utility, the validation gain per second.

## The Experiment_Memory and the learned choice

After every Verdict the Session appends one record to the Experiment_Memory, a file of JSON lines that is only ever appended to: the dataset's fingerprint and meta-features (rows, columns, shares of numeric and categorical columns, empty cells, class balance and others), the task and metric, the Diagnoses answered, the Strategy and its parameters, the Verdict, the Judge's mean difference and its spread, and the cost in seconds. Similar datasets are found by standardising the meta-features over the stored datasets and taking the Euclidean distance, among datasets of the same task only [@vanschoren2018meta].

An arm is a pair of Strategy and diagnosis kind. For the 5 nearest earlier datasets, never the current one, each record is weighted by 1/(1 + distance), which gives an arm its weighted keeps K and discards D, the mean Judge difference of its kept records G, its mean cost in seconds C and its number of records n. Four selectors order the Proposals of the rule table: `rules` keeps the written order, `random` shuffles it, and `thompson` and `ucb1` score each arm by Equations (@eq:thompson) and (@eq:ucb), where N is the number of records of all arms of the same diagnosis kind [@thompson1933; @auer2002ucb]. The Thompson score is a sampled expected validation gain per second.

::equation eq:thompson

::equation eq:ucb

An arm with fewer than 3 records, or with records from only one dataset, keeps its place in the rule order, and a kept record that changed no fold score counts as not kept. Each step records how its Proposal was placed. Four switches turn parts of the loop off for ablation studies: the memory, the strategy choice (a random order), the Critic (random valid changes instead of Diagnoses) and the Resource_Manager (equal shares of the time, no reserve and no patience rule).

## Optional extensions

The following parts were added after the core loop and are switched on by settings.

- Group-aware splits: when the user names a grouping column, whole groups go to the Test_Part and each group stays inside one Session_Fold, so scores reflect new groups [@roberts2017groupcv].
- SHAP explanations of a Session's final model on Training_Part rows: exact tree SHAP for tree learners and a permutation estimate for the others [@lundberg2017shap; @lundberg2020treeshap].
- Target suggestion and a leakage check: candidate target columns are ranked by simple points, and each feature column is scored alone by 5-fold cross-validation, with a warning at 0.99 or more [@kaufman2012leakage].
- A plain-language summary of a Session written by a local language model through Ollama, labelled as generated and never read by any decision [@ollama; @qwen2024].
- A local web API with FastAPI and a Docker image that runs the API and the dashboard [@fastapi; @merkel2014docker].

## Evaluation method

### Component studies

The Critic and the Judge were first measured on synthetic data with known answers. Each synthetic dataset of 1,000 rows comes with a fresh sample of 20,000 rows from the same generator, so a model's score on the fresh sample estimates its true score. The Critic study used 270 binary datasets, each with one planted problem (class imbalance, label noise, useless columns, an identifier column, a constant column, a high-cardinality column, empty cells or duplicated columns) at three strengths, or no problem, over 5 tuning seeds and 5 reporting seeds. The Judge study used 84 pairs of Setups, where the second Setup changes one thing, and counted a false keep when the Judge accepted a change whose true effect was below -0.005, and power as the share of truly helpful changes (above +0.01) it kept, over 9 settings of alpha and fold repeats. The rule that picks the settings was written before the runs.

### Datasets

The benchmark datasets were chosen by a written rule before any of them was downloaded. From the 107 datasets of two curated OpenML suites, OpenML-CC18 for classification and OpenML-CTR23 for regression [@vanschoren2013openml; @bischl2021suites], those with 500 to 10,000 rows and at most 100 feature columns were kept and sorted into six groups. In each group the candidates were shuffled with a generator seeded with 2026: the first became an evaluation dataset and the second a development dataset. Development datasets, with the four sample datasets, are used to build the memory and may be looked at; evaluation datasets are used only in the benchmark. Table @tab:datasets lists them.

::table tab:datasets | Evaluation and development datasets (OpenML ID in brackets) | widths=2.6,6.6,6.6 | align=cjj
| Group | Evaluation dataset | Development dataset |
|---|---|---|
| Binary, imbalanced | pc4 (1049): 1,458 rows, 37 numeric columns, class ratio 7.2 | wilt (40983): 4,839 rows, 5 columns, ratio 17.5 |
| Binary, messy | cylinder-bands (6332): 540 rows, 37 columns of which 14 text, 5% empty cells | tic-tac-toe (50): 958 rows, 9 text columns |
| Multiclass, clean | segment (40984): 2,310 rows, 18 columns, 7 classes | vehicle (54): 846 rows, 18 columns, 4 classes |
| Multiclass, messy | MiceProtein (40966): 1,080 rows, 77 columns, 8 classes, 1.7% empty cells | balance-scale (11): 625 rows, 4 columns, 3 classes |
| Regression, clean | energy_efficiency (44960): 768 rows, 8 columns | kin8nm (44980): 8,192 rows, 8 columns |
| Regression, messy | solar_flare (44966): 1,066 rows, 10 columns of which 3 text | student_performance_por (44967): 649 rows, 30 columns of which 17 text |
| Samples | none | titanic, breast_cancer, penguins, california_housing |
::end

### Comparison systems

Every system gets the same training and test rows of a dataset and seed, the same columns, the same time budget and all CPU cores, and its test predictions are scored by the same evaluator. Table @tab:systems lists the eight systems. AutoGluon could not be installed next to the project's libraries, because it needs an older pandas, so it runs in its own environment as a separate process [@erickson2020autogluon].

::table tab:systems | The systems compared in the benchmark | widths=4.0,11.8 | align=jj
| System | What it does |
|---|---|
| manual | LightGBM with its library defaults (100 trees, 31 leaves, learning rate 0.1), trained once; no search [@ke2017lightgbm] |
| flaml | One FLAML search with the whole budget [@wang2021flaml] |
| autogluon | AutoGluon-Tabular 1.6.3 with the medium_quality preset, CPU only [@erickson2020autogluon] |
| autopilot_rules | A Session with the rule order of the Strategy_Engine |
| autopilot_full | A Session with Thompson sampling on the frozen memory |
| random_choice | Ablation: the same changes in a random order |
| no_critic | Ablation: random valid changes instead of Diagnoses |
| no_resource_manager | Ablation: equal shares of the time, no reserve and no patience rule |
::end

### Benchmark grid and statistics

The grid has 390 cells, one per system, dataset, budget and seed: the five main systems at 120 and 300 seconds and the three ablations at 300 seconds, on the 6 evaluation datasets with seeds 1 to 5. A third budget of 600 seconds was dropped because it would have added about 20 hours on one laptop. The memory the full system reads was built beforehand by 30 Sessions on the development datasets, frozen and checked by its SHA-256 hash in every row. The harness appends one row per finished cell, never edits a row, can be stopped and restarted, and waits while the laptop runs on battery.

The analysis was fixed before any cell ran [@demsar2006]. A system's score on a dataset is its mean test score over the 5 seeds, with error metrics negated. Within each dataset the systems are ranked, and the Friedman statistic of Equation (@eq:friedman), with R_{j} the average rank of system j over N datasets and k systems, tests whether any system stands apart [@friedman1937; @iman1980]. The Nemenyi critical difference of Equation (@eq:cd) gives the smallest difference of average ranks that counts as real [@nemenyi1963].

::equation eq:friedman

::equation eq:cd

Pairs of systems are compared with the Wilcoxon signed-rank test with Holm's correction, the rank-biserial correlation as effect size and a 95% bootstrap interval of the median difference from 10,000 resamples [@wilcoxon1945; @holm1979; @kerby2014; @efron1993bootstrap]. For time-to-target, the target of a dataset and seed is 99% of the best score any main system reached at either budget, and a system's time-to-target is the smallest budget at which it reached it.

### Judge agreement study

The last study asks whether the cheap Judge decides like a strict comparison that searches again. From the steps the Judge really judged while building the memory, 20 pairs were drawn, at most 2 per development dataset. For each pair, both Setups were searched again by FLAML on 5 random 80/20 splits of the Training_Part, at most 25 trials each, and the 5 differences were judged with the Judge's own rules. Agreement is reported with Cohen's kappa of Equation (@eq:kappa), where p_{o} is the observed share of equal verdicts and p_{e} the share expected by chance [@cohen1960kappa].

::equation eq:kappa
