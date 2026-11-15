# Proposed methodology and work description

## Overall approach

The work followed one rule: build and measure a reliable baseline first, then add one autonomous part at a time, each with its own tests and a way to measure it. Mature open-source libraries were used instead of being rebuilt: FLAML for the search [@wang2021flaml], scikit-learn for splits, metrics and importance [@pedregosa2011sklearn], MLflow for tracking and Streamlit for the dashboard. Every decision of the system is made by rules and statistics that run offline, with no language model, no paid service and no cloud, and the business logic is kept apart from the user interface.

The autonomous layer was planned as a written specification: 32 requirements with 160 numbered acceptance criteria, a design of the components and their data formats, and 35 tasks, each linked to the criteria it satisfies and built, tested and committed on its own. We worked in Kiro, an AI-assisted development environment, and used it to write the specification, the code and the documents [@kiro]. The work went from the baseline pipeline to the autonomous layer and its studies, then to the learned choice of changes and the benchmark, and finally to the optional extensions.

## One experiment: the baseline pipeline

An experiment is one FLAML search with fixed settings. It checks its inputs, detects the task from the target column by simple rules (two values give binary classification, few integer values or text give multiclass, other numbers give regression), and profiles the columns: types, empty cells, duplicates, identifier-like and constant columns, the number of different values and the class balance. It then makes the outer split, stratified 80% training and 20% test rows, before FLAML sees any data, because the score FLAML optimises cannot also be an independent final score [@cawley2010overfit].

FLAML searches on the training part only. The best model is scored once on the test part with every metric of its task, next to a trivial baseline (the most common class, or the mean for regression) on the same rows, and permutation importance is measured on the original columns [@hastie2009esl]. Every experiment writes a small JSON record and an MLflow run with its settings, metrics, model and files.

## The autonomous session

A session runs experiments one after another within a total time budget. Each experiment searches one setup: the feature columns, the data steps (column changes learned only from training rows, such as an encoding), class weights or a resampler, the learner families FLAML may use, limits on its search space, the search metric and the time budget. The setup currently believed to be best is the accepted setup.

A session first makes the outer split into a training part and a test part, and then splits the training part into 3 repeats of 5-fold cross-validation, the 15 session folds. Every decision of the session is made on these folds of the training part, and the test part is used only once, at the end, to report the first and the final model [@cawley2010overfit].

The first setup is the user's settings with identifier-like and constant columns left out. It is searched with the first budget, assessed and diagnosed. Then the loop repeats: the Strategy Engine proposes changes to the accepted setup, the Resource Manager decides whether to stop and gives the next budget, the first proposal is applied to a copy of the accepted setup, FLAML searches it starting from the accepted setup's best configurations, and the Judge compares it with the accepted setup on the same folds. A kept change becomes the new accepted setup and is diagnosed again. Chapter 5 gives the loop as Algorithm @alg:session.

## The Assessor

FLAML's own validation score is the best of many trials on its own folds, so it is optimistic and differs between two searches. The Assessor therefore refits the configuration FLAML chose, with no search, on the training rows of each of the 15 session folds and scores it on that fold's validation rows. It records every metric of the task for training and validation rows, the recall of each class, the share of validation rows with a category value not seen in the fold's training rows, and the trivial baseline's scores. Data steps and class weights are fitted again inside each fold, on that fold's training rows only, so no validation row ever influences its own score. When a candidate would refit exactly the same model as an earlier setup, its fold scores are reused.

## The Critic

The Critic turns the profile of the training part, the assessment and a summary of FLAML's search into diagnoses. Each diagnosis has a kind, a severity (low, moderate or high), the columns concerned, its evidence numbers and a plain message. The rules run in a fixed order, and their thresholds are kept in a versioned settings file. They use a diagnostic metric m: the primary metric if it is bounded and higher is better, otherwise F1 or R². For example, the evidence for overfitting is the gap g of Equation (@eq:gap), the mean difference between the training and validation scores over the J = 15 folds [@hastie2009esl].

::equation eq:gap

Table @tab:diagnoses lists all 14 kinds of diagnosis. For each kind it gives the evidence the Critic measures and the values at which the Critic reports a moderate or a high severity; these thresholds were tuned on synthetic data with known answers (section 4.11). The possible_leakage kind is a warning only, because only the user knows whether a column is available before the outcome [@kaufman2012leakage].

::table tab:diagnoses | The 14 kinds of diagnosis and their thresholds | widths=4.2,5.8,3.0,2.8 | align=jjcc
| Kind | Evidence | Moderate | High |
|---|---|---|---|
| overfitting | Gap g of Equation (@eq:gap) | 0.11 or more | 0.20 or more |
| underfitting | Skill over the trivial baseline, the share of the possible gain that the model reaches, only when the gap is below 0.03 | Below 0.50 | Below 0.25 |
| budget_limited | Gain of the search's best score in the last quarter of its time | 0.005 or more | 0.02 or more |
| wasted_budget | Share of the search time after its best configuration | 0.50 or more | 0.75 or more |
| class_imbalance | Largest over smallest class; validation recall of the smallest class | Ratio 3 or more, or recall below 0.5 | Ratio 9 or more, or recall below 0.3 |
| missing_values | Share of empty cells in a used column | 0.30 or more | 0.70 or more |
| high_cardinality | Different values of a used text column; share of validation rows with an unseen value | Over 50 values, or 0.30 unseen | 0.60 unseen |
| identifier_column | The profiler's identifier flag | | Always |
| constant_column | One value, or the same value in 98% or more of rows | Low | |
| redundant_features | Absolute correlation of two used numeric columns | 0.95 or more | 0.99 or more |
| unreliable_validation | Standard deviation of validation m over the folds | 0.05 or more | 0.08 or more |
| weak_model_families | A learner family 0.10 or more behind the best one after 5% or more of the search time | Low | |
| unused_words | A text column with frequent words that no encoding uses | Low if used whole | Moderate if left out |
| possible_leakage | Cross-validated score of one column alone | | 0.99 or more |
::end

## The Strategy Engine

A strategy is a named change that returns a changed copy of a setup and applies only when it makes sense, for example class weights only for classification. The Strategy Engine has a rule table that gives, for each kind of diagnosis, the strategies to try in order. It makes one proposal per change and column, merges a change asked for by several diagnoses into one proposal, leaves out changes already tried in the session or that do not apply, and ranks the rest by severity (high first), cost (cheap first), table position and finally a fixed key, so that the order is always the same. Table @tab:strategies lists the 12 strategies.

In Table @tab:strategies, each row gives what a strategy changes, the diagnoses it answers and whether it makes the setup simpler, the same or more complex; the Judge's keep_simpler verdict applies only to the simpler ones. SMOTE is applied through FLAML's resampling hook, so FLAML resamples only the training rows of each of its folds and never the validation rows [@chawla2002smote]. Target encoding replaces a value by the mean target of the training rows that have it, pulled towards the overall mean with a weight of 10 rows so that rare values are not trusted too much [@micci2001target]. It is cross-fitted: the training rows are split into 5 parts, and each part is encoded with the means of the other 4, so that no row's encoding uses its own label.

::table tab:strategies | The 12 strategies, what they change and what they answer | widths=4.7,5.5,3.5,2.2 | align=jjcc
| Strategy | What it changes | Answers | Complexity |
|---|---|---|---|
| drop_columns | Removes a column and its data steps | Identifier, constant, high cardinality, missing values (high only), redundant pair, overfitting | Simpler |
| class_weight_balanced | Balanced sample weights in the search and the refits | Class imbalance | Same |
| regularised_search_space | Smaller trees: at most 32 leaves for LightGBM and XGBoost, depth at most 4 for depth-limited XGBoost, at most 64 leaves for the forests | Overfitting | Simpler |
| frequency_encode | A text value becomes the share of training rows that have it | High cardinality | Same |
| group_rare_categories | Values seen in under 1% of training rows become one value | High cardinality | Simpler |
| missing_indicator | Adds a 0/1 column that marks the empty cells of a numeric column | Missing values | More complex |
| restrict_model_families | Searches only the learner families that were not weak | Weak model families | Simpler |
| more_search_time | The same setup with at least twice the search time | Underfitting, budget limited | Same |
| smoother_search_metric | Searches with log loss instead of the primary metric | Class imbalance, unreliable validation | Same |
| smote | Synthetic rows of the smaller classes, in each fold's training rows only | Class imbalance | More complex |
| target_encode | Cross-fitted, smoothed mean target of each value | High cardinality | Same |
| token_features | 0/1 columns for up to 20 frequent words of a text column | Unused words | More complex |
::end

## The Judge

The Judge decides whether a candidate setup replaces the accepted setup. On each fold j it takes the difference d of the candidate's and the accepted setup's validation scores, with the sign flipped for error metrics so that a positive value always favours the candidate. Because the 15 folds share most of their training rows, their differences are not independent, so the Judge uses the corrected resampled t statistic of Nadeau and Bengio in the repeated form of Bouckaert and Frank, Equation (@eq:ttest), where the ratio of validation to training rows is 1/4 for 5 folds and the variance is that of Equation (@eq:var) [@nadeau2003inference; @bouckaert2004replicability].

::equation eq:ttest

::equation eq:var

From t the Judge computes p(better) and p(worse) with Student's t distribution with J − 1 degrees of freedom. It returns one of three verdicts: **keep** when the mean difference is above the minimum gain (0) and p(better) is at most alpha (0.20), **keep_simpler** when the candidate is simpler, its mean difference is at least -0.005 and p(worse) is above alpha, and **discard** in every other case. When all differences are equal, that is, their standard deviation is at most the refit tolerance of 10^{−12}, the Judge decides on the mean alone. As an example with 5 folds, the differences 0.02, 0.01, 0.03, -0.01 and 0.02 have a mean of 0.014 and give a corrected t of 1.376 with p(better) 0.120, while an uncorrected paired t-test, which ignores the overlap, would give t = 2.064 and p = 0.054 [@dietterich1998tests].

## The Resource Manager

The Resource Manager holds the total time T, the largest number of experiments, the CPU cores and three settings: a first share of 35% of T, a minimum of 5 seconds per search and a patience of 3. The first search gets 35% of T. Before each later step it keeps back a reserve ρ, 1.2 times the assessment and other time of the previous step, and gives the next search the budget b of Equation (@eq:budget), where t_{e} is the time used so far and s the number of experiments still allowed.

::equation eq:budget

A more_search_time step gets at least twice the accepted setup's budget, as far as the time allows, and after a wasted_budget diagnosis the next search is shortened to 1.5 times the time the accepted setup's search needed to find its best. The session stops, in this order of checks, on a user request, an error, no time left, no experiments left, no proposal left, or three discards in a row. Every step records its budget, search time, assessment time, overrun and utility, the validation gain per second.

## The Experiment Memory and the learned choice

After every verdict the session appends one record to the Experiment Memory, a file of JSON lines that is only ever appended to: the dataset's fingerprint and meta-features (rows, columns, shares of numeric and categorical columns, empty cells, class balance and others), the task and metric, the diagnoses answered, the strategy and its parameters, the verdict, the Judge's mean difference and its spread, and the cost in seconds. Similar datasets are found by standardising the meta-features over the stored datasets and taking the Euclidean distance, among datasets of the same task only [@vanschoren2018meta].

An arm is a pair of strategy and diagnosis kind. For the 5 nearest earlier datasets, never the current one, each record is weighted by 1/(1 + distance), which gives an arm its weighted keeps K and discards D, the mean Judge difference of its kept records G, its mean cost in seconds C and its number of records n. Four selectors order the proposals of the rule table: `rules` keeps the written order, `random` shuffles it, and `thompson` and `ucb1` score each arm by Equations (@eq:thompson) and (@eq:ucb), where N is the number of records of all arms of the same diagnosis kind [@thompson1933; @auer2002ucb]. The Thompson score is a sampled expected validation gain per second.

::equation eq:thompson

::equation eq:ucb

An arm with fewer than 3 records, or with records from only one dataset, keeps its place in the rule order, so the learned choice moves only changes with evidence from at least two other datasets.

## Optional extensions

The following parts were added after the core loop and are switched on by settings.

- Group-aware splits: when the user names a grouping column, whole groups go to the test part and each group stays inside one session fold, so scores reflect new groups.
- SHAP explanations of a session's final model on training part rows: exact tree SHAP for tree learners and a permutation estimate for the others [@lundberg2017shap].
- Target suggestion and a leakage check: candidate target columns are ranked by simple points, and each feature column is scored alone by 5-fold cross-validation, with a warning at 0.99 or more [@kaufman2012leakage].
- A plain-language summary of a session written by a local language model (Qwen2.5 7B through Ollama), labelled as generated and never read by any decision.
- A local web API with FastAPI and a Docker image that runs the API and the dashboard.

## Studies of the Critic and the Judge

The Critic and the Judge were first checked on synthetic data with known answers. Each synthetic dataset of 1,000 rows comes with a fresh sample of 20,000 rows from the same generator, so a model's score on the fresh sample estimates its true score. The Critic study used 270 binary datasets, each with one planted problem (class imbalance, label noise, useless columns, an identifier column, a constant column, a high-cardinality column, empty cells or duplicated columns) at three strengths, or no problem, with one set of seeds for tuning the thresholds and another for reporting. The Judge study used 84 pairs of setups, where the second setup changes one thing with a known true effect, and counted how many harmful changes (a true effect below -0.005) the Judge kept and how many helpful changes (above +0.01) it found.

## Datasets and comparison systems

The benchmark datasets were chosen by a written rule before any of them was downloaded. From the 107 datasets of the curated OpenML suites OpenML-CC18 and OpenML-CTR23 [@vanschoren2013openml; @bischl2021suites], those with 500 to 10,000 rows and at most 100 feature columns were sorted into six groups, and one evaluation and one development dataset were drawn at random from each group. The evaluation datasets are pc4 (1,458 rows, binary, class ratio 7.2), cylinder-bands (540 rows, binary, 14 text columns), segment (2,310 rows, 7 classes), MiceProtein (1,080 rows, 8 classes), energy_efficiency (768 rows, regression) and solar_flare (1,066 rows, regression). The six development datasets and the four sample datasets were used only to build the memory.

Every system gets the same training and test rows of a dataset and seed, the same columns, the same time budget and all CPU cores, and its test predictions are scored by the same evaluator. AutoPilot-ML is compared with three systems: FLAML alone with the whole budget [@wang2021flaml], AutoGluon-Tabular 1.6.3 with its medium_quality preset [@erickson2020autogluon], and LightGBM with its library defaults trained once, with no search [@ke2017lightgbm]. AutoPilot-ML runs in two forms: with the rule order of the Strategy Engine (AutoPilot-ML rules) and with Thompson sampling on a memory built beforehand (AutoPilot-ML full).

## Benchmark runs and analysis

The benchmark has 390 runs on the 6 evaluation datasets with seeds 1 to 5: the five systems at budgets of 120 and 300 seconds, and three variants of AutoPilot-ML with one part switched off at 300 seconds. The memory of the full system was built by 30 sessions on the development datasets and then frozen, and its SHA-256 hash was checked in every run. The harness writes one row per finished run and can be stopped and restarted.

The analysis follows Demšar's method for comparing systems over several datasets [@demsar2006]. A system's score on a dataset is its mean test score over the 5 seeds. Within each dataset the systems are ranked, and each system's average rank over the six datasets summarises its standing. Two systems are compared by the median of their differences over the datasets, with a 95% bootstrap interval from 10,000 resamples, and by the number of datasets on which each one was better [@efron1993bootstrap].
