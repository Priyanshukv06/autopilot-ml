# Introduction

## Background

Most data that organisations use for prediction is tabular: records in rows, with numeric and text columns, such as customers, patients, machines or transactions. On this kind of data, ensembles of decision trees trained by gradient boosting, such as LightGBM and XGBoost, are still among the most accurate models, often ahead of deep neural networks [@grinsztajn2022trees; @ke2017lightgbm; @chen2016xgboost].

Getting a good model is rarely a single step. A data scientist trains a first model, looks at its scores, finds out why it is weak, changes the data, the features or the model family, trains again and compares the two results. Studies of machine learning practice describe this loop of experiments as the main part of the work, and also as a source of hidden cost and mistakes [@amershi2019se4ml; @sculley2015debt].

Automated machine learning (AutoML) takes over one part of this loop: choosing a learning algorithm and its hyperparameters for a given dataset, known as the combined algorithm selection and hyperparameter optimisation (CASH) problem [@thornton2013autoweka; @hutter2019automl]. Open-source systems such as Auto-sklearn, TPOT, AutoGluon and FLAML solve it well and are compared on public benchmarks [@feurer2015autosklearn; @olson2016tpot; @erickson2020autogluon; @wang2021flaml; @gijsbers2024amlb].

## Motivation

An AutoML run returns a model and a validation score, but the questions around the run are still left to the user. Is the model overfitting? Is a column misleading it? Is the validation score trustworthy? Which change should be tried next, is a better score after that change real or only noise, and when is it time to stop? What was learned on one dataset is also rarely used on the next one, although work on meta-learning shows that such experience can guide a new search [@vanschoren2018meta; @cawley2010overfit].

A small example from the first phase of this project shows the gap. On the Titanic passenger data, one FLAML search of 60 seconds (seed 42) gave a model with a test F1 score of 0.673 but a training F1 of 0.934, so the model was overfitting. The `Ticket` column has 681 different values in 891 rows, and 63% of the test passengers had a ticket number the model never saw in training [@wang2021flaml]. A person would drop or re-encode such a column, run the search again and compare the two models fairly [@micci2001target]. The AutoML library cannot make that decision, because it only searches the data it is given.

The project started from working with FLAML, an AutoML library from Microsoft Research released under the MIT licence [@wang2021flaml; @flamlrepo]. FLAML finds good models quickly on a laptop because its search takes the cost of every trial into account [@wu2021cfo]. The questions after every run were still left to the user, and that gap became this project. FLAML is used unchanged as the search engine of this work: the project does not modify it and does not claim to have built it, and comparing against FLAML alone is part of the evaluation [@wang2021flaml].

The work also had practical limits: a laptop CPU, no GPU, no paid services and no cloud. Recent systems use large language models to plan and run machine learning experiments, but they depend on large hosted models and are hard to reproduce and measure [@huang2024mlagentbench; @hollmann2023caafe]. Every decision of this system therefore had to be made by rules and statistics that run offline and can be checked from recorded numbers [@demsar2006].

## Proposed solution in brief

AutoPilot-ML is an autonomous experimentation layer built on top of FLAML. It runs the loop of the data scientist by itself, as a Session of Experiments within a time budget. After each Experiment, a Critic diagnoses weaknesses from evidence measured on 15 fixed folds of the training data, a Strategy_Engine proposes changes that answer the Diagnoses, and a Judge keeps a change only if a corrected resampled t-test on the same folds says it is better, or if it is simpler and not worse [@nadeau2003inference; @bouckaert2004replicability]. A Resource_Manager divides the time budget between Experiments, and an Experiment_Memory records every change with its measured effect and cost, so that a bandit method can learn which changes tend to work on similar datasets [@thompson1933; @auer2002ucb]. Figure @fig:loop shows this loop and the part that FLAML performs inside it [@wang2021flaml].

::figure fig:loop | figures/fig_loop.png | 15.5 | The experimentation loop of AutoPilot-ML; the search inside each Experiment is done by FLAML

## Contributions

The work of this project, built and measured by the team on top of FLAML [@wang2021flaml], is the following.

- An autonomous experimentation layer around FLAML with 14 kinds of Diagnosis and 12 Strategies, where every decision is recorded with its numbers and can be replayed from the saved records [@wang2021flaml].
- A cheap acceptance rule, the Judge, that compares refits of the configurations FLAML found on fixed folds with a corrected resampled t-test, together with a study of its error rates on data with known answers [@nadeau2003inference; @bouckaert2004replicability].
- A learned choice of the next change from the Experiment_Memory with Thompson sampling or UCB1, with switches that turn each part off for ablation studies [@thompson1933; @auer2002ucb].
- A benchmark fixed in writing before it ran: 390 runs on 6 OpenML datasets, 5 seeds and 2 time budgets, against FLAML alone, AutoGluon and a manual LightGBM baseline, analysed with the Friedman, Nemenyi and Wilcoxon tests with Holm's correction [@vanschoren2013openml; @demsar2006]. Its results are reported exactly as measured.
- Optional parts: SMOTE, cross-fitted target encoding, word features, group-aware folds, SHAP explanations, a leakage warning, plain-language summaries by a local language model, a local web API and a Docker image [@chawla2002smote; @micci2001target; @lundberg2017shap].

## Organisation of the report

Chapter 2 reviews hyperparameter search, AutoML systems, the way FLAML searches, learning across datasets and the statistics of comparing models [@hutter2019automl; @demsar2006]. Chapter 3 states the problem, the objectives and the hypotheses, and chapter 4 describes the method: the phases, the components of the loop, the datasets and the benchmark [@vanschoren2013openml]. Chapter 5 gives the main algorithms and chapter 6 the block diagram, the data flow diagrams and the flowcharts. Chapter 7 lists the tools and describes the implementation and its tests, chapter 8 reports every measurement with its settings, and chapter 9 draws the conclusions and lists the future work [@wang2021flaml].
