# Introduction

## Background

Most of the data that organisations use for prediction is tabular. Each row is a record, such as a customer, a patient, a machine or a transaction, and each column holds a number or a short piece of text. On this kind of data, ensembles of decision trees trained by gradient boosting, such as LightGBM and XGBoost, are still among the most accurate models, and they often beat deep neural networks [@grinsztajn2022trees; @ke2017lightgbm; @chen2016xgboost].

A good model rarely comes out of a single training run. A data scientist trains a first model and looks at its scores. If the model is weak, they try to find out why, change the data, the features or the model family, train again and compare the two results. Studies of how machine learning is done in practice describe this loop of experiments as the main part of the job [@amershi2019se4ml].

Automated machine learning (AutoML) takes over one piece of this loop: choosing a learning algorithm and its hyperparameters for a given dataset. This is known as the combined algorithm selection and hyperparameter optimisation (CASH) problem [@thornton2013autoweka; @hutter2019automl]. Open-source systems such as Auto-sklearn, TPOT, AutoGluon and FLAML handle it well, and they are regularly compared on public benchmarks [@feurer2015autosklearn; @olson2016tpot; @erickson2020autogluon; @wang2021flaml; @gijsbers2024amlb].

## Motivation

An AutoML run hands back a model and a validation score, but the questions around the run are still left to the user. Is the model overfitting? Is one of the columns misleading it? Can the validation score be trusted? Which change should come next, is a better score after that change real or only noise, and when is it time to stop? What was learned on one dataset is also rarely carried over to the next one, even though work on meta-learning shows that such experience can guide a new search [@vanschoren2018meta; @cawley2010overfit].

A small example shows the gap. On the Titanic passenger data, one FLAML search of 60 seconds (seed 42) gave a model with a test F1 score of 0.673 but a training F1 of 0.934, so the model was clearly overfitting. The `Ticket` column has 681 different values in 891 rows, and 63% of the test passengers had a ticket number that the model never saw in training [@wang2021flaml]. A person would drop or re-encode such a column, run the search again and compare the two models fairly [@micci2001target]. The AutoML library cannot make that call, because it only searches the data it is given.

Our project grew out of working with FLAML, an AutoML library from Microsoft Research released under the MIT licence [@wang2021flaml; @flamlrepo]. FLAML finds good models quickly on a laptop because its search takes the cost of every trial into account [@wu2021cfo]. Even so, every run left us with the same questions, and that gap became this project. We use FLAML unchanged as the search engine of our system: we did not modify it, we do not claim to have built it, and comparing our system against FLAML alone is part of the evaluation [@wang2021flaml].

We also had practical limits to work within: a laptop CPU, no GPU, no paid services and no cloud. Some recent systems use large language models to plan and run machine learning experiments, but they depend on large hosted models and are hard to reproduce and measure [@huang2024mlagentbench; @hollmann2023caafe]. We therefore decided that every decision of our system had to come from rules and statistics that run offline and can be checked against recorded numbers [@demsar2006].

## Proposed solution in brief

AutoPilot-ML is an autonomous experimentation layer built on top of FLAML. It runs the data scientist's loop by itself, as a session of experiments within a time budget. After each experiment, a Critic diagnoses weaknesses from evidence measured on 15 fixed folds of the training data. A Strategy Engine proposes changes that answer these diagnoses, and a Judge keeps a change only if a corrected resampled t-test on the same folds says that it is better, or that it is simpler and not worse [@nadeau2003inference; @bouckaert2004replicability]. A Resource Manager divides the time budget between the experiments, and an Experiment Memory stores every change with its measured effect and cost, so that a bandit method can learn which changes tend to work on similar datasets [@thompson1933; @auer2002ucb]. Figure @fig:loop shows this loop and the part of it that FLAML performs [@wang2021flaml]. Each box is one step, and each arrow is labelled with what passes to the next step. Only the grey box, the search itself, is done by FLAML; we built every other step in this project.

::figure fig:loop | figures/fig_loop.png | 15.5 | The experimentation loop of AutoPilot-ML; the search inside each experiment is done by FLAML

## Contributions

The main contributions of the project, which we built and measured on top of FLAML [@wang2021flaml], are these.

- An autonomous experimentation layer around FLAML with 14 kinds of diagnosis and 12 strategies. Every decision it takes is recorded with its numbers and can be replayed from the saved records [@wang2021flaml].
- The Judge, a cheap acceptance rule that refits the configurations found by FLAML on fixed folds and compares them with a corrected resampled t-test, together with a study of its error rates on data with known answers [@nadeau2003inference; @bouckaert2004replicability].
- A learned choice of the next change, made from the Experiment Memory with Thompson sampling or UCB1, so that changes that worked on similar datasets are tried first [@thompson1933; @auer2002ucb].
- A benchmark of 390 runs on 6 OpenML datasets, with 5 seeds and 2 time budgets, against FLAML alone, AutoGluon and a LightGBM baseline with default settings. We fixed its datasets and its analysis in writing before it ran [@vanschoren2013openml; @demsar2006].
- Optional parts: SMOTE, cross-fitted target encoding, word features, group-aware folds, SHAP explanations, a leakage warning, plain-language summaries written by a local language model, a local web API and a Docker image [@chawla2002smote; @micci2001target; @lundberg2017shap].

## Organisation of the report

Chapter 2 reviews hyperparameter search, AutoML systems, the way FLAML searches, learning across datasets and the statistics of comparing models [@hutter2019automl; @demsar2006]. Chapter 3 states the problem and the objectives, together with what was delivered for each, and chapter 4 describes our method: the components of the loop, the datasets and the benchmark [@vanschoren2013openml]. Chapter 5 gives the main algorithms, and chapter 6 the block diagram, the data flow diagrams and the flowchart. Chapter 7 covers the tools, the implementation and its tests, chapter 8 presents and discusses the results, and chapter 9 gives our conclusions and the future scope [@wang2021flaml].
