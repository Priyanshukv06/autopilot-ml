# Background details and literature review

This chapter reviews the work that AutoPilot-ML builds on and the work it is compared with. It covers models for tabular data, hyperparameter search, AutoML systems, the internals of FLAML, learning across datasets, common data problems, fair comparison of models, explanations and recent agents based on language models [@hutter2019automl; @demsar2006].

## Machine learning on tabular data

A tabular dataset has one row per example and one column per feature, with a target column to predict. Decision trees split the rows on one feature at a time, and ensembles of many trees are the usual strong choice: random forests average trees grown on random samples of rows and features [@breiman2001rf], while gradient boosting adds small trees one after another, each correcting the errors of the earlier ones, as in LightGBM and XGBoost [@ke2017lightgbm; @chen2016xgboost].

A large benchmark of 45 tabular datasets found that tree-based models remain ahead of deep learning on medium-sized tabular data, partly because they are not disturbed by uninformative features and irregular target functions [@grinsztajn2022trees]. This is why the learners searched in this project are mainly tree ensembles, with linear models as cheap alternatives [@wang2021flaml].

Building such a model in practice is a loop of experiments around the learner. A case study at Microsoft describes a workflow of data collection, cleaning, feature engineering, training, evaluation and monitoring, with frequent returns to earlier steps [@amershi2019se4ml]. Sculley et al. point out that most of the cost of a machine learning system lies outside the learning code, in data dependencies and configuration, and that undeclared changes in data quietly harm models [@sculley2015debt].

## Hyperparameter optimisation

Every learner has hyperparameters, such as the number of trees or the maximum depth, that must be chosen before training. Grid search tries every combination of a few values, and random search samples combinations at random. Bergstra and Bengio showed that random search finds good settings with far fewer trials than a grid when only a few hyperparameters matter, which is the usual case [@bergstra2012random].

Bayesian optimisation builds a model of the validation score as a function of the hyperparameters and chooses the next trial where improvement looks likely. SMAC uses random forests as this model [@hutter2011smac], the tree-structured Parzen estimator (TPE) models good and bad regions separately [@bergstra2011tpe], and Gaussian processes give a smooth model with uncertainty [@snoek2012bo].

A third family saves time by stopping weak trials early. Successive Halving trains many configurations on a small budget and keeps only the best fraction for a larger one [@jamieson2016sh], Hyperband runs Successive Halving with several starting budgets [@li2018hyperband], and BOHB replaces Hyperband's random sampling with a TPE-like model [@falkner2018bohb].

## AutoML systems

Thornton et al. defined the combined algorithm selection and hyperparameter optimisation (CASH) problem and solved it with Auto-WEKA, which treats the choice of learner as one more hyperparameter of a single search space [@thornton2013autoweka]. A book edited by Hutter, Kotthoff and Vanschoren summarises the methods and systems that followed [@hutter2019automl].

Auto-sklearn applies Bayesian optimisation to scikit-learn pipelines, starts its search from configurations that worked on similar datasets (meta-learning) and combines the models it found into an ensemble [@feurer2015autosklearn]. Its second version replaces the meta-features by a portfolio of configurations chosen offline and selects its validation strategy by policy [@feurer2022autosklearn2].

TPOT represents a whole pipeline of preprocessing steps and a model as a tree and improves a population of such pipelines with genetic programming [@olson2016tpot]. It can find unusual pipelines, but a search takes many generations.

AutoGluon-Tabular takes a different view: instead of a long search, it trains a fixed list of strong models with good default settings and stacks them in several layers, which gave high accuracy in its authors' benchmarks [@erickson2020autogluon]. H2O AutoML trains a sequence of models, partly with random grid search, and combines them into stacked ensembles [@ledell2020h2o].

Comparing these systems fairly needs common datasets and protocols. The AutoML Benchmark (AMLB) runs systems on curated OpenML datasets with fixed time budgets and reports where each system wins [@gijsbers2024amlb]. OpenML itself is an open platform for sharing datasets, tasks and results [@vanschoren2013openml], and its curated benchmarking suites, such as OpenML-CC18, are designed for such studies [@bischl2021suites].

## How FLAML searches

FLAML, the Fast and Lightweight AutoML library from Microsoft Research, is the search engine used inside every Experiment of AutoPilot-ML [@wang2021flaml]. Its central idea is that trials have very different costs: a model with 4 small trees trains many times faster than one with thousands of large trees, so a search that starts cheap and becomes expensive only when that looks worthwhile finds good models quickly on small budgets [@wang2021flaml; @wu2021cfo]. Because the autonomous layer depends on exactly what FLAML does, we studied its paper and the source code of version 2.7.0 [@flamlrepo]. This section explains what we found. FLAML is the work of its authors; AutoPilot-ML uses it without any change.

FLAML keeps a separate search state for each learner, such as LightGBM, XGBoost or random forest, and in each pass of its loop it picks one learner, asks that learner's search for one configuration, trains and scores it, and updates the learner's record [@wang2021flaml]. The learner is picked by its estimated cost for improvement (ECI), the time it is expected to need before it beats its own best score. In the source code [@flamlrepo] this is computed as in Equation (@eq:eci), where for learner l, t_{l} is the total time spent on it, t_{l}^{b} the time at which its current best was found, t_{l}^{b′} the time of its previous best, ε_{l} its best loss, ε^{*} the best loss of all learners and v_{l} its recent rate of improvement. The third term only counts for a learner that is behind the leader: it is the time it would need, at its recent speed, to close twice the gap.

::equation eq:eci

A learner that has not been tried yet gets a starting guess, its typical cost relative to LightGBM multiplied by the cost of the first trial, and it is tried first when that guess is the smallest. Otherwise FLAML draws the learner at random with the probability of Equation (@eq:select), so cheap learners that still improve are tried most often [@wang2021flaml; @flamlrepo]. In our classification runs the learners were LightGBM, XGBoost, XGBoost with limited depth, random forest, extra trees, a linear model trained by stochastic gradient descent and L1-regularised logistic regression, with starting guesses of 1, 1.6, 64, 2, 1.9, 1 and 160 times the cost of LightGBM.

::equation eq:select

Within one learner, FLAML's default search is CFO (Cost-Frugal Optimization) with its local search FLOW² [@wu2021cfo]. The first configuration uses small values for the settings that drive cost, such as 4 trees and 4 leaves for LightGBM and XGBoost, and default values for the rest. All settings are scaled to the range 0 to 1 (on a log scale for wide ranges). From the best point x*, CFO draws a random direction u and tries the point x′; if that is not better it tries x″ on the other side, as in Equation (@eq:cfo), where d is the number of settings [@wu2021cfo; @flamlrepo]. The step δ shrinks only after 2 to the power min(d, 9) proposals in a row without improvement, which is 256 for LightGBM and 512 for XGBoost. When the step falls below a lower bound, the local search has converged and restarts from the starting point with random noise.

::equation eq:cfo

FLAML also offers BlendSearch, which combines a global search method with the local search of CFO so that it can leave a poor region of the space [@wang2021blend]. In the source code, the default setting `auto` resolves to CFO when trials run one at a time, without the Ray or Spark back ends, which is how AutoPilot-ML runs it; BlendSearch would also need the Optuna package, which is not installed [@flamlrepo].

The validation of each trial follows a fixed rule in the source code: 5-fold cross-validation when the number of rows times the number of columns divided by 0.9 is below 10 million times the time budget in hours and there are fewer than 100,000 rows, and otherwise a holdout of 10% of the training rows [@flamlrepo]. After the search, the best configuration is trained again on all training rows. The search may stop early, but only when every learner's local search has converged and the search has run more than ten times as long as it took to find its current best, which did not happen in our runs on small data. FLAML also has a zero-shot mode that picks data-dependent default configurations mined offline from many datasets [@kayali2022zeroshot].

Figure @fig:flaml summarises this loop as we read it from the code [@flamlrepo]. Two of its properties shaped the design of AutoPilot-ML. First, FLAML searches only the columns it is given: it cannot decide to drop a misleading column or to weight a rare class. Second, the validation score it reports is the best of hundreds of trials on the same folds, so it is optimistic and cannot be used to compare two different searches fairly [@cawley2010overfit]. The autonomous layer therefore changes the data and settings around FLAML, and compares changes by refitting FLAML's chosen configurations on folds of its own.

::figure fig:flaml | figures/fig_flaml.png | 10.6 | FLAML's search loop as read from its source code (version 2.7.0), drawn for this report

## Learning across datasets

Meta-learning uses experience from earlier datasets to make learning on a new one faster or better; Vanschoren's survey groups the methods by what they transfer: model evaluations, dataset properties (meta-features) or trained models [@vanschoren2018meta]. Auto-sklearn starts its search from the best configurations of the most similar earlier datasets [@feurer2015autosklearn], and FLAML's zero-shot mode chooses defaults from a portfolio mined offline [@kayali2022zeroshot].

OBOE treats model selection as collaborative filtering: it fits a low-rank model to a matrix of model errors on many datasets, and on a new dataset it runs a few cheap models within the time budget to predict which other models will do well [@yang2019oboe]. All these methods transfer good model and hyperparameter choices. AutoPilot-ML's memory transfers something different: changes to an experiment, such as dropping a column or weighting the classes, with their measured effect and cost on the user's own earlier datasets [@vanschoren2018meta].

Choosing which change to try when the evidence is uncertain is an exploration and exploitation problem, studied as the multi-armed bandit. Thompson sampling draws a plausible success rate for each option from its posterior distribution and picks the best draw [@thompson1933], and it performs well in practice [@chapelle2011ts]. UCB1 adds to each option's mean reward a bonus that shrinks as the option is tried more often, with a proven bound on regret [@auer2002ucb]. AutoPilot-ML uses both as optional ways to order its Strategies.

## Data problems and the changes that answer them

Class imbalance, when one class is much rarer than the others, makes learners favour the majority class [@he2009imbalance]. Common answers are weighting the classes in the loss, a form of cost-sensitive learning [@elkan2001cost], and creating synthetic rows of the rare class with SMOTE, which places new points between a rare example and one of its nearest neighbours of the same class [@chawla2002smote]. The imbalanced-learn library implements SMOTE and its variants for mixed and categorical data [@lemaitre2017imblearn].

Text columns with many different values, such as ticket numbers or postcodes, are hard for tree learners: most values are rare, and values in new data were never seen in training. Frequency encoding replaces a value by how often it occurs, and target encoding replaces it by a smoothed mean of the target for that value [@micci2001target]. Target encoding computed on the same rows the model trains on leaks the label into the feature, which is why AutoPilot-ML computes it in a cross-fitted way.

Data leakage means that a feature contains information that will not be available when the model is used, for example a column filled in after the outcome; Kaufman et al. show how common it is and how it makes models look far better than they are [@kaufman2012leakage]. A related problem appears when rows come in groups, such as several rows per patient: random folds put rows of the same group on both sides, and grouped cross-validation is needed for an honest estimate [@roberts2017groupcv].

Overfitting happens when a model learns noise in the training rows; it shows as a large gap between training and validation scores, and the usual answers are simpler models or stronger regularisation, at the price of some bias [@hastie2009esl]. AutoPilot-ML's Critic looks for each of these problems, and its Strategy_Engine has at least one change that answers each of them.

## Comparing models fairly

Cross-validation estimates how a model will do on new data by training on part of the rows and scoring on the rest, several times over [@kohavi1995cv]. When the best of many configurations is chosen on the same validation folds, its validation score is biased upwards; Cawley and Talbot show that this selection bias can be as large as the differences between learning methods [@cawley2010overfit].

Deciding whether one model is better than another needs a statistical test, but the scores of different folds are not independent, because their training sets overlap. Dietterich found that the usual paired t-test on resampled scores rejects far too often [@dietterich1998tests]. Nadeau and Bengio proposed a corrected resampled t-test that inflates the variance by a term for the overlap of training sets [@nadeau2003inference], and Bouckaert and Frank found that this correction applied to repeated cross-validation gives more replicable decisions than the usual tests [@bouckaert2004replicability]. The Judge of AutoPilot-ML uses this test.

Comparing systems over several datasets needs other tests, because each dataset gives one observation. Demšar recommends the Wilcoxon signed-rank test for two systems and the Friedman test with the Nemenyi post-hoc test for several [@demsar2006; @wilcoxon1945; @friedman1937; @nemenyi1963], with Iman and Davenport's version of the Friedman statistic as a less conservative option [@iman1980]. When several pairs are tested together, Holm's step-down method controls the chance of any false claim [@holm1979], and bootstrap intervals show how uncertain a difference is [@efron1993bootstrap].

## Explaining models

Permutation importance measures how much a model's score drops when the values of one column are shuffled; it was introduced with random forests [@breiman2001rf] and studied in a model-agnostic form by Fisher et al. [@fisher2019importance]. SHAP gives each feature a share of a single prediction, based on Shapley values from game theory, so that the shares add up to the prediction minus the average prediction [@lundberg2017shap]. For tree ensembles these values can be computed exactly and quickly [@lundberg2020treeshap]. AutoPilot-ML shows permutation importance for every model and SHAP values for the final model of a Session.

## Language-model agents for machine learning

Large language models (LLMs) are now used to automate parts of data science. CAAFE asks an LLM to write feature-engineering code from a description of the dataset and keeps the code only if cross-validation improves [@hollmann2023caafe]. MLAgentBench gives LLM agents machine learning tasks with files and tools and measures how often they improve a baseline [@huang2024mlagentbench], and MLE-bench measures agents on 75 Kaggle competitions [@chan2024mlebench]. AIDE searches over whole solution scripts written by an LLM, improving the best script found so far [@jiang2025aide].

These agents can propose changes that no fixed rule table contains, which is a strength. They also depend on large models, most of them hosted, and their decisions are written in generated text that is hard to check or repeat [@huang2024mlagentbench; @chan2024mlebench]. AutoPilot-ML takes the opposite path: its decisions come from rules and a statistical test that run offline and can be replayed from the records, and a small local model (Qwen2.5 7B through Ollama) is used only to write a labelled summary that no decision reads [@qwen2024].

## Research gap

Table @tab:compare compares AutoPilot-ML with the systems above on five abilities. The AutoML systems search models and settings well, and several transfer good configurations across datasets, but none of them explains why a model is weak or decides on its own to change the data before searching again [@feurer2015autosklearn; @erickson2020autogluon; @wang2021flaml]. LLM agents can make such changes, but not with a measured, repeatable rule [@hollmann2023caafe; @huang2024mlagentbench].

::table tab:compare | Comparison of AutoPilot-ML with existing systems | widths=2.2,2.9,2.6,2.2,2.7,3.3 | align=cccccc
| System | Model and hyperparameter search | Changes to data and features | Explains why a model is weak | Learns across datasets | Budget decisions across experiments |
|---|---|---|---|---|---|
| Auto-sklearn [@feurer2015autosklearn] | Bayesian optimisation of pipelines | Preprocessing choices inside the search space | No | Warm start from configurations found offline | Time limit of one search |
| TPOT [@olson2016tpot] | Genetic programming of pipelines | Pipeline operators | No | No | Generations or time limit |
| AutoGluon-Tabular [@erickson2020autogluon] | Little search: strong defaults and stacking | Built-in preprocessing | No | Fixed presets | Time limit and presets |
| FLAML [@wang2021flaml] | Cost-frugal search (ECI and CFO) | Built-in preprocessing | No | Zero-shot defaults mined offline | Cost-aware within one search |
| OBOE [@yang2019oboe] | Time-limited model selection | No | No | Error matrix of many datasets | Within one selection |
| LLM agents [@hollmann2023caafe; @huang2024mlagentbench] | Through code written by the model | Yes, written by the model | In generated text, not checked | Not as a rule | No explicit rule |
| AutoPilot-ML (this work) | Through FLAML, unchanged | Yes: 12 Strategies, fitted inside each fold | Yes: Critic with 14 kinds of Diagnosis and their evidence | Memory of changes with their effect and cost | Resource_Manager with budgets and stop rules |
::end

The gap this project addresses is therefore an experimentation layer that diagnoses, changes, judges and records, with every step measurable, offline and repeatable [@demsar2006]. Whether such a layer also gives better models than a single AutoML search with the same budget is an empirical question, which chapter 8 measures on six public datasets and two budgets [@wang2021flaml].
