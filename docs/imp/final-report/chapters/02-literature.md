# Background details and literature review

In this chapter we review the work that AutoPilot-ML builds on and the systems we compare it with. We start with models for tabular data and hyperparameter search, then turn to AutoML systems and the inner workings of FLAML, and end with learning across datasets, common data problems, fair comparison of models, explanations and recent agents built on language models [@hutter2019automl; @demsar2006].

## Machine learning on tabular data

In a tabular dataset each row is one example and each column one feature, and one of the columns is the target to be predicted. A decision tree splits the rows on one feature at a time, and ensembles of many trees are the usual strong choice. Random forests average trees grown on random samples of the rows and features [@hastie2009esl]. Gradient boosting, as in LightGBM and XGBoost, adds small trees one after another, and each new tree corrects the errors left by the trees before it [@ke2017lightgbm; @chen2016xgboost].

A large benchmark on 45 tabular datasets found that tree-based models are still ahead of deep learning on medium-sized tabular data. Part of the reason is that trees are not thrown off by uninformative features or by irregular target functions [@grinsztajn2022trees]. This is why the learners searched in our project are mainly tree ensembles, with linear models as cheap alternatives [@wang2021flaml].

In practice, building such a model means running a loop of experiments around the learner. A case study at Microsoft describes a workflow that runs from data collection and cleaning through feature engineering, training and evaluation to monitoring, with frequent returns to earlier steps whenever a result shows that the data or the model has to change [@amershi2019se4ml].

## Hyperparameter optimisation

Every learner has hyperparameters, such as the number of trees or the maximum depth, that have to be set before training. Grid search tries every combination of a few chosen values, while random search samples combinations at random. When only a few hyperparameters really matter, which is the usual case, random search finds good settings with far fewer trials than a grid [@hutter2019automl].

Bayesian optimisation fits a model of the validation score as a function of the hyperparameters and places the next trial where an improvement looks likely. This model can be a Gaussian process, which gives a smooth estimate together with its uncertainty, a random forest, as in SMAC, or the tree-structured Parzen estimator (TPE), which models the good and the bad regions of the space separately [@hutter2019automl].

A third family of methods saves time by stopping weak trials early. Successive Halving trains many configurations on a small budget and passes only the best fraction on to a larger one. Hyperband runs Successive Halving with several starting budgets, and BOHB combines Hyperband with a model-based choice of configurations [@hutter2019automl]. FLAML, described in section 2.4, takes a different route and controls the cost of each trial directly [@wang2021flaml].

## AutoML systems

Thornton et al. defined the combined algorithm selection and hyperparameter optimisation (CASH) problem and solved it with Auto-WEKA, which treats the choice of learner as one more hyperparameter in a single search space [@thornton2013autoweka]. The book edited by Hutter, Kotthoff and Vanschoren gives an overview of the methods and systems that came after it [@hutter2019automl].

Auto-sklearn runs Bayesian optimisation over scikit-learn pipelines. It starts its search from configurations that worked well on similar datasets and combines the models it finds into an ensemble [@feurer2015autosklearn]. TPOT represents a whole pipeline of preprocessing steps and a model as a tree and improves a population of such pipelines with genetic programming. It can find unusual pipelines, but a search needs many generations [@olson2016tpot].

AutoGluon-Tabular takes another approach. Instead of running a long search, it trains a fixed list of strong models with good default settings and stacks them in several layers, which gave high accuracy in its authors' benchmarks [@erickson2020autogluon]. H2O AutoML, another system in the AutoML Benchmark, also trains a sequence of models and combines them into stacked ensembles [@gijsbers2024amlb].

A fair comparison of these systems needs shared datasets and protocols. The AutoML Benchmark (AMLB) runs the systems on curated OpenML datasets with fixed time budgets and reports where each one wins [@gijsbers2024amlb]. OpenML itself is an open platform for sharing datasets, tasks and results [@vanschoren2013openml], and its curated benchmarking suites, such as OpenML-CC18, were designed for this kind of study [@bischl2021suites].

## How FLAML searches

FLAML, the Fast and Lightweight AutoML library from Microsoft Research, is the search engine inside every experiment of AutoPilot-ML [@wang2021flaml]. Its central idea is that trials differ greatly in cost. A model with 4 small trees trains many times faster than one with thousands of large trees, so a search that starts cheap and becomes expensive only when that seems worth it finds good models quickly on small budgets [@wang2021flaml; @wu2021cfo]. Our layer depends on exactly what FLAML does, so we studied its paper and the source code of version 2.7.0 [@flamlrepo], and this section explains what we found. FLAML is the work of its authors, and AutoPilot-ML uses it without any change.

FLAML keeps a separate search state for each learner, such as LightGBM, XGBoost or random forest. In each pass of its loop it picks one learner, asks that learner's search for one configuration, trains and scores it, and updates the learner's record [@wang2021flaml]. The learner is picked by its estimated cost for improvement (ECI), the time it is expected to need before it beats its own best score. In the source code [@flamlrepo], ECI is computed as in Equation (@eq:eci). For a learner l, t_{l} is the total time spent on it, t_{l}^{b} the time at which its current best was found, t_{l}^{b′} the time of its previous best, ε_{l} its best loss, ε^{*} the best loss over all learners and v_{l} its recent rate of improvement. The third term only counts for a learner that is behind the leader: it is the time that learner would need, at its recent speed, to close twice the gap.

::equation eq:eci

A learner that has not been tried yet gets a starting guess, which is its typical cost relative to LightGBM multiplied by the cost of the first trial, and it is tried first when that guess is the smallest. Otherwise FLAML draws the learner at random with the probability of Equation (@eq:select), so that cheap learners that are still improving are tried most often [@wang2021flaml; @flamlrepo]. In our classification runs the learners were LightGBM, XGBoost, XGBoost with limited depth, random forest, extra trees, a linear model trained by stochastic gradient descent and L1-regularised logistic regression, with starting guesses of 1, 1.6, 64, 2, 1.9, 1 and 160 times the cost of LightGBM.

::equation eq:select

Within one learner, FLAML's default search is CFO (Cost-Frugal Optimization) with its local search FLOW² [@wu2021cfo]. The first configuration uses small values for the settings that drive the cost, such as 4 trees and 4 leaves for LightGBM and XGBoost, and default values for the others. All settings are scaled to the range 0 to 1, on a log scale for wide ranges. From the best point x*, CFO draws a random direction u and tries the point x′. If that point is not better, it tries x″ on the other side, as in Equation (@eq:cfo), where d is the number of settings [@wu2021cfo; @flamlrepo]. The step δ shrinks only after 2 to the power min(d, 9) proposals in a row without improvement, which is 256 for LightGBM and 512 for XGBoost. When the step falls below a lower bound, the local search has converged, and it restarts from the starting point with some random noise.

::equation eq:cfo

FLAML also offers BlendSearch, which combines a global search method with the local search of CFO so that the search can escape from a poor region of the space. In the source code, the default setting `auto` resolves to CFO when trials run one at a time, without the Ray or Spark back ends, and this is how AutoPilot-ML runs it. BlendSearch would also need the Optuna package, which is not installed [@flamlrepo].

Each trial is validated by a fixed rule in the source code. FLAML uses 5-fold cross-validation when the number of rows times the number of columns, divided by 0.9, is below 10 million times the time budget in hours and there are fewer than 100,000 rows; otherwise it holds out 10% of the training rows [@flamlrepo]. After the search, the best configuration is trained again on all the training rows. The search may stop early, but only when the local search of every learner has converged and the search has run more than ten times as long as it took to find its current best, which did not happen in any of our runs on small data. FLAML also has a zero-shot mode that picks data-dependent default configurations mined offline from many datasets.

Figure @fig:flaml sums up this loop as we read it from the code [@flamlrepo]. Two of its properties shaped the design of AutoPilot-ML. First, FLAML searches only the columns it is given, so it cannot decide to drop a misleading column or to give a rare class more weight. Second, the validation score it reports is the best of hundreds of trials on the same folds. Such a score is optimistic and cannot be used to compare two different searches fairly [@cawley2010overfit]. Our layer therefore changes the data and the settings around FLAML, and it compares changes by refitting FLAML's chosen configurations on folds of its own.

::figure fig:flaml | figures/fig_flaml.png | 10.6 | FLAML's search loop as read from its source code (version 2.7.0), drawn for this report

## Learning across datasets

Meta-learning uses experience from earlier datasets to make learning on a new one faster or better. Vanschoren's survey groups the methods by what they transfer: model evaluations, properties of the datasets (meta-features) or trained models [@vanschoren2018meta]. Auto-sklearn, for example, starts its search from the best configurations of the most similar earlier datasets [@feurer2015autosklearn], and FLAML's zero-shot mode chooses its defaults from a portfolio mined offline [@flamlrepo].

These methods carry over good choices of model and hyperparameters. AutoPilot-ML's memory carries over something else: changes to an experiment, such as dropping a column or weighting the classes, together with their measured effect and cost on the user's own earlier datasets, which are found through their meta-features [@vanschoren2018meta].

Deciding which change to try when the evidence is uncertain is a problem of exploration and exploitation, studied as the multi-armed bandit. Thompson sampling draws a plausible success rate for each option from its posterior distribution and picks the option with the best draw [@thompson1933]. UCB1 adds to the mean reward of each option a bonus that shrinks the more often the option is tried, and it has a proven bound on regret [@auer2002ucb]. AutoPilot-ML offers both as optional ways of ordering its strategies.

## Data problems and the changes that answer them

Class imbalance, where one class is much rarer than the others, pushes learners towards the majority class. The common remedies are to weight the classes in the loss, a form of cost-sensitive learning, or to create synthetic rows of the rare class [@he2009imbalance]. SMOTE, the best known of these methods, places new points between a rare example and one of its nearest neighbours from the same class [@chawla2002smote]. The imbalanced-learn library implements SMOTE and its variants for mixed and categorical data.

Text columns with many different values, such as ticket numbers or postcodes, are hard for tree learners. Most of their values are rare, and new data brings values that were never seen in training. Frequency encoding replaces each value by how often it occurs, and target encoding replaces it by a smoothed mean of the target for that value [@micci2001target]. If target encoding is computed on the same rows that the model trains on, it leaks the label into the feature, which is why AutoPilot-ML computes it in a cross-fitted way.

Data leakage means that a feature carries information that will not be available when the model is used, for example a column that is filled in after the outcome is known. Kaufman et al. show how common leakage is and how it makes models look far better than they really are [@kaufman2012leakage]. A related problem arises when rows come in groups, such as several rows per patient. Random folds then put rows of the same group on both sides of a split, and an honest estimate needs folds that keep each group together.

Overfitting happens when a model learns the noise in its training rows. It shows up as a large gap between the training and the validation scores, and the usual answers are simpler models or stronger regularisation, at the price of some bias [@hastie2009esl]. AutoPilot-ML's Critic looks for each of these problems, and its Strategy Engine has at least one change that answers each of them.

## Comparing models fairly

Cross-validation estimates how a model will do on new data by training on part of the rows and scoring on the rest, several times over [@hastie2009esl]. When the best of many configurations is chosen on the same validation folds, its validation score is biased upwards, and Cawley and Talbot show that this selection bias can be as large as the differences between learning methods [@cawley2010overfit].

To decide whether one model is better than another we need a statistical test, but the scores of different folds are not independent, because their training sets overlap. Dietterich found that the usual paired t-test on resampled scores rejects the null hypothesis far too often [@dietterich1998tests]. Nadeau and Bengio proposed a corrected resampled t-test that inflates the variance by a term for the overlap of the training sets [@nadeau2003inference], and Bouckaert and Frank found that this correction, applied to repeated cross-validation, gives more replicable decisions than the usual tests [@bouckaert2004replicability]. The Judge of AutoPilot-ML uses this test.

Comparing systems over several datasets calls for other tests, because each dataset gives only one observation. Demšar recommends the Wilcoxon signed-rank test for two systems, the Friedman test with the Nemenyi post-hoc test for several, and Holm's step-down method when several pairs are tested together, while the average rank of each system over the datasets summarises where it stands [@demsar2006]. Bootstrap intervals, made by resampling the datasets many times, show how uncertain a difference is [@efron1993bootstrap].

## Explaining models

Permutation importance measures how much a model's score drops when the values of one column are shuffled, and random forests use it to rank their input features [@hastie2009esl]. SHAP gives each feature a share of a single prediction, based on Shapley values from game theory, so that the shares add up to the prediction minus the average prediction [@lundberg2017shap]. The SHAP library computes these values exactly for tree ensembles and estimates them for other models. AutoPilot-ML shows permutation importance for every model and SHAP values for the final model of a session.

## Language-model agents for machine learning

Large language models (LLMs) are now used to automate parts of data science. CAAFE asks an LLM to write feature-engineering code from a description of the dataset and keeps the code only if cross-validation improves [@hollmann2023caafe]. MLAgentBench gives LLM agents machine learning tasks with files and tools, and measures how often they manage to improve a baseline [@huang2024mlagentbench].

The strength of these agents is that they can propose changes that no fixed rule table contains. On the other hand, they depend on large models, most of them hosted, and their decisions are written as generated text that is hard to check or repeat [@huang2024mlagentbench]. AutoPilot-ML goes the other way. Its decisions come from rules and a statistical test that run offline and can be replayed from the records, and a small local model (Qwen2.5 7B through Ollama) only writes a labelled summary that no decision reads.

## Research gap

Table @tab:compare compares AutoPilot-ML with the systems above on five abilities. The AutoML systems search models and settings well, and several of them transfer good configurations across datasets, but none of them explains why a model is weak or decides by itself to change the data before searching again [@feurer2015autosklearn; @erickson2020autogluon; @wang2021flaml]. LLM agents can make such changes, but not with a measured, repeatable rule [@hollmann2023caafe; @huang2024mlagentbench].

::table tab:compare | Comparison of AutoPilot-ML with existing systems | widths=2.5,2.9,2.7,2.6,2.6,2.6 | align=cccccc
| System | Search of models and settings | Changes to data and features | Explains why a model is weak | Learns across datasets | Budget across experiments |
|---|---|---|---|---|---|
| Auto-sklearn [@feurer2015autosklearn] | Bayesian optimisation of pipelines | Preprocessing inside the search space | No | Warm start from earlier datasets | Time limit of one search |
| TPOT [@olson2016tpot] | Genetic programming of pipelines | Pipeline operators | No | No | Generations or time limit |
| AutoGluon-Tabular [@erickson2020autogluon] | Strong defaults and stacking | Built-in preprocessing | No | Fixed presets | Time limit and presets |
| FLAML [@wang2021flaml] | Cost-frugal search (ECI and CFO) | Built-in preprocessing | No | Zero-shot defaults | Cost-aware in one search |
| LLM agents [@hollmann2023caafe; @huang2024mlagentbench] | Code written by the model | Yes, written by the model | In text, not checked | Not as a rule | No explicit rule |
| AutoPilot-ML (this work) | Through FLAML, unchanged | Yes: 12 strategies, fitted in each fold | Yes: 14 kinds of diagnosis with evidence | Memory of changes, effects and costs | Budgets and stop rules |
::end

The gap our project addresses is therefore an experimentation layer that diagnoses, changes, judges and records, with every step measurable, offline and repeatable [@demsar2006]. In chapter 8 we compare such a layer with a single AutoML search given the same budget, and with AutoGluon and a LightGBM baseline, on six public datasets and two budgets [@wang2021flaml].
