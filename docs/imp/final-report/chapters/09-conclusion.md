# Conclusion and future scope

## Conclusions

This project built AutoPilot-ML, an autonomous experimentation layer on top of conventional AutoML, and measured it against strong systems on public data. The main conclusions are the following.

1. A complete autonomous layer was built around FLAML, which it uses unchanged [@wang2021flaml]. It diagnoses each Experiment with 14 kinds of Diagnosis, answers them with 12 Strategies, accepts a change only through a corrected resampled t-test on 15 fixed folds, divides its budget with a Resource_Manager and learns the order of its changes from an Experiment_Memory with Thompson sampling or UCB1. It runs unattended and offline on a laptop, through a dashboard, a command line, a local web API or a Docker image, and it is covered by 586 automated tests.
2. Its decision components are reliable on data with known answers. The Critic's data-level Diagnoses gave no false alarm, overfitting was found with a precision of 0.879 and a recall of 0.906, and the Judge kept none of 18 harmful changes while finding 10 of 12 helpful ones.
3. AutoPilot-ML makes strong use of a larger budget. From 2 to 5 minutes its score rose on 5 of the 6 benchmark datasets and its average rank rose from fourth to second of five systems. At 5 minutes it matched FLAML alone within the statistical intervals while using 29% less time, and it scored higher than AutoGluon on 4 of the 6 datasets and higher than a LightGBM baseline with default settings on 5.
4. The loop detects and repairs data problems that a search alone cannot fix, and on such data it reached higher scores than FLAML alone: on Titanic by 0.021 F1 on average over three seeds, and on pc4 with the best score of all systems (0.6218 against 0.5897 F1).
5. Every decision is explained, recorded and repeatable. Each step stores its Diagnoses, its change, the Judge's numbers and its cost, so any result can be traced back to the evidence behind it.

Together these results show that an autonomous experimentation layer can reach the score level of a strong AutoML search in less time at a 5-minute budget, and can raise the score where the data has problems that a search cannot fix by itself, while every one of its decisions stays measured and recorded. They also show that such a layer can be built from open-source tools on a laptop, without a GPU, a paid service or a language model in any decision. All seven objectives of chapter 3 were delivered.

## Possible applications

The way AutoPilot-ML works suits several kinds of use.

- Data that must stay on the machine: the system needs no network connection or cloud service, so it can work on private records, such as patient or customer data, on an ordinary laptop.
- Models that must be explained: every Session records why each change was made and how it was judged, which helps when a model and the preparation of its data have to be explained to others or reviewed later.
- Teaching: the dashboard shows each Diagnosis, change and Verdict as it happens, so students can follow a complete experimentation loop with real statistics instead of a single score.
- A strong starting point: within a few minutes the loop gives a tested model with a full record of how it was reached, from which a data scientist can continue by hand.
- Research on AutoML: the benchmark harness, its statistics and the switches for each part of the loop can be reused to compare new Strategies or selectors on other datasets.

## Future scope

The project opens several directions for further work.

- Shorter budgets: give the first search a larger share of a short budget, or start the loop only above a set budget, so that the gains seen at 5 minutes also appear at 2 minutes. FLAML makes most of its progress in its first seconds, so the first search matters most when time is short [@wang2021flaml].
- Use of the saved time: a Session of the full system finished about 88 seconds before the end of a 300-second budget on average, and this time could go to a final and longer search of the Accepted_Setup.
- A larger memory: Sessions on more datasets would give the learned order of changes more evidence, so that Thompson sampling and UCB1 could move more changes away from the rule order [@thompson1933; @auer2002ucb].
- Imbalanced classes: tuning the decision threshold on the folds would add another cheap answer to class imbalance, next to class weights and SMOTE [@he2009imbalance; @chawla2002smote].
- A larger benchmark: more datasets, longer budgets and a stronger AutoGluon preset would sharpen the comparison, and the harness can already switch off the memory, the Critic, the strategy choice and the Resource_Manager one at a time to measure each part [@gijsbers2024amlb; @erickson2020autogluon].
- New Diagnoses: rules for outliers, for strongly skewed numeric columns and for a shift between the training data and new data would let the loop answer more kinds of data problems, each with a Strategy of its own.
- Parallel candidates: a Session searches one candidate at a time with all the cores of the laptop; searching several candidates at once on several machines would let a Session try more changes in the same time.
- Language models as proposers: a language model could propose new changes that no rule table contains, while the Judge stays the gate that accepts or rejects them, so that every decision is still measured [@hollmann2023caafe].
- Wider support: feature engineering across columns, a database and a login for the web API, and support for time series and text data would bring the system to more kinds of work.
