# Result discussion and analysis

This chapter presents the results of AutoPilot-ML in the order of the work: the baseline pipeline, the first autonomous Session, the reliability of the Critic and the Judge on data with known answers, and the comparison with FLAML, AutoGluon and a LightGBM baseline on public datasets. All numbers were measured on the laptop described in chapter 7, and the records of every run are kept with the project. FLAML's search is limited by time, so repeated runs can differ slightly; the benchmark scores are therefore means over 5 seeds.

## The baseline pipeline

One FLAML search of 10 seconds on each sample dataset (seed 42), scored on the held-out test rows, was far ahead of a trivial baseline on the same rows [@wang2021flaml]. It reached a test F1 of 0.6935 on Titanic and 0.9512 on breast cancer, where the trivial baseline scores 0, a macro F1 of 0.9829 on penguins against 0.2020, and an R² of 0.8566 on California housing against -0.0002.

Two findings from the baseline shaped the design of the loop. On Titanic with 60 seconds, dropping the `Ticket` and `Cabin` columns raised the test F1 from 0.673 to 0.709, but a paired bootstrap over the 179 test rows gave a 95% interval of -0.042 to +0.117 for the difference, so a single split could not show whether the change helped [@efron1993bootstrap]. A fair comparison needs many folds and a proper test, which became the Judge. FLAML's trial history also showed that 81% of the rise in its validation F1 came in the first 0.8 seconds of a 60-second search, while the search went on until the budget ran out [@wang2021flaml]. Part of a budget can therefore be spent on changes to the data and the settings, which became the Resource_Manager.

## The first autonomous Session

The first Session ran on Titanic with 300 seconds, at most 8 Experiments and seed 42, and Table @tab:titanic shows its steps, with the validation F1 as the mean over the 15 Session_Folds. Its first Setup left out `PassengerId` and `Name` as identifier-like and was searched for 105 seconds, 35% of the budget. The Critic found strong overfitting (training F1 0.996 against 0.706 on the folds) and high cardinality in `Cabin` and `Ticket`; for `Ticket`, 72% of the validation rows had a ticket not seen in training. The Strategy_Engine answered with three changes in turn, and the Judge kept all three: dropping `Cabin` as simpler and not worse, dropping `Ticket` as better with p(better) 0.010, and smaller trees as simpler. The Session then stopped after 206 seconds because no change was left to try.

::table tab:titanic | Steps of the first autonomous Session on Titanic (300 s, seed 42) | widths=1.1,4.6,1.9,2.1,2.0,1.7,2.4 | align=ccccccc
| Step | Change tried | Search budget (s) | Validation F1 | Mean difference | p(better) | Verdict |
|---|---|---|---|---|---|---|
| 0 | First Setup, without PassengerId and Name | 105 | 0.7056 | | | start |
| 1 | drop_columns: Cabin | 25 | 0.7143 | +0.0087 | 0.284 | keep_simpler |
| 2 | drop_columns: Ticket | 23 | 0.7649 | +0.0506 | 0.010 | keep |
| 3 | regularised_search_space | 23 | 0.7628 | -0.0022 | 0.561 | keep_simpler |
::end

On the test rows, used once at the end, the final model was better than the first on the main metrics: an F1 of 0.7244 against 0.6838, an accuracy of 0.8045 against 0.7933, a ROC AUC of 0.8123 against 0.7808 and a log loss of 0.5261 against 0.6411. The loop had found by itself the change that the manual analysis of the baseline pointed to, and confirmed it on the 15 folds.

## Reliability of the Critic and the Judge

The Critic was measured on the 135 synthetic datasets of the reporting seeds, which the tuning of its thresholds never saw. Precision is the share of its alarms that were real problems, and recall the share of real problems it found. All six data-level kinds of Diagnosis (class imbalance, constant, high-cardinality and identifier columns, empty cells and redundant columns) had a precision of 1.000, so none of their alarms was false. The Critic found every planted case of class imbalance, high cardinality, constant columns and empty cells, and it found overfitting, the kind most affected by noise, with a precision of 0.879 and a recall of 0.906 [@hastie2009esl].

The Judge was measured on 84 pairs of Setups with known true effects. With its chosen settings, 3 repeats of 5-fold cross-validation and alpha 0.20, it kept none of the 18 harmful changes of the reporting seeds and found 10 of the 12 truly helpful ones [@nadeau2003inference]. This matters most for an autonomous loop, which tries many changes and must not accept the harmful ones.

## First comparison on the sample datasets

Before the benchmark, a Session with the rule order and FLAML alone were run with the same 300 seconds on the four sample datasets and three seeds. Over the 12 runs the Session matched FLAML alone on average (a mean difference of +0.0004 in test score) while using about two thirds of the time. On Titanic, where the Critic found real data problems that FLAML cannot fix by itself, the Session was better by 0.021 F1 on average over the three seeds. On penguins both reached the same test score in every run, and the Session needed 132 seconds against 303 for FLAML alone.

## The benchmark

The benchmark completed all 390 runs of its grid in about 16 hours on the laptop, with no failed and no excluded run. The memory read by the full system held 95 records from the 30 development Sessions.

### Test scores at the 300-second budget

Table @tab:scores300 gives the mean test score of each system over the 5 seeds at the 300-second budget. The scores are F1 for the classification datasets (macro F1 for the multiclass ones) and R² for energy_efficiency and solar_flare. The best value of each row is in bold.

::table tab:scores300 | Mean test score over 5 seeds at the 300 s budget | widths=3.5,2.3,2.2,2.3,2.6,2.6 | align=jccccc
| Dataset | Manual LightGBM | FLAML alone | AutoGluon | AutoPilot-ML rules | AutoPilot-ML full |
|---|---|---|---|---|---|
| pc4 | 0.6125 | 0.5897 | 0.5507 | 0.6194 | **0.6218** |
| cylinder-bands | 0.8485 | **0.8547** | 0.8375 | 0.8426 | 0.8529 |
| segment | **0.9814** | 0.9805 | 0.9723 | 0.9766 | 0.9771 |
| MiceProtein | 0.9855 | **0.9989** | 0.9947 | 0.9866 | 0.9921 |
| energy_efficiency | 0.9972 | **0.9989** | 0.9987 | 0.9988 | 0.9988 |
| solar_flare | 0.0457 | 0.1043 | **0.1268** | 0.1166 | 0.1132 |
::end

At 300 seconds the full system was level with FLAML alone: the median of its differences over the six datasets was -0.0010, with a 95% bootstrap interval from -0.0051 to +0.0205 that contains zero [@efron1993bootstrap]. It ranked second of the five systems (Table @tab:ranks), scored higher than AutoGluon on 4 of the 6 datasets (median difference +0.0025) and higher than the LightGBM baseline on 5 of the 6 (median difference +0.0055). On the imbalanced pc4 dataset it gave the best score of all systems, an F1 of 0.6218 against 0.5897 for FLAML alone.

### Improvement with a larger budget

Figure @fig:scores shows the scores of the five systems at both budgets. From 120 to 300 seconds the score of the full system rose on 5 of the 6 datasets, more than any other system, and stayed level on energy_efficiency (0.9989 and 0.9988); FLAML alone rose on 2. On pc4 the full system went from 0.5840 to 0.6218 F1 and the rule-order Session from 0.5483 to 0.6194, and on solar_flare the full system went from 0.1023 to 0.1132 R². The median gap to FLAML alone closed from 0.0137 at 120 seconds to 0.0010 at 300 seconds.

Table @tab:ranks gives the average rank of each system over the six datasets at both budgets. The two AutoPilot-ML systems were the only ones whose rank improved with the larger budget: the full system moved from 3.58, fourth of five, to 2.50, second of five, and the rule-order Session from 3.75 to 3.00.

::table tab:ranks | Average rank of the five systems over the six datasets at each budget (1 is best) | widths=5.2,2.6,2.6 | align=jcc
| System | 120 s | 300 s |
|---|---|---|
| AutoPilot-ML full | 3.58 | 2.50 |
| AutoPilot-ML rules | 3.75 | 3.00 |
| FLAML alone | 1.50 | 2.17 |
| AutoGluon | 3.33 | 3.67 |
| Manual LightGBM | 2.83 | 3.67 |
::end

::figure fig:scores | figures/fig_scores.png | 15.5 | Mean test score of the five systems at 120 s and 300 s on the six evaluation datasets

This fits the design of the loop. The first search receives 35% of the budget and every change is then checked on 15 folds, so a longer budget leaves more room for the changes that the Critic proposes, while a single FLAML search makes most of its progress in its first seconds [@wang2021flaml].

### Efficiency

Figure @fig:time shows the mean wall time per run at the 300-second budget. FLAML alone uses the whole budget by design, while a Session can stop early once it has no promising change left to try. The full system needed 212.3 seconds per run on average against 300.9 seconds for FLAML alone, 29% less time for the same level of score. AutoGluon's medium_quality preset finished after 98.5 seconds on average, with a lower score than the full system on 4 of the 6 datasets, and the LightGBM baseline trains a single model in under a second.

::figure fig:time | figures/fig_time.png | 15.5 | Mean wall time per run of the five systems at the 300 s budget

### What the Sessions changed

Across all Session runs of the benchmark, the Judge kept 399 of the 738 changes that were tried after the first Experiment. The most frequent change was dropping one of two nearly equal columns, and most kept changes made the Setup simpler, with fewer columns or learner families, without a measurable loss in validation score. Word features were kept on cylinder-bands and SMOTE on the imbalanced pc4 dataset [@chawla2002smote]. The changes raised the test score most on these two datasets: at 300 seconds the mean test score of the full system rose from 0.6038 for its first model to 0.6218 for its final model on pc4, where its kept changes were column drops, and from 0.8472 to 0.8529 on cylinder-bands.

## Discussion

The results show that AutoPilot-ML delivers what it was designed for. It runs the whole experimentation loop by itself, offline on a laptop, and bases every decision on measured evidence that is saved with the run. Its two decision components were reliable on data with known answers: the Critic's data-level Diagnoses raised no false alarm, and the Judge accepted none of 18 harmful changes.

Against strong systems the loop performed well and made good use of a larger budget. At 5 minutes it was level with FLAML alone within the statistical intervals while using 29% less time, ranked second of five systems, and scored higher than AutoGluon and the LightGBM baseline on most datasets. FLAML is a mature engine from Microsoft Research, so matching it while adding Diagnoses, statistical decisions and a full record of every step is a strong outcome [@wang2021flaml].

The loop helped most where the data had a problem that a search cannot fix by itself. On Titanic it found and dropped the misleading `Ticket` column and raised the test F1 from 0.6838 to 0.7244, and on the imbalanced pc4 dataset it gave the best score of all systems. In both cases the reason for the result can be read from the recorded Diagnoses and Verdicts, which a single AutoML search does not provide.

AutoPilot-ML also returns more than the other systems. FLAML alone and AutoGluon return a model with its validation score, and the LightGBM baseline returns a single model [@wang2021flaml; @erickson2020autogluon]. A Session returns its final model together with the Diagnoses of every step, the changes it tried, the Judge's numbers behind each decision, the first and the final model side by side on the test rows, and a memory record of every Verdict that later Sessions can learn from.
