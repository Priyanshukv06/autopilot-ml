# Result discussion and analysis

This chapter presents our results in the order of the work: the baseline, the first autonomous session, the reliability of the Critic and the Judge, and the comparison with FLAML, AutoGluon and a LightGBM baseline. All numbers were measured on the laptop of chapter 7, and the records of every run are kept. Because FLAML's search is limited by time, the benchmark scores are means over 5 seeds.

## The baseline pipeline

One FLAML search of 10 seconds per sample dataset (seed 42) was far ahead of a trivial baseline on the same held-out test rows [@wang2021flaml]. It reached a test F1 of 0.6935 on Titanic and 0.9512 on breast cancer, where the trivial baseline scores 0, a macro F1 of 0.9829 on penguins against 0.2020, and an R² of 0.8566 on California housing against -0.0002.

Two findings from the baseline shaped the design of the loop. On Titanic with 60 seconds, dropping the `Ticket` and `Cabin` columns raised the test F1 from 0.673 to 0.709, but a paired bootstrap over the 179 test rows gave a 95% interval of -0.042 to +0.117 for the difference, so one split could not tell us whether the change had helped [@efron1993bootstrap]. A fair comparison needs many folds and a proper test, and that became the Judge. FLAML's trial history also showed that 81% of the rise in its validation F1 came in the first 0.8 seconds of a 60-second search, yet the search ran until the budget was used up [@wang2021flaml]. Part of a budget can therefore go to changes in the data and the settings, an idea that became the Resource Manager.

## The first autonomous session

The first session ran on Titanic with 300 seconds, at most 8 experiments and seed 42. Table @tab:titanic shows each step: the change tried, its budget, the mean validation F1 over the 15 session folds and the Judge's numbers. The first setup left out `PassengerId` and `Name` as identifier-like and was searched for 105 seconds, 35% of the budget. The Critic found strong overfitting (training F1 0.996 against 0.706 on the folds) and high cardinality in `Cabin` and `Ticket`, where 72% of the validation rows had a ticket not seen in training. The Judge kept all three changes that followed: dropping `Cabin` as simpler and not worse, dropping `Ticket` as better (p(better) 0.010), and smaller trees as simpler. The session stopped after 206 seconds, with no change left to try.

::table tab:titanic | Steps of the first autonomous session on Titanic (300 s, seed 42) | widths=1.1,4.6,1.9,2.1,2.0,1.7,2.4 | align=ccccccc
| Step | Change tried | Search budget (s) | Validation F1 | Mean difference | p(better) | Verdict |
|---|---|---|---|---|---|---|
| 0 | First setup | 105 | 0.7056 | | | Start |
| 1 | drop_columns: Cabin | 25 | 0.7143 | +0.0087 | 0.284 | keep_simpler |
| 2 | drop_columns: Ticket | 23 | 0.7649 | +0.0506 | 0.010 | keep |
| 3 | regularised_search_space | 23 | 0.7628 | -0.0022 | 0.561 | keep_simpler |
::end

On the test rows, which were used only once at the end, the final model beat the first one on all the main metrics: an F1 of 0.7244 against 0.6838, an accuracy of 0.8045 against 0.7933, a ROC AUC of 0.8123 against 0.7808 and a log loss of 0.5261 against 0.6411. The loop had found on its own the change that our manual analysis of the baseline pointed to, and it confirmed that change on the 15 folds.

## Reliability of the Critic and the Judge

We measured the Critic on the 135 synthetic datasets of the reporting seeds, which the tuning of its thresholds never saw. Precision is the share of its alarms that were real problems, and recall the share of real problems it found. All six data-level kinds of diagnosis (class imbalance, constant, high-cardinality and identifier columns, empty cells and redundant columns) had a precision of 1.000, so none of their alarms was false. The Critic found every planted case of class imbalance, high cardinality, constant columns and empty cells, and it found overfitting, the kind most affected by noise, with a precision of 0.879 and a recall of 0.906 [@hastie2009esl].

We measured the Judge on 84 pairs of setups with known true effects. With its chosen settings, 3 repeats of 5-fold cross-validation and alpha 0.20, it kept none of the 18 harmful changes of the reporting seeds and found 10 of the 12 truly helpful ones [@nadeau2003inference]. This matters most for an autonomous loop, which tries many changes and must not accept the harmful ones.

## First comparison on the sample datasets

Before the benchmark, we ran a session with the rule order and FLAML alone with the same 300 seconds on the four sample datasets and three seeds. Over these 12 runs the session matched FLAML alone on average (a mean difference of +0.0004 in test score) while using about two thirds of the time. On Titanic, where the Critic found real data problems that FLAML cannot fix by itself, the session was better by 0.021 F1 on average over the three seeds. On penguins both reached the same test score in every run, and the session needed 132 seconds against 303 for FLAML alone.

## The benchmark

The benchmark completed all 390 runs of its grid in about 16 hours on the laptop, with no failed and no excluded run. The memory read by the full system held 95 records from the 30 development sessions.

## Test scores at the 300-second budget

Table @tab:scores300 gives the mean test score of each system over the 5 seeds at the 300-second budget. The scores are F1 for the classification datasets (macro F1 for the multiclass ones) and R² for energy_efficiency and solar_flare, and the best value in each row is in bold. At 300 seconds the full system was level with FLAML alone: the median of its differences over the six datasets was -0.0010, with a 95% bootstrap interval from -0.0051 to +0.0205 that contains zero [@efron1993bootstrap]. It ranked second of the five systems (Table @tab:ranks), scored higher than AutoGluon on 4 of the 6 datasets (median difference +0.0025) and higher than the LightGBM baseline on 5 of the 6 (median difference +0.0055). On the imbalanced pc4 dataset it gave the best score of all systems, an F1 of 0.6218 against 0.5897 for FLAML alone.

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

## Improvement with a larger budget

Figure @fig:scores shows the scores of the five systems at both budgets. Between 120 and 300 seconds the score of the full system rose on 5 of the 6 datasets, more often than for any other system, and it stayed level on energy_efficiency (0.9989 and 0.9988). FLAML alone rose on 2. On pc4 the full system went from 0.5840 to 0.6218 F1 and the rule-order session from 0.5483 to 0.6194, and on solar_flare the full system went from 0.1023 to 0.1132 R². The median gap to FLAML alone closed from 0.0137 at 120 seconds to 0.0010 at 300 seconds.

::figure fig:scores | figures/fig_scores.png | 15.5 | Mean test score of the five systems at 120 s and 300 s on the six evaluation datasets

Table @tab:ranks gives the average rank of each system over the six datasets at both budgets. The two AutoPilot-ML systems were the only ones whose rank improved with the larger budget. The full system moved from 3.58, fourth of five, to 2.50, second of five, and the rule-order session from 3.75 to 3.00.

::table tab:ranks | Average rank of the five systems over the six datasets at each budget (1 is best) | widths=5.2,2.6,2.6 | align=jcc
| System | 120 s | 300 s |
|---|---|---|
| AutoPilot-ML full | 3.58 | 2.50 |
| AutoPilot-ML rules | 3.75 | 3.00 |
| FLAML alone | 1.50 | 2.17 |
| AutoGluon | 3.33 | 3.67 |
| Manual LightGBM | 2.83 | 3.67 |
::end

This fits the design of the loop: the first search receives 35% of the budget and every change is then checked on 15 folds, so a longer budget leaves more room for the changes that the Critic proposes, while a single FLAML search makes most of its progress in its first seconds [@wang2021flaml].

## Efficiency

Figure @fig:time shows the mean wall time per run at the 300-second budget. FLAML alone uses the whole budget by design, while a session can stop early once it has no promising change left to try. The full system needed 212.3 seconds per run on average against 300.9 seconds for FLAML alone, 29% less time for the same level of score. The saving held on every dataset, from 16% on MiceProtein to 64% on solar_flare. On solar_flare the sessions found no change to try after their first search, and the full system still scored above FLAML alone (R² 0.1132 against 0.1043). AutoGluon's medium_quality preset finished after 98.5 seconds on average, with a lower score than the full system on 4 of the 6 datasets, and the LightGBM baseline trains a single model in under a second.

::figure fig:time | figures/fig_time.png | 15.5 | Mean wall time per run of the five systems at the 300 s budget

## What the sessions changed

At 300 seconds, a session of the full system ran 4.5 experiments on average, from one on solar_flare to seven on pc4. Across all session runs of the benchmark, the Judge kept 399 of the 738 changes that were tried after the first experiment. The most frequent change was dropping one of two strongly correlated columns, and most of the kept changes made the setup simpler, with fewer columns or learner families, without a measurable loss in validation score. Word features were kept on cylinder-bands and SMOTE on the imbalanced pc4 dataset [@chawla2002smote], and on these two datasets the changes raised the test score the most. At 300 seconds the mean test score of the full system rose from 0.6038 for its first model to 0.6218 for its final model on pc4, where the kept changes were column drops, and from 0.8472 to 0.8529 on cylinder-bands. On cylinder-bands, which has 14 text columns, the sessions kept five kinds of change: column drops, fewer learner families, smaller trees, grouped rare categories and word features.

The largest gain came in the run on pc4 with seed 5. The Critic found strong overfitting (training F1 0.964 against 0.545 on the folds), class imbalance and several pairs of strongly correlated code-metric columns. The session dropped two of these columns, both kept as simpler and not worse, and the Judge refused three further drops that lowered the validation F1 by 0.043 to 0.065. The final model reached a test F1 of 0.6875, against 0.5672 for the first model and for FLAML alone with the same seed.

## Discussion

The results show that AutoPilot-ML does what it was designed to do. It runs the whole experimentation loop by itself, offline on a laptop, and bases every decision on measured evidence that is saved with the run. Its two decision components were reliable on data with known answers: the Critic's data-level diagnoses raised no false alarm, and the Judge accepted none of 18 harmful changes.

Against strong systems the loop held its own and made good use of a larger budget. At 5 minutes it was level with FLAML alone within the statistical intervals while using 29% less time, ranked second of five systems and scored higher than AutoGluon and the LightGBM baseline on most datasets. FLAML is a mature engine from Microsoft Research, so matching it while also adding diagnoses, statistical decisions and a full record of every step is a strong outcome [@wang2021flaml].

For the research question of chapter 3, the results at the 5-minute budget show that the layer made conventional AutoML more efficient, reaching the score level of FLAML alone in 29% less time, and more effective where the data had problems that a search alone cannot fix. On Titanic it found and dropped the misleading `Ticket` column and raised the test F1 from 0.6838 to 0.7244, and on the imbalanced pc4 dataset it gave the best score of all systems. In both cases the reason behind the result can be read from the recorded diagnoses and verdicts, which a single AutoML search does not provide.

AutoPilot-ML also gives back more than the other systems. FLAML alone and AutoGluon return a model with its validation score, and the LightGBM baseline returns a single model [@wang2021flaml; @erickson2020autogluon]. A session returns its final model together with the diagnoses of every step, the changes it tried, the Judge's numbers behind each decision, the first and the final model side by side on the test rows, and a memory record of every verdict that later sessions can learn from.
