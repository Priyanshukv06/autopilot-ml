# Result discussion and analysis

This chapter reports every measurement of the project with its settings. All numbers are real measurements on the laptop of Table @tab:machine; none is estimated or copied from elsewhere. FLAML's search is limited by time, so a repeated run can differ slightly. Table @tab:measurements lists the measurements in the order they were made, with the place where their raw files are kept.

::table tab:measurements | The measurements reported in this chapter | widths=4.6,2.6,7.2 | align=jcj
| Measurement | Date (2026) | Raw files |
|---|---|---|
| Baseline on the four sample datasets | 7 to 12 October | experiments/EXP-001, EXP-002; numbers in the report notes |
| First autonomous Session on Titanic | 26 October | experiments/SES-001 and EXP-003 to EXP-006 |
| Critic study, 270 synthetic datasets | 26 October | results/studies/critic_study_*.csv |
| Judge study, 84 pairs of Setups | 26 October | results/studies/judge_study_*.csv |
| The loop against FLAML alone, four samples | 27 October | results/phase2/*.csv and SES-002 to SES-013 |
| Judge agreement study, 20 pairs | 31 October | results/studies/judge_agreement_*.csv |
| Full benchmark, 390 runs, final code | 12 and 13 November | results/benchmark-v2/ and experiments/benchmark-v2/ |
::end

A first benchmark run started on 28 October was cut short when the laptop went into standby on battery; only 23 of its runs were usable, so it is not used for any claim here. The harness was then changed to wait while on battery, and the whole grid was run again with the final code.

## The baseline

Table @tab:phase1 shows one FLAML search of 10 seconds on each sample dataset (seed 42), scored on the held-out test rows next to a trivial baseline on the same rows [@wang2021flaml]. The baseline pipeline is far ahead of the trivial baseline on every dataset.

::table tab:phase1 | The baseline on the four sample datasets (10 s, seed 42) | widths=3.2,2.4,2.8,2.4,2.4,2.6 | align=jccccc
| Dataset | Task | Training / test rows | Best model | Test score | Trivial baseline |
|---|---|---|---|---|---|
| Titanic | binary | 712 / 179 | XGBoost | F1 0.6935 | F1 0.0000 |
| Breast cancer | binary | 455 / 114 | LightGBM | F1 0.9512 | F1 0.0000 |
| Penguins | multiclass | 275 / 69 | LightGBM | macro F1 0.9829 | macro F1 0.2020 |
| California housing | regression | 16,512 / 4,128 | XGBoost | R² 0.8566 | R² -0.0002 |
::end

Two observations from this phase motivated the design of the loop. First, on Titanic with 60 seconds, dropping the `Ticket` and `Cabin` columns raised the test F1 from 0.673 to 0.709, but a paired bootstrap over the 179 test rows gave a 95% interval of -0.042 to +0.117 for the difference, so one split could not show that the change helped [@efron1993bootstrap]. A fair comparison needs many folds and a proper test, which became the Judge. Second, FLAML's trial history showed that 81% of the rise in its validation F1 came in the first 0.8 seconds and 90% by 9.6 seconds of a 60-second search, while the search went on until the budget ran out [@wang2021flaml]. That suggested that part of a budget could be spent on other changes, which became the Resource_Manager.

## The first autonomous Session

The first Session ran on Titanic on 26 October with 300 seconds, at most 8 Experiments and seed 42. Its first Setup left out `PassengerId` and `Name` as identifier-like. The Critic found strong overfitting (training F1 0.996 against 0.706 on the folds) and high cardinality in `Ticket` (72% of validation rows had an unseen ticket). The Session dropped `Cabin` (kept as simpler, mean difference +0.0087, p(better) 0.284), dropped `Ticket` (kept as better, +0.0506, p(better) 0.010) and tried smaller trees (kept as simpler, -0.0022), then stopped after 206 seconds with no change left. On the test rows, used once at the end, the final model scored F1 0.7244 against 0.6838 for the first model. The loop had found by itself the change that the manual analysis of the baseline pointed to.

## Component studies

### The Critic

Table @tab:critic shows how often the Critic was right on the 135 synthetic datasets of the reporting seeds, which the tuning never saw. Precision is the share of the Critic's alarms that were real, and recall the share of real problems it found. The tuning on the other seeds changed three thresholds, as described in chapter 4 [@hastie2009esl].

::table tab:critic | Precision and recall of the Critic on the reporting seeds | widths=4,1.8,2.2,2.2,2.2,2.2 | align=jccccc
| Diagnosis | True cases | Precision, start | Recall, start | Precision, tuned | Recall, tuned |
|---|---|---|---|---|---|
| class_imbalance | 15 | 1.000 | 1.000 | 1.000 | 1.000 |
| constant_column | 15 | 1.000 | 0.667 | 1.000 | 1.000 |
| high_cardinality | 15 | 1.000 | 1.000 | 1.000 | 1.000 |
| identifier_column | 15 | 1.000 | 0.667 | 1.000 | 0.667 |
| missing_values | 15 | 1.000 | 0.667 | 1.000 | 1.000 |
| overfitting | 32 | 0.711 | 1.000 | 0.879 | 0.906 |
| redundant_features | 15 | 1.000 | 0.933 | 1.000 | 0.933 |
| underfitting | 5 | 1.000 | 0.600 | 1.000 | 0.600 |
| unreliable_validation | 4 | 0.273 | 0.750 | 0.273 | 0.750 |
::end

The data-level Diagnoses gave no false alarm. Overfitting, the kind most affected by noise, reached a precision of 0.879 and a recall of 0.906 after tuning. The unreliable-validation Diagnosis is weak: only 3 of its 11 alarms were right. It was kept because its answer, a smoother search metric, is cheap and the Judge still decides whether it helps.

### The Judge

Table @tab:judgestudy shows the Judge study on 84 pairs of Setups with known true effects. A false keep is an accepted change whose true effect was below -0.005, and power is the share of truly helpful changes (above +0.01) that were kept.

::table tab:judgestudy | False keeps and power of the Judge for some of its settings | widths=1.8,1.6,2.8,2.8,2.8,2.8 | align=cccccc
| Fold repeats | Alpha | False keeps, tuning (of 22) | Power, tuning (of 11) | False keeps, reporting (of 18) | Power, reporting (of 12) |
|---|---|---|---|---|---|
| 1 | 0.20 | 3 (0.136) | 9 (0.818) | 0 | 9 (0.750) |
| 3 | 0.05 | 2 (0.091) | 7 (0.636) | 0 | 7 (0.583) |
| 3 | 0.10 | 2 (0.091) | 8 (0.727) | 0 | 8 (0.667) |
| 3 | 0.20 | 2 (0.091) | 9 (0.818) | 0 | 10 (0.833) |
| 5 | 0.20 | 2 (0.091) | 10 (0.909) | 0 | 10 (0.833) |
::end

The rule fixed before the run picked 5 repeats and alpha 0.20. Alpha 0.20 was adopted. The default stayed at 3 repeats, a recorded departure from the rule: 3 and 5 repeats accepted the same helpful changes on both seed sets, and with the corrected t-test 25 folds give a standard error only about 4% smaller than 15 folds, while every Assessment would cost 25 refits instead of 15 [@nadeau2003inference]. With 3 repeats and alpha 0.20 the Judge kept none of the 18 harmful changes and 10 of the 12 helpful ones on the reporting seeds.

## The loop against FLAML alone on the samples

Before the benchmark, a Session with the rule order and FLAML alone were run with the same 300 seconds on the four sample datasets and three seeds. Table @tab:phase2 shows the difference of their test scores.

::table tab:phase2 | Test score of the Session minus FLAML alone (300 s, rule order) | widths=3.4,1.8,1.8,1.8,1.8,3.4 | align=jccccc
| Dataset | Seed 1 | Seed 2 | Seed 3 | Mean | Time used, Session / FLAML (s) |
|---|---|---|---|---|---|
| Titanic | 0.0000 | +0.0336 | +0.0288 | +0.0208 | 197 / 304 |
| Breast cancer | +0.0129 | -0.0256 | -0.0347 | -0.0158 | 293 / 307 |
| Penguins | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 132 / 303 |
| California housing | +0.0001 | -0.0039 | -0.0069 | -0.0035 | 201 / 312 |
::end

Over the 12 pairs the Session was better 4 times, equal 4 times and worse 4 times (mean +0.0004), using about two thirds of the time. It helped on Titanic, where the Critic found real data problems that FLAML cannot fix by itself, and cost a little where its changes were neutral (breast cancer) or the 15-fold Assessments were slow (California housing, 13,200 training rows).

## The full benchmark

The 390 runs of the grid were made with the final code (commit af8f14d) from 12 November 2026, 21:46, to 13 November, 14:02: 16.25 hours of run time on mains power, with no failed and no excluded run. The frozen memory read by the full system was built just before by 30 Sessions on the development datasets: 95 records (13 keep, 45 keep_simpler, 37 discard), with 7 of its 21 arms holding enough evidence to move a Strategy (SHA-256 beginning d678080a).

### Test scores

Table @tab:scores300 gives the mean test score of each system over the 5 seeds at 300 seconds, and Figure @fig:scores shows the five main systems at both budgets. The scores are F1 for the classification datasets (macro F1 for the multiclass ones) and R² for energy_efficiency and solar_flare. In the tables, Full is the full system (autopilot_full), Rules the Session with the rule order (autopilot_rules) and Manual the LightGBM defaults; the other names are those of Table @tab:systems.

::table tab:scores300 | Mean test score over 5 seeds at 300 s | widths=2.95,1.55,1.45,1.85,1.35,1.35,1.75,1.6,1.95 | align=jcccccccc
| Dataset | Manual | FLAML | AutoGluon | Rules | Full | Random order | No Critic | No Resource Manager |
|---|---|---|---|---|---|---|---|---|
| pc4 | 0.6125 | 0.5897 | 0.5507 | 0.6194 | **0.6218** | 0.6074 | 0.6035 | 0.5892 |
| cylinder-bands | 0.8485 | **0.8547** | 0.8375 | 0.8426 | 0.8529 | 0.8400 | 0.8438 | 0.8420 |
| segment | **0.9814** | 0.9805 | 0.9723 | 0.9766 | 0.9771 | 0.9784 | 0.9784 | 0.9766 |
| MiceProtein | 0.9855 | **0.9989** | 0.9947 | 0.9866 | 0.9921 | 0.9954 | 0.9956 | 0.9956 |
| energy_efficiency | 0.9972 | 0.9989 | 0.9987 | 0.9988 | 0.9988 | 0.9989 | **0.9990** | 0.9989 |
| solar_flare | 0.0457 | 0.1043 | **0.1268** | 0.1166 | 0.1132 | 0.1132 | 0.1134 | 0.1023 |
::end

::figure fig:scores | figures/fig_scores.png | 15.5 | Mean test score of the five main systems at 120 s and 300 s on the six evaluation datasets

At 120 seconds the full system was behind FLAML alone on every dataset: its means were 0.5840, 0.8454, 0.9737, 0.9837, 0.9989 and 0.1023 against FLAML's 0.6038, 0.8610, 0.9780, 0.9956, 0.9989 and 0.1192 in the order of Table @tab:scores300. At 300 seconds the two were close: the full system was better on pc4 (0.6218 against 0.5897) and solar_flare (0.1132 against 0.1043), and worse on MiceProtein (0.9921 against 0.9989) and segment (0.9771 against 0.9805). Figure @fig:diff shows these differences per dataset.

::figure fig:diff | figures/fig_diff.png | 15.5 | Difference of the mean test score, AutoPilot-ML full minus FLAML alone, per dataset and budget

### Pairwise comparisons

Table @tab:pairs gives the comparisons fixed before the run [@demsar2006]. The difference is the median over the 6 datasets of the per-dataset difference, with its 95% bootstrap interval [@efron1993bootstrap]; W, T and L count the datasets where the first system won, tied or lost; p is the two-sided Wilcoxon p-value and Holm the p-value after Holm's correction within the family [@wilcoxon1945; @holm1979].

::table tab:pairs | Pairwise comparisons over the 6 evaluation datasets | widths=1.4,4.6,2.2,3.4,1.6,1.2,1.4 | align=ccccccc
| Budget | Pair | Median difference | 95% interval | W / T / L | p | Holm |
|---|---|---|---|---|---|---|
| 120 s | Full against FLAML alone | -0.0137 | -0.0183 to -0.0022 | 0 / 0 / 6 | 0.031 | 0.156 |
| 120 s | Full against Manual | -0.0024 | -0.0181 to 0.0292 | 2 / 0 / 4 | 0.563 | 1.000 |
| 120 s | Full against AutoGluon | -0.0003 | -0.0177 to 0.0206 | 3 / 0 / 3 | 1.000 | 1.000 |
| 120 s | Full against Rules | 0.0001 | -0.0013 to 0.0182 | 3 / 1 / 2 | 0.625 | 1.000 |
| 120 s | Rules against FLAML alone | -0.0141 | -0.0363 to -0.0009 | 0 / 0 / 6 | 0.031 | 0.156 |
| 300 s | Full against FLAML alone | -0.0010 | -0.0051 to 0.0205 | 2 / 0 / 4 | 1.000 | 1.000 |
| 300 s | Full against Manual | 0.0055 | -0.0013 to 0.0384 | 5 / 0 / 1 | 0.094 | 0.469 |
| 300 s | Full against AutoGluon | 0.0025 | -0.0081 to 0.0432 | 4 / 0 / 2 | 0.438 | 1.000 |
| 300 s | Full against Rules | 0.0015 | -0.0017 to 0.0078 | 4 / 0 / 2 | 0.313 | 1.000 |
| 300 s | Rules against FLAML alone | -0.0020 | -0.0122 to 0.0210 | 2 / 0 / 4 | 1.000 | 1.000 |
| 300 s | Full against Random order | -0.0001 | -0.0023 to 0.0137 | 2 / 1 / 3 | 0.813 | 1.000 |
| 300 s | Full against No Critic | -0.0002 | -0.0024 to 0.0137 | 2 / 0 / 4 | 1.000 | 1.000 |
| 300 s | Full against No Resource_Manager | 0.0057 | -0.0018 to 0.0218 | 4 / 0 / 2 | 0.219 | 0.875 |
::end

The Friedman test of Equation (@eq:friedman) found no system that stands apart: for the main systems χ² = 7.97 with p = 0.093 at 120 seconds (Iman and Davenport's version p = 0.077) and p = 0.355 at 300 seconds, and for the ablations p = 0.434 [@friedman1937; @iman1980]. Figure @fig:cd shows the average ranks: at 120 seconds FLAML alone was first (1.50) and the two AutoPilot-ML systems last (3.58 and 3.75), and at 300 seconds FLAML alone was first (2.17) and the full system second (2.50). Every difference of ranks is below the Nemenyi critical difference of 2.49 from Equation (@eq:cd), so at every budget all systems fall in one group [@nemenyi1963].

::figure fig:cd | figures/fig_cd.png | 15.5 | Critical-difference diagrams of the average ranks (1 is best); a bar joins systems whose ranks do not differ by the critical difference

### Time to the target and use of the budget

Table @tab:ttt shows how quickly each main system reached the target, 99% of the best score any system reached for that dataset and seed, over the 30 pairs of dataset and seed. FLAML alone reached it by 120 seconds in 18 pairs and by 300 seconds in 22, the full system in 13 and 16. Counted pair by pair, the full system reached the target at an earlier budget than FLAML alone in 2 pairs, at the same budget in 18 and later in 10.

::table tab:ttt | Time-to-target over 30 (dataset, seed) pairs | widths=4.4,2.6,2.6,2.2,2.6 | align=jcccc
| System | Reached by 120 s | Reached by 300 s | Never | Median time |
|---|---|---|---|---|
| FLAML alone | 18 | 22 | 8 | 120 s |
| AutoGluon | 18 | 18 | 12 | 120 s |
| AutoPilot-ML rules | 13 | 18 | 12 | 300 s |
| AutoPilot-ML full | 13 | 16 | 14 | 300 s |
| Manual LightGBM | 13 | 13 | 17 | never |
::end

Figure @fig:time shows how the systems used the 300-second budget. FLAML alone always used all of it, while the Sessions stopped early: the full system used 212.3 seconds on average. Of the 150 Session runs at 300 seconds, 79 stopped because no change was left to try, 27 after three discards in a row, 39 for time and 5 at the Experiment limit. AutoGluon's medium_quality preset also stopped early, after 98.5 seconds on average.

::figure fig:time | figures/fig_time.png | 15.5 | Mean wall time per run at the 300 s budget, and why the Session systems stopped

### What the Sessions changed

Table @tab:strategyuse counts, over all Session runs of the grid, how often each Strategy was tried after the first Experiment and how often the Judge kept it. Dropping columns, mostly one of a pair of nearly equal columns, was by far the most frequent change, and most kept changes were kept as simpler and not worse rather than better. Of the optional Strategies, word features were kept 3 times out of 26 on cylinder-bands and SMOTE twice out of 4 on pc4, while target encoding was never kept [@chawla2002smote; @micci2001target].

::table tab:strategyuse | Strategies tried and kept in all Session runs of the benchmark | widths=5.2,2.4,2.4 | align=jcc
| Strategy | Tried | Kept |
|---|---|---|
| drop_columns | 472 | 279 |
| restrict_model_families | 104 | 86 |
| regularised_search_space | 40 | 16 |
| more_search_time | 26 | 3 |
| token_features | 26 | 3 |
| class_weight_balanced | 17 | 1 |
| group_rare_categories | 14 | 9 |
| frequency_encode | 11 | 0 |
| smoother_search_metric | 10 | 0 |
| missing_indicator | 7 | 0 |
| target_encode | 7 | 0 |
| smote | 4 | 2 |
::end

## Does the Judge decide like a strict comparison?

Table @tab:agreement compares the Judge's Verdicts on 20 real decisions with those of a strict comparison that searched both Setups again on 5 inner splits. They agreed on 8 of the 20 Verdicts (0.40, 95% bootstrap interval 0.20 to 0.60), and Cohen's kappa of Equation (@eq:kappa) was 0.048 (interval -0.23 to 0.34), which is little more than chance [@cohen1960kappa]. On accepted against discarded they agreed on 11 of 20 (kappa 0.043). Three of the Judge's four significant gains were judged worse by the strict comparison. The strict comparison cost 2.38 times as much in total, although its searches were capped at 25 trials.

::table tab:agreement | Verdicts of the Judge (rows) against the strict comparison (columns), 20 pairs | widths=4.4,2.6,2.6,2.6 | align=jccc
| Judge | keep | keep_simpler | discard |
|---|---|---|---|
| keep | 0 | 1 | 3 |
| keep_simpler | 2 | 5 | 1 |
| discard | 1 | 4 | 3 |
::end

The two methods do not answer the same question: the Judge compares the configurations a Session's search found, refitted on the same 15 folds, while the strict comparison measures what a new, shorter search finds on one split at a time, and its own differences were noisy. Across the 42 development Sessions, the Sessions whose final Setup held at least one keep Verdict did gain on the test part (+0.0162 on average, 8 of 11 better), while those with only keep_simpler steps changed by -0.0031 on average. So a single Verdict should be read as evidence from fixed refits, not as a proven gain.

## Outcome of the hypotheses

Table @tab:outcome applies the criteria of Table @tab:hypotheses, written before the benchmark ran, to the measured intervals.

::table tab:outcome | Outcome of the hypotheses by the criteria fixed in advance | widths=4.6,7.6,3.6 | align=cjc
| Hypothesis | Measured | Outcome |
|---|---|---|
| H1: reaches the target sooner than FLAML | FLAML earlier in 10 pairs, the full system in 2; median 120 s against 300 s | Contradicted |
| H2: worse without the memory | Interval -0.0017 to 0.0078 at 300 s | Not shown |
| H3: worse without strategy selection | Interval -0.0023 to 0.0137 at 300 s | Not shown |
| H4: not worse than FLAML at the same budget | Intervals -0.0183 to -0.0022 at 120 s and -0.0051 to 0.0205 at 300 s | Not shown |
| RQ3 and the Critic: worse without them | Intervals -0.0018 to 0.0218 and -0.0024 to 0.0137 | Not shown |
::end

H4 is not supported, because at 120 seconds the interval reaches below the margin of -0.005, and it is not contradicted, because the interval does not lie wholly below the margin. At 300 seconds the interval only just reaches below -0.005.

## Discussion

The benchmark shows where the loop stands against a strong engine. At 300 seconds it was level with FLAML alone within the intervals and ranked second of the five main systems, ahead of AutoGluon's medium_quality preset and of LightGBM's defaults, which it beat on 4 and 5 of the 6 datasets, while using 29% less wall time than FLAML (212.3 against 300.9 seconds). At 120 seconds FLAML alone was ahead on every dataset. A likely reason, which was not measured separately, is the split of the budget: the first search gets only 35% of the time and the 15-fold Assessments take part of the rest, while the trial histories show that FLAML gains most in its first seconds [@wang2021flaml]. FLAML is a mature engine from Microsoft Research; matching it at the longer budget while adding Diagnoses, statistical decisions and a full record of every step is the main outcome of this comparison.

The ablations did not show any part of the loop making a difference larger than the intervals over six datasets, and the run without the Critic had the best average rank of the ablations. Three findings explain much of this. Most kept changes were simpler but not better, and such changes did not raise test scores in the development Sessions. The Sessions often ran out of useful changes and left about 30% of a 300-second budget unused. And the memory was thin: with 10 development datasets, few arms had evidence from two or more datasets, so the learned order rarely differed from the rule order.

Where the data had a problem that FLAML cannot fix by itself, the loop did help: on Titanic it found and dropped the high-cardinality `Ticket` column, and on the imbalanced pc4 dataset it reached the best score of any system at 300 seconds (0.6218 F1, where FLAML alone reached 0.5897). Every decision it made can be checked from its recorded numbers, and its data-level Diagnoses gave no false alarm in the Critic study. So the loop adds what a single search cannot give: explained, recorded and repeatable decisions and the detection of data problems, and on data with such problems it also gave higher scores.

These results have clear limits. Six evaluation datasets give little statistical power: the smallest possible two-sided Wilcoxon p-value is 0.031, so no pair can pass Holm's correction at 0.05 [@demsar2006]. All runs used one laptop and two short budgets, time-limited searches vary with timing, and AutoGluon ran with a preset that stopped early, so it does not show what AutoGluon can do with its best settings [@erickson2020autogluon; @gijsbers2024amlb].
