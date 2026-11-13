# Conclusion and future scope

## Conclusions

This project set out to build an autonomous experimentation layer on top of conventional AutoML and to measure whether it improves on a single AutoML search. The main conclusions are the following.

1. A complete autonomous layer was built around FLAML, which it uses unchanged [@wang2021flaml]. It diagnoses each Experiment with 14 kinds of Diagnosis, answers them with 12 Strategies, accepts a change only through a corrected resampled t-test on 15 fixed folds, divides its budget with a Resource_Manager and learns the order of its changes from an Experiment_Memory with Thompson sampling or UCB1. It runs unattended and offline on a laptop, through a dashboard, a command line, a local web API or a Docker image, and it is covered by 586 automated tests.
2. Its two decision components work as designed on data with known answers. The Critic's data-level Diagnoses gave no false alarm, overfitting was found with a precision of 0.879 and a recall of 0.906, and with its chosen settings the Judge kept none of 18 harmful changes and 10 of 12 helpful ones.
3. At the 300-second budget the full system matched a single FLAML search within the bootstrap intervals while using 29% less time, and ranked second of five systems, ahead of AutoGluon's medium_quality preset and of LightGBM's default settings. At 120 seconds FLAML alone was ahead on all six datasets (median difference 0.0137). By the criteria fixed before the benchmark, the hypothesis that the loop reaches the target sooner is contradicted, and the hypotheses about the memory, the strategy choice, the Critic and the Resource_Manager are not shown on six datasets.
4. The cheap Judge's Verdicts were not reproduced by a strict comparison that searched again (Cohen's kappa 0.048 on 20 decisions), and only significant gains, not changes that were merely simpler, raised test scores in the development Sessions. A single Verdict is evidence from fixed refits, not a proven gain.
5. The loop adds what an AutoML search alone does not give: explained, recorded and repeatable decisions, and the detection and repair of data problems that a search cannot fix by itself. On such data it also reached higher scores than FLAML alone: on Titanic by 0.021 F1 on average over three seeds, and on pc4 at 300 seconds (0.6218 against 0.5897 F1).

## Lessons learned

The most useful practice was to fix the method before looking at results: the dataset selection rule, the success criteria and the analysis were all written down before the benchmark ran, so the negative result could not be explained away afterwards [@demsar2006]. Measuring each component on data with known answers before using it, and reading the saved records after each run, found real flaws: rounding noise once counted as a gain, steps that changed nothing counted as successes in the memory, and evidence from a single dataset steered the learned order. Each was fixed and tested. Long runs on a laptop also need care: the first benchmark run was lost to standby on battery, and the harness now waits for mains power.

## Limitations

The benchmark used six evaluation datasets, which gives little statistical power, and only two short budgets on one laptop [@gijsbers2024amlb]. The memory was built from ten development datasets, so most arms had too little evidence to move a Strategy. AutoGluon ran with a preset that stopped early, the strict comparison of the Judge used shorter searches than the Sessions, and the optional Strategies were tried only a few times in the benchmark. The rule table can only propose the changes written into it.

## Future scope

The measurements point to the following work, ordered by how directly the data supports it.

- Give the first search more of a short budget, or skip the loop below some budget, since FLAML gains most in its first seconds [@wang2021flaml].
- Use the time a Session leaves unused, about 30% of a 300-second budget, for a last and longer search of the Accepted_Setup.
- After a discard, try a different kind of change before another of the same kind; on pc4 the rule order spent its patience on similar column drops and never reached class weights.
- Add stronger answers for imbalanced classes, such as tuning the decision threshold on the folds [@he2009imbalance].
- Repeat the Judge agreement study with strict searches as large as the Sessions' searches.
- Run the benchmark on more datasets, so that the Wilcoxon tests can reach significance after Holm's correction, and with a stronger AutoGluon preset [@holm1979; @erickson2020autogluon].
- Let a language model propose new changes while the Judge stays the gate that accepts or rejects them, keeping every decision measured [@hollmann2023caafe].
- Add feature engineering across columns, a database and a login for the web API, and support for time series and text data.
