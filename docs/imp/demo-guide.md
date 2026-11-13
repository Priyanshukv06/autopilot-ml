# Demo guide: showing AutoPilot-ML to the guide

A plan for a live demo of about 15 minutes, with the exact clicks and commands, what to say at each step, and what to do if something goes wrong. All numbers quoted here are the measured ones from the final report (`docs/imp/final-report/`).

## The day before

1. Plug the laptop into mains power and set Windows so that it does not sleep while plugged in. A time-limited search that is paused by sleep gives worse results.
2. Open a terminal and check that everything works:

   ```
   conda activate automl
   cd C:\Users\priya\Desktop\project\AutoML\autopilot-ml
   python -m pytest -q -m "not slow"
   ```

   The quick tests take a few minutes; all must pass. The full suite (`python -m pytest -q`, 586 tests) takes about six minutes.
3. Do one rehearsal of the whole demo below, with the same budgets, so you know how long each step takes on the day.
4. Optional parts need their programs running: the summary needs Ollama (`ollama serve`, model `qwen2.5:7b-instruct`), and the container needs Docker Desktop. If either is not running, skip that part.
5. Keep the final report PDF open in a viewer, at Figure 1 (the loop) and at Table 21 (the pairwise results).

## On the day: start the dashboard

```
conda activate automl
cd C:\Users\priya\Desktop\project\AutoML\autopilot-ml
streamlit run app/main.py --server.address localhost
```

Open http://localhost:8501 in the browser. The sidebar has the Mode switch and lists what is built.

## Part 1: the problem (1 minute, no clicks)

Say: "AutoML libraries such as FLAML search for a model within a time budget. They do not tell you why a model is weak, what to change next, or whether a change really helped. AutoPilot-ML runs that loop by itself: it diagnoses, changes, judges with a statistical test and records every decision. FLAML is Microsoft Research's library; we use it unchanged as the search engine. What we built is the layer around it." Show Figure 1 of the report.

## Part 2: one plain AutoML run, the baseline (3 minutes)

1. Mode: **Single experiment**.
2. Section 1, Data source: **Sample dataset**, then **Titanic passenger survival**. Point at the data-quality cards: missing values in Age and Cabin, `PassengerId` and `Name` flagged as identifier-like, `Ticket` with 681 different values.
3. Section 2: the target `Survived` is suggested and detected as binary classification. Leave the suggested exclusions.
4. Section 3: time budget **30 seconds**, metric F1, seed 42.
5. Click **Run Experiment** and wait about 35 seconds.
6. In the results, show: the test F1 next to the trivial baseline (F1 0), the gap between training and test scores (overfitting), the note that many test passengers have a ticket number never seen in training, and the permutation importance.

Say: "A person would now drop or re-encode `Ticket`, run again and compare fairly. The autonomous mode does exactly this by itself."

## Part 3: the autonomous Session (6 minutes)

1. Mode: **Autonomous run**. The dataset and target stay selected.
2. Section 3, Session settings: total time **3 minutes**, at most **6** Experiments, seed 42. Leave the advanced settings at their defaults (3 fold repeats, Judge alpha 0.20, the rule order).
3. Click **Start autonomous run**. The page redraws every second while the Session runs; you can talk over it.
4. While it runs, explain the timeline table, row by row:
   - The **Diagnoses** of the first Setup: overfitting and high cardinality for `Ticket` and `Cabin`, each with its evidence numbers.
   - The **change** the Strategy_Engine chose for each step, for example dropping `Ticket`.
   - The **Judge's numbers**: the mean difference over the 15 folds and p(better). "Keep" needs a significant gain; "keep_simpler" means simpler and not worse; "discard" otherwise.
   - Point out that every decision uses only the training part. The test part is scored once at the end.
5. When it ends, show the stop reason, and the first and final model side by side on the test part, under the label "Not used for decisions".
6. Click **Explain the final model** to show the SHAP values (on Titanic, Sex usually comes first), and pick one row to show how one prediction is split into contributions.
7. Optional, only if Ollama runs: click **Write a summary**. Point out the label: the text is generated, not checked, and no decision ever reads it.
8. Scroll to section 5, the Experiment_Memory records of this dataset: one record per Verdict, with its effect and cost.

The Session's exact results vary from run to run, because FLAML's search is limited by time. If a step is discarded, that is fine: explain that the Judge refused a change that was not shown to help.

## Part 4: the evaluation and the honest result (3 minutes)

Open the report at chapter 8 and say:

- "We fixed the datasets, the comparisons and the success criteria in writing before the benchmark ran: 390 runs on 6 OpenML datasets, 5 seeds, budgets of 2 and 5 minutes, against FLAML alone, AutoGluon and LightGBM's defaults."
- "With the same budget, one FLAML search was as good as our loop or better. At 2 minutes FLAML was ahead on all six datasets; at 5 minutes the two were level within the confidence intervals, and our full system ranked second of five." (Figure 12, Table 21, Figure 14)
- "So the hypothesis that the loop reaches the target sooner was contradicted, and the others were not shown. We report this as measured."
- "Where the data had a problem FLAML cannot fix by itself, the loop helped: on the imbalanced pc4 dataset it had the best score at 5 minutes, 0.6218 F1 against FLAML's 0.5897."
- "The components were checked on synthetic data with known answers: the Critic's data-level Diagnoses gave no false alarm, and the Judge accepted none of 18 harmful changes."

## Part 5: optional extras (2 minutes, pick what time allows)

- Command line: `python scripts/run_autopilot.py --dataset data/sample/titanic.csv --target Survived --seconds 120 --max-experiments 5 --seed 42` prints each step.
- Web API: in a second terminal, `python scripts/run_api.py`, then open http://127.0.0.1:8000/docs, the API's own page, and run `GET /sessions`.
- Docker: with Docker Desktop running, `docker run --rm -p 127.0.0.1:8501:8501 -p 127.0.0.1:8000:8000 autopilot-ml:0.1.0`, then open http://localhost:8501.
- MLflow: `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000 --workers 1`, then http://localhost:5000 shows each Session as a parent run with its Experiments.

## Questions you may be asked

- **Did you build FLAML?** No. FLAML is Microsoft Research's open-source library (Wang et al., MLSys 2021, MIT licence). We studied its paper and its source code, explain how it searches in section 2.4 of the report, and use it unchanged. Our work is the experimentation layer: the Critic, the Strategy_Engine, the Judge, the Resource_Manager, the memory and the learned choice, the dashboard, the API and the whole evaluation.
- **Why not just give FLAML all the time?** In this benchmark that was indeed as good or better. Our loop spends part of the budget on refits on 15 folds to decide fairly, and its first search gets only 35% of the time, so at short budgets it falls behind. Chapter 9 lists the fixes the data points to.
- **Why the corrected t-test?** The 15 folds share most of their training rows, so an ordinary paired t-test is far too confident. The Nadeau and Bengio correction adds a term for that overlap (Equation 7 of the report).
- **Why not use a large language model to decide?** It would need a hosted model, its decisions are hard to check and repeat, and we wanted every decision measurable and offline. The local model only writes a labelled summary.
- **What is new here?** Diagnosing, changing, statistically judging and recording each step, with a memory of changes and their measured effect, all offline and repeatable, and an honest benchmark of whether that helps.

## If something goes wrong

- The page shows an error: press the browser's refresh and select the sample again. A Session that failed is recorded and a new one can be started.
- A run is slow: press **Stop**; the Session ends after its current step and still scores the first and final model.
- Nothing works: show a finished Session from the records instead (`experiments/SES-001/session.json`, the first Session, described in section 8.2 of the report), and the benchmark figures in `results/benchmark-v2/figures/`.
