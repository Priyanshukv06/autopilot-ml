# AutoPilot-ML: how to explain it to the professor

A plan for a 10-minute meeting.

- **Part A** is the explanation, about 6 minutes, with the report open.
- **Part B** is a live demo of about 4 minutes. Use it **only if he asks** to see the system running.
- **Part C** has short answers to likely questions.

The numbers come from the final report, and from one rehearsal of the demo on 14 November 2026. The rehearsal record is in `results/demo-rehearsal/session.json`.

Before you start, open the report PDF and keep these pages ready:

- Figure 1 on page 3 (the loop).
- Table 7 on page 31 (the Titanic session).
- Table 8 on page 33 (the benchmark scores).
- Figure 9 on page 34 (the time used).

---

## Part A: the explanation (about 6 minutes)

### 1. The problem (1 minute)

Say: "AutoML libraries such as FLAML find a good model and its settings within a time budget. But the work around the search is still done by hand: finding out why a model is weak, deciding what to change in the data, checking whether the change really helped, and deciding when to stop. Our project automates that work."

### 2. What we built (1 minute)

Show Figure 1. Say: "AutoPilot-ML is a layer on top of FLAML. FLAML is Microsoft's open-source library, and we use it unchanged for the search: that is the grey box. Everything around it was built by us."

Then name the five parts, one sentence each:

| Part | What it does |
|---|---|
| Critic | Looks at a model and names its weaknesses: 14 kinds, such as overfitting, class imbalance or a misleading column |
| Strategy Engine | Proposes a change that answers each weakness: 12 kinds, such as dropping a column, weighting the classes or using smaller trees |
| Judge | Keeps a change only if a statistical test on 15 folds shows it is better, or simpler and not worse |
| Resource Manager | Divides the time budget between the steps and decides when to stop |
| Experiment Memory | Records every change with its effect, so later runs learn which changes work on similar data |

Say: "It runs fully offline on a laptop, and every decision is saved with its numbers."

### 3. One workflow, step by step, on Titanic (3 minutes)

Show Table 7. This is the first real session: the Titanic passenger data, target `Survived`, a budget of 5 minutes, seed 42.

1. **Split.** 20% of the rows, the test part, are locked away until the end. The other 80%, the training part, is cut into 15 folds (3 repeats of 5-fold cross-validation). Every decision uses only these folds.
2. **First setup.** The system leaves out `PassengerId` and `Name` because they look like identifiers. FLAML then searches for 105 seconds, 35% of the budget.
3. **Diagnose.** The model scores F1 0.996 on its own training rows but only 0.706 on the folds. The Critic reports strong overfitting, plus too many different values in `Ticket` and `Cabin`: 72% of the validation rows have a ticket number the model has never seen.
4. **Propose and judge.** The Strategy Engine proposes three changes, one after another. The Judge compares each with the best setup so far, on the same 15 folds:
   - Drop `Cabin`: +0.0087. Not proven better, but simpler, so it is kept as **keep_simpler**.
   - Drop `Ticket`: +0.0506 with p(better) 0.010. A proven gain, so **keep**.
   - Smaller trees: −0.0022. Simpler and not worse, so **keep_simpler**.
5. **Stop.** No change is left to try, so the session stops after 206 of its 300 seconds.
6. **Final test.** Only now is the locked test part used. The test F1 rose from 0.6838 for the first model to 0.7244 for the final model.

Say: "It found by itself what a data scientist would do here, dropping the `Ticket` column. It proved the change with a statistical test, and it recorded why."

### 4. Results (1 minute)

Show Table 8 and Figure 9. Say:

- "We first tested the Critic and the Judge on synthetic data where the right answer is known. The data-level diagnoses gave no false alarm, and the Judge accepted none of 18 harmful changes."
- "Then we ran a benchmark: 390 runs on 6 public OpenML datasets, with 5 seeds and budgets of 2 and 5 minutes. We compared against FLAML alone, AutoGluon and a LightGBM baseline."
- "At 5 minutes it was level with FLAML within the statistical intervals, while using 29% less time. It ranked second of the five systems. It scored higher than AutoGluon on 4 of the 6 datasets and higher than the LightGBM baseline on 5 of the 6, and it gave the best score of all systems on the imbalanced pc4 dataset."
- "It also improves with more time: from 2 to 5 minutes it moved from fourth to second place."

### 5. Close (15 seconds)

Say: "So the project delivers an autonomous, offline and tested experimentation loop. It has 586 automated tests, a dashboard, a command line and a web API, and every decision is explained. The next steps are larger benchmarks and more kinds of diagnosis."

---

## Part B: the demo (only if he asks, about 4 minutes)

### Before the meeting

1. Plug the laptop into power and turn sleep off. A search that is paused by sleep gives worse results.
2. Start the dashboard and leave the browser tab open:

   ```
   conda activate automl
   cd C:\Users\priya\Desktop\project\AutoML\autopilot-ml
   streamlit run app/main.py --server.address localhost
   ```

   It opens at http://localhost:8501.
3. Do one practice run with exactly the settings below. It takes about 2.5 minutes.

### The clicks

1. In the sidebar, set **Mode** to **Autonomous run**.
2. **1 · Dataset**: choose **Sample dataset**, then **Titanic passenger survival**. In the **Data quality** tab, point at `Ticket`: it is flagged for having very many different values (681 in 891 rows).
3. **2 · Target**: choose `Survived`. It is detected as binary classification, and the identifier-like columns `PassengerId` and `Name` are already selected to be left out.
4. **3 · Session settings**: set **Total time** to 2.5 minutes, **Maximum Experiments** to 6 and **Random seed** to 42. Keep the metric at F1, the default, and leave the advanced settings as they are.
5. **4 · Autonomous run**: click **Start autonomous run**. While it runs, explain the timeline one row at a time.

### What will happen

This is what the rehearsal did with these settings:

| Step | What the session did | Verdict |
|---|---|---|
| 0 | First setup without `PassengerId` and `Name`, searched for 52 s. The Critic found overfitting (training F1 0.963 against 0.696 on the folds), too many values in `Ticket` and empty cells in `Cabin` | Start |
| 1 | Drop `Cabin`: +0.0082, p(better) 0.380 | keep_simpler |
| 2 | Drop `Ticket`: +0.0586, p(better) 0.008 | keep |
| 3 | Smaller trees: −0.0001 | keep_simpler |
| 4 | Word features from `Name`: +0.0013, p(better) 0.467 | discard |

After step 4 the time was used up (141 s). On the locked test part, the F1 rose from 0.6949 to 0.7143 and the ROC AUC from 0.8026 to 0.8276.

The exact numbers can differ a little on the day, because FLAML's search is limited by time. The pattern stays the same: dropping `Ticket` is the change that really helps.

What to say at each verdict:

- **keep**: "Proven better on the 15 folds."
- **keep_simpler**: "Not proven better, but simpler and not worse, so the simpler setup is preferred."
- **discard**: "The word features from `Name` showed no real gain, so the Judge refused them. The system does not accept a change that only looks good."

### When it finishes

1. Show the first and the final model side by side. They are scored on the test part, which no decision used.
2. Click **Explain the final model**. The SHAP chart splits each prediction into one contribution per column. In the rehearsal, the final model relied most on `Sex` and then `Pclass`: the test score dropped the most when those columns were shuffled.
3. Scroll to **5 · Experiment memory for this dataset**: there is one record for every verdict, with its effect and its cost. Say: "This is what later sessions learn from."

### If something goes wrong

- If the run is slow, press **Stop**. The session ends after its current step and still scores the test part.
- If nothing works, show Table 7 of the report and the recorded session `experiments/SES-001/session.json`. It is the same workflow, already finished.

---

## Part C: short answers to likely questions

- **Did you build FLAML?** No. FLAML is Microsoft's open-source library, under the MIT licence. We use it unchanged, and section 2.4 of the report explains how it searches. Our work is everything around it: the Critic, the Strategy Engine, the Judge, the Resource Manager, the memory, the dashboard and the whole evaluation.
- **How is this different from plain AutoML?** AutoML only searches models on the data it is given. Ours diagnoses the data and the model, changes them, proves each change with a statistical test and records why.
- **Why 15 folds and a corrected t-test?** One split is too noisy to tell whether a change helped. The 15 folds share most of their rows, so a normal t-test would be too confident. The correction by Nadeau and Bengio (Equation 5) accounts for that overlap.
- **Why is the test part locked?** If decisions used it, the final score would look better than it really is. All decisions use the folds; the test part is used once, at the end.
- **Why not let an AI chatbot decide?** It would need the internet, and its decisions are hard to check and repeat. Our decisions are rules and statistics that run offline. A small local model only writes an optional summary, which no decision reads.
- **Why is it behind FLAML at 2 minutes?** The first search gets only 35% of the budget, and every change is checked on 15 folds. At 5 minutes it catches up and still uses 29% less time. Giving the first search more of a short budget is the first item of our future scope.
- **How do you know it works?** All 586 automated tests pass. The Critic and the Judge were checked on data with known answers, and the benchmark is repeatable, with fixed datasets, fixed seeds and a frozen memory.
