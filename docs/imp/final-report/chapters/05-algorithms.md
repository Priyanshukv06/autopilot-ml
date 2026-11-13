# Proposed algorithms

This chapter gives the three algorithms at the centre of AutoPilot-ML, in the notation of chapter 4. FLAML's own search, which runs inside every "search" step below, is described in section 2.4 and is not part of this work [@wang2021flaml].

## The Session loop

Algorithm @alg:session is the autonomous loop. Every decision in it uses only the Training_Part and its Session_Folds; the Test_Part is read once, in the last two lines.

::algorithm alg:session | The Session loop
Input: dataset D with target y; total time T, largest number of Experiments, CPU cores, selector and seed
Output: the final Setup, its model, and the test scores of the first and the final model
1 drop the rows without a target; split D into the Training_Part and the Test_Part (seeded, stratified)
1 make the 15 Session_Folds of the Training_Part and profile the Training_Part only
1 S ← the first Setup: the user's settings without identifier-like and constant columns
1 search S for 35% of T; A ← Assess(S); Δ ← Critic(S, A)
1 **repeat**
2 P ← the Proposals of the Strategy_Engine for Δ, without changes already tried, ordered by the selector
2 **if** the Resource_Manager says stop (user, error, time, Experiments, P empty, three discards) **then** leave the loop
2 b ← the next search budget of Equation (@eq:budget)
2 C ← the first Proposal of P applied to a copy of S
2 search C for b seconds, starting from the best configurations found for S
2 v ← Judge(Assess(C), A) by Algorithm @alg:judge; append one record to the Experiment_Memory
2 **if** v is keep **or** keep_simpler **then** S ← C; A ← Assess(C); Δ ← Critic(S, A)
1 **until** stopped
1 score the first and the final model once on the Test_Part
1 **return** S, its model and the two test scores
::end

## The Judge

Algorithm @alg:judge decides whether a candidate replaces the Accepted_Setup. It needs only the two Assessments, because both were made on the same folds.

::algorithm alg:judge | The Judge
Input: Assessments of the Accepted_Setup and the candidate on the same J folds; alpha α = 0.20; margin 0.005; whether the candidate is simpler
Output: a Verdict: keep, keep_simpler or discard
1 **for** each fold j **do** d(j) ← candidate score − accepted score (negated for error metrics)
1 **if** the standard deviation of d is at most 10^{−12} **then**
2 **if** mean(d) > 10^{−12} **then return** keep
2 **if** simpler **and** mean(d) ≥ −0.005 **then return** keep_simpler
2 **return** discard
1 t ← Equation (@eq:ttest); p(better) ← P(T > t); p(worse) ← P(T < t), with T a Student t with J − 1 degrees of freedom
1 **if** mean(d) > 0 **and** p(better) ≤ α **then return** keep
1 **if** simpler **and** mean(d) ≥ −0.005 **and** p(worse) > α **then return** keep_simpler
1 **return** discard
::end

## The learned order of the changes

Algorithm @alg:select reorders the Proposals of the rule table with what the memory says about similar datasets. Arms with too little evidence keep their rule-order place, so the learned choice can only move changes it has seen work or fail on at least two other datasets [@thompson1933; @auer2002ucb].

::algorithm alg:select | Learned ordering of the Proposals
Input: Proposals P in rule order; the Experiment_Memory M; the current dataset x; selector thompson or ucb1; the Session's random generator
Output: the Proposals in their new order
1 N ← the 5 nearest earlier datasets of the same task in M, never x itself
1 **for** each Proposal q in P **do**
2 a ← (the Strategy of q, the kind of its most severe Diagnosis)
2 K, D, G, C, n ← the statistics of arm a from the records of N, each weighted by 1/(1 + distance)
2 **if** n < 3 **or** all these records come from one dataset **then** mark q as fixed
2 **else** score(q) ← Equation (@eq:thompson) for thompson, or Equation (@eq:ucb) for ucb1
1 keep every fixed Proposal at its rule-order place
1 fill the other places with the scored Proposals, highest score first
1 **return** the new order
::end
