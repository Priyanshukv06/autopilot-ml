# Proposed algorithms

This chapter presents the three main algorithms of AutoPilot-ML, written in the notation of chapter 4. FLAML's own search, which runs inside every "search" step below, is described in section 2.4 and is not part of our work [@wang2021flaml].

## The session loop

Algorithm @alg:session is the autonomous loop. Every decision in it uses only the training part and its session folds, and the test part is read just once, in the last two lines.

::algorithm alg:session | The session loop
Input: dataset D with target y; time T; limits on experiments and CPU cores; selector; seed
Output: the final setup, its model, and the test scores of the first and the final model
1 drop rows without a target; split D into training and test parts (seeded, stratified)
1 make the 15 session folds of the training part and profile the training part only
1 S ← the first setup: the user's settings without identifier-like and constant columns
1 search S for 35% of T; A ← Assess(S); Δ ← Critic(S, A)
1 **repeat**
2 P ← the Strategy Engine's proposals for Δ that were not tried yet, ordered by the selector
2 **if** the Resource Manager says stop (section 4.8) **then** leave the loop
2 b ← the next search budget of Equation (@eq:budget)
2 C ← the first proposal of P applied to a copy of S
2 search C for b seconds, starting from the best configurations found for S
2 v ← Judge(Assess(C), A) by Algorithm @alg:judge; append one record to the Experiment Memory
2 **if** v is keep **or** keep_simpler **then** S ← C; A ← Assess(C); Δ ← Critic(S, A)
1 **until** stopped
1 score the first and the final model once on the test part
1 **return** S, its model and the two test scores
::end

## The Judge

Algorithm @alg:judge decides from the two assessments whether a candidate replaces the accepted setup.

::algorithm alg:judge | The Judge
Input: assessments of both setups on the same J folds; α = 0.20; whether simpler
Output: a verdict: keep, keep_simpler or discard
1 **for** each fold j **do** d(j) ← candidate score − accepted score (negated for error metrics)
1 **if** the standard deviation of d is at most 10^{−12} **then**
2 **if** mean(d) > 10^{−12} **then return** keep
2 **if** simpler **and** mean(d) ≥ −0.005 **then return** keep_simpler
2 **return** discard
1 t ← Equation (@eq:ttest); p(better) ← P(T > t) and p(worse) ← P(T < t) for T ~ Student t(J − 1)
1 **if** mean(d) > 0 **and** p(better) ≤ α **then return** keep
1 **if** simpler **and** mean(d) ≥ −0.005 **and** p(worse) > α **then return** keep_simpler
1 **return** discard
::end

## The learned order of the changes

Algorithm @alg:select reorders the proposals of the rule table using what the memory says about similar datasets. Arms with too little evidence keep their place in the rule order, so the learned choice can only move a change that it has seen succeed or fail on at least two other datasets [@thompson1933; @auer2002ucb].

::algorithm alg:select | Learned ordering of the proposals
Input: proposals P in rule order; Experiment Memory M; current dataset x; selector; random generator
Output: the proposals in their new order
1 N ← the 5 nearest earlier datasets of the same task in M, never x itself
1 **for** each proposal q in P **do**
2 a ← (the strategy of q, the kind of its most severe diagnosis)
2 K, D, G, C, n ← the statistics of arm a in the records of N, weighted by 1/(1 + distance)
2 **if** n < 3 **or** all these records come from one dataset **then** mark q as fixed
2 **else** score(q) ← Equation (@eq:thompson) for thompson, or Equation (@eq:ucb) for ucb1
1 keep every fixed proposal at its rule-order place
1 fill the other places with the scored proposals, highest score first
1 **return** the new order
::end
