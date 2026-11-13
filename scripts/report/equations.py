"""The report's display equations, keyed by the ID used in the chapter files (::equation ID).

The builder numbers them in the order they appear in the text. Each entry is the content
of one equation, written with the helpers in omml.py. Word adds the spacing around
operators itself, so the operators are written without spaces.
"""

from omml import bar, frac, func, hat, nary, paren, sqrt, sub, subsup, sup, t, v

SUM = "\u2211"
MINUS = "\u2212"
PRIME = "\u2032"
GAP = ",\u2003\u2003"  # comma and two em spaces between two formulas on one line

EQUATIONS = {
    # FLAML's estimated cost for improvement of learner l (FLAML 2.7.0, automl.py and state.py).
    "eq:eci": [
        sub(t("ECI"), v("l")), t("="),
        func("max", paren(
            subsup(v("t"), v("l"), v("b")), t(MINUS), subsup(v("t"), v("l"), [v("b"), t(PRIME)]),
            t(", "),
            sub(v("t"), v("l")), t(MINUS), subsup(v("t"), v("l"), v("b")),
            t(", "),
            frac([t("2"), paren(sub(v("\u03b5"), v("l")), t(MINUS), sup(v("\u03b5"), t("*")))], sub(v("v"), v("l"))),
        )),
    ],
    # Probability that FLAML picks learner l for the next trial.
    "eq:select": [
        v("P"), paren(v("l")), t("="),
        frac(subsup(t("ECI"), v("l"), t(MINUS + "1")),
             nary(SUM, v("k"), None, subsup(t("ECI"), v("k"), t(MINUS + "1")))),
    ],
    # CFO's two trial points around the incumbent and its first step length.
    "eq:cfo": [
        sup(v("x"), t(PRIME)), t("="), sup(v("x"), t("*")), t("+"), v("\u03b4"), v("u"),
        t(GAP),
        sup(v("x"), t(PRIME * 2)), t("="), sup(v("x"), t("*")), t(MINUS), v("\u03b4"), v("u"),
        t(GAP),
        sub(v("\u03b4"), t("0")), t("=0.1"), sqrt(v("d")),
    ],
    # Critic: overfitting gap over the J Session_Folds.
    "eq:gap": [
        v("g"), t("="), frac(t("1"), v("J")),
        nary(SUM, [v("j"), t("=1")], v("J"),
             paren(subsup(v("m"), v("j"), t("train")), t(MINUS), subsup(v("m"), v("j"), t("val")))),
    ],
    # Critic: skill over the trivial baseline (underfitting).
    "eq:skill": [
        v("s"), t("="),
        frac([sup(bar(v("m")), t("val")), t(MINUS), sup(v("m"), t("base"))],
             [t("1"), t(MINUS), sup(v("m"), t("base"))]),
    ],
    # Judge: corrected resampled t statistic (Nadeau and Bengio; Bouckaert and Frank).
    "eq:ttest": [
        v("t"), t("="),
        frac(bar(v("d")),
             sqrt(paren(frac(t("1"), v("J")), t("+"), frac(sub(v("n"), t("val")), sub(v("n"), t("train")))),
                  sup(hat(v("\u03c3")), t("2")))),
    ],
    # Judge: sample variance of the fold differences.
    "eq:var": [
        sup(hat(v("\u03c3")), t("2")), t("="), frac(t("1"), [v("J"), t(MINUS), t("1")]),
        nary(SUM, [v("j"), t("=1")], v("J"), sup(paren(sub(v("d"), v("j")), t(MINUS), bar(v("d"))), t("2"))),
    ],
    # Resource_Manager: search budget of the next Experiment.
    "eq:budget": [
        v("b"), t("="),
        func("max", paren(t("5, "),
                          paren(frac([v("T"), t(MINUS), sub(v("t"), t("e")), t(MINUS), v("\u03c1")], v("s")),
                                left="\u230a", right="\u230b"))),
    ],
    # Data_Step target_encode: smoothed mean target of a value c.
    "eq:te": [
        func("enc", paren(v("c"))), t("="),
        frac([nary(SUM, [v("i"), t(":"), sub(v("x"), v("i")), t("="), v("c")], None, sub(v("y"), v("i"))),
              t("+"), v("m"), bar(v("y"))],
             [sub(v("n"), v("c")), t("+"), v("m")]),
    ],
    # Learned strategy choice: Thompson sampling score of an arm a.
    "eq:thompson": [
        sub(v("p"), v("a")), t("\u223c"),
        func("Beta", paren(t("1+"), sub(v("K"), v("a")), t(", 1+"), sub(v("D"), v("a")))),
        t(GAP),
        sub(t("score"), v("a")), t("="),
        frac([sub(v("p"), v("a")), sub(bar(v("G")), v("a"))],
             func("max", paren(sub(bar(v("C")), v("a")), t(", 1")))),
    ],
    # Learned strategy choice: UCB1 score of an arm a.
    "eq:ucb": [
        sub(t("score"), v("a")), t("="),
        frac(sub(v("K"), v("a")), [sub(v("K"), v("a")), t("+"), sub(v("D"), v("a"))]),
        t("+"),
        sqrt(frac([t("2"), func("ln", v("N"))], sub(v("n"), v("a")))),
    ],
    # Benchmark: Friedman statistic over k systems and N datasets.
    "eq:friedman": [
        subsup(v("\u03c7"), v("F"), t("2")), t("="),
        frac([t("12"), v("N")], [v("k"), paren(v("k"), t("+1"))]),
        paren(nary(SUM, v("j"), None, subsup(v("R"), v("j"), t("2"))), t(MINUS),
              frac([v("k"), sup(paren(v("k"), t("+1")), t("2"))], t("4")),
              left="[", right="]"),
    ],
    # Benchmark: Nemenyi critical difference of average ranks.
    "eq:cd": [
        t("CD"), t("="), sub(v("q"), v("\u03b1")),
        sqrt(frac([v("k"), paren(v("k"), t("+1"))], [t("6"), v("N")])),
    ],
    # Judge agreement study: Cohen's kappa.
    "eq:kappa": [
        v("\u03ba"), t("="),
        frac([sub(v("p"), t("o")), t(MINUS), sub(v("p"), t("e"))], [t("1"), t(MINUS), sub(v("p"), t("e"))]),
    ],
}
