"""Draw the figures of the final report: the diagrams and the result charts.

Run from autopilot-ml/:  python scripts/report/report_figures.py [name ...]

Every figure is drawn at the width it has in the report and in Times New Roman, so that
component names print at 12 pt and other text at 10 to 11 pt, as the department asks.
The charts read the second benchmark run's summary files in results/benchmark-v2/;
the diagrams are drawn from the structure of the code. Output: docs/imp/final-report/figures/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "imp" / "final-report" / "figures"
RES = ROOT / "results" / "benchmark-v2"
CM = 1 / 2.54
NAME, TEXT, TICK = 12, 10.5, 10
GREY = "#d9d9d9"
LIGHT = "#f2f2f2"
plt.rcParams.update({
    "font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": TEXT,
    "axes.titlesize": NAME, "axes.labelsize": NAME, "xtick.labelsize": TICK, "ytick.labelsize": TICK,
    "legend.fontsize": TEXT, "axes.linewidth": 0.6, "lines.linewidth": 1.2, "savefig.dpi": 300,
})
SYSTEM_NAMES = {
    "flaml": "FLAML alone", "autopilot_full": "AutoPilot-ML full", "autopilot_rules": "AutoPilot-ML rules",
    "autogluon": "AutoGluon", "manual": "Manual LightGBM", "random_choice": "Random order",
    "no_critic": "Without Critic", "no_resource_manager": "Without Resource_Manager",
}
SHORT_NAMES = {
    "flaml": "FLAML alone", "autopilot_full": "AutoPilot full", "autopilot_rules": "AutoPilot rules",
    "autogluon": "AutoGluon", "manual": "Manual LightGBM", "random_choice": "Random order",
    "no_critic": "No Critic", "no_resource_manager": "No Resource_Manager",
}
DATASETS = ["pc4", "cylinder-bands", "segment", "MiceProtein", "energy_efficiency", "solar_flare"]


# ----------------------------------------------------------------------------- drawing helpers


class Canvas:
    """A blank drawing in centimetres, so positions and font sizes match the printed page."""

    def __init__(self, width: float, height: float):
        self.fig = plt.figure(figsize=(width * CM, height * CM))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.axis("off")

    def text(self, x, y, s, size=TEXT, ha="center", va="center", **kw):
        self.ax.text(x, y, s, ha=ha, va=va, fontsize=size, linespacing=1.15, **kw)

    def rect(self, x, y, w, h, fill="white", rounded=True, lw=0.8, ls="-"):
        style = "round,pad=0,rounding_size=0.18" if rounded else "square,pad=0"
        self.ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style, fc=fill, ec="black",
                                         lw=lw, linestyle=ls))

    def box(self, x, y, w, h, name, sub=None, fill="white", rounded=True, lw=0.8):
        self.rect(x, y, w, h, fill, rounded, lw)
        if sub:
            self.text(x, y + 0.21, name, NAME)
            self.text(x, y - 0.26, sub, TEXT)
        else:
            self.text(x, y, name, NAME)

    def terminal(self, x, y, w, h, text):
        self.ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                         boxstyle=f"round,pad=0,rounding_size={h / 2}", fc=LIGHT, ec="black", lw=0.8))
        self.text(x, y, text, NAME)

    def diamond(self, x, y, w, h, text):
        pts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
        self.ax.add_patch(Polygon(pts, closed=True, fc="white", ec="black", lw=0.8))
        self.text(x, y, text, NAME)

    def circle(self, x, y, r, number, name):
        self.ax.add_patch(Circle((x, y), r, fc="white", ec="black", lw=0.8))
        self.text(x, y + 0.62 * r, number, TEXT)
        self.text(x, y - 0.14, name, NAME)

    def store(self, x, y, w, label, name):
        """A data store: an open box with its label in a cell on the left."""
        h = 0.75
        for yy in (y + h / 2, y - h / 2):
            self.ax.plot([x - w / 2, x + w / 2], [yy, yy], color="black", lw=0.8)
        for xx in (x - w / 2, x - w / 2 + 0.8):
            self.ax.plot([xx, xx], [y - h / 2, y + h / 2], color="black", lw=0.8)
        self.text(x - w / 2 + 0.4, y, label, TEXT)
        self.text(x - w / 2 + 0.8 + (w - 0.8) / 2, y, name, NAME)

    def entity(self, x, y, w, h, name):
        self.box(x, y, w, h, name, rounded=False, lw=1.4)

    def line(self, points, label=None, at=None, dashed=False, head=True, ha="center", va="bottom", rotation=0):
        xs, ys = zip(*points)
        self.ax.plot(xs, ys, color="black", lw=0.8, linestyle=(0, (4, 2.5)) if dashed else "-")
        if head:
            self.ax.annotate("", xy=points[-1], xytext=points[-2],
                             arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8, mutation_scale=9,
                                             shrinkA=0, shrinkB=0))
        if label:
            self.text(at[0], at[1], label, TEXT, ha=ha, va=va, rotation=rotation)

    def save(self, name: str) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(OUT / name, dpi=300)
        plt.close(self.fig)


# ----------------------------------------------------------------------------- diagrams


def fig_loop() -> None:
    c = Canvas(15.5, 7.7)
    c.box(12.85, 6.85, 4.8, 1.2, "Resource_Manager", "budgets and stop rules")
    c.box(7.75, 5.7, 5.0, 1.3, "Search a Setup", "FLAML, used unchanged", fill=GREY)
    c.box(12.85, 4.0, 4.8, 1.3, "Assess", "refits on 15 Session_Folds")
    c.box(12.85, 1.1, 4.8, 1.3, "Judge", "keep, keep_simpler, discard")
    c.box(2.65, 1.1, 4.8, 1.3, "Diagnose", "Critic, 14 kinds")
    c.box(2.65, 4.0, 4.8, 1.3, "Propose a change", "Strategy_Engine")
    c.box(7.75, 2.75, 4.4, 1.3, "Experiment_Memory", "changes, effects and costs")
    c.line([(2.65, 4.65), (2.65, 5.7), (5.25, 5.7)], "next change", (2.8, 5.85), ha="left")
    c.line([(10.25, 5.7), (12.85, 5.7), (12.85, 4.65)], "best\nconfiguration", (13.0, 5.2), ha="left",
           va="center")
    c.line([(12.85, 3.35), (12.85, 1.75)], "fold scores", (13.0, 2.55), ha="left", va="center")
    c.line([(10.45, 1.1), (5.05, 1.1)], "if kept: the new Accepted_Setup", (7.75, 0.85), va="top")
    c.line([(2.65, 1.75), (2.65, 3.35)], "Diagnoses", (2.8, 2.55), ha="left", va="center")
    c.line([(10.9, 1.75), (9.95, 2.35)], "record", (10.55, 2.25), dashed=True, ha="left")
    c.line([(5.55, 3.15), (5.05, 3.55)], "past effects", (5.7, 3.55), dashed=True, ha="left")
    c.line([(10.45, 6.85), (9.65, 6.35)], "budget", (10.0, 6.9), dashed=True, ha="right")
    c.save("fig_loop.png")


def fig_flaml() -> None:
    c = Canvas(10.6, 13.6)
    x, w = 4.6, 8.0
    c.terminal(x, 12.95, 7.0, 0.9, "Start: training rows and time budget")
    c.box(x, 11.5, w, 1.2, "Choose the validation", "5-fold cross-validation or a 10% holdout")
    c.box(x, 9.85, w, 1.2, "Pick a learner", "estimated cost for improvement (ECI)")
    c.box(x, 8.2, w, 1.2, "Propose one configuration", "a CFO step around the learner's best")
    c.box(x, 6.55, w, 1.2, "Train and score it", "on the validation folds")
    c.box(x, 4.9, w, 1.2, "Update the learner's record", "best loss, times and ECI")
    c.diamond(x, 3.3, 3.4, 1.5, "Time\nleft?")
    c.box(x, 1.75, w, 1.2, "Train the best configuration", "on all training rows")
    c.terminal(x, 0.5, 6.4, 0.8, "Return the model and its history")
    for y0, y1 in ((12.5, 12.1), (10.9, 10.45), (9.25, 8.8), (7.6, 7.15), (5.95, 5.5), (4.3, 4.05)):
        c.line([(x, y0), (x, y1)])
    c.line([(x, 2.55), (x, 2.35)], "no", (x + 0.2, 2.45), ha="left", va="center")
    c.line([(x, 1.15), (x, 0.9)])
    c.line([(x + 1.7, 3.3), (9.9, 3.3), (9.9, 9.85), (x + w / 2, 9.85)], "yes", (6.6, 3.45), ha="left")
    c.save("fig_flaml.png")


def fig_split() -> None:
    c = Canvas(15.5, 6.4)
    ax = c.ax
    left, right, split = 0.4, 15.1, 11.9
    c.box((left + right) / 2, 5.9, right - left, 0.8, "Dataset D (the rows with a target)", rounded=False)
    c.box((left + split) / 2, 4.6, split - left - 0.1, 0.8, "Training_Part: 80% of the rows", rounded=False,
          fill=LIGHT)
    c.box((split + right) / 2 + 0.05, 4.6, right - split - 0.1, 0.8, "Test_Part: 20%", rounded=False, fill=GREY)
    c.line([(6.15, 5.5), (6.15, 5.0)], "seeded, stratified split", (6.35, 5.25), ha="left", va="center")
    c.line([(13.5, 5.5), (13.5, 5.0)])
    cell_w = (split - left - 1.9) / 5
    for r in range(3):
        y = 3.45 - r * 0.8
        c.text(left, y, f"Repeat {r + 1}", ha="left")
        for k in range(5):
            x0 = left + 1.9 + k * cell_w
            val = k == (2 * r + 1) % 5
            ax.add_patch(FancyBboxPatch((x0 + 0.04, y - 0.28), cell_w - 0.08, 0.56, boxstyle="square,pad=0",
                                        fc=GREY if val else "white", ec="black", lw=0.6))
            c.text(x0 + cell_w / 2, y, f"part {k + 1}", TICK)
    c.text(left, 0.55, "15 Session_Folds: in each fold one part gives the validation rows (grey in this "
           "example) and the\nother four parts the training rows. The Assessor, the Critic and the Judge use "
           "only these folds.", ha="left")
    c.text((split + right) / 2 + 0.05, 2.65, "Used once, at\nthe end, to score\nthe first and\nthe final model")
    c.save("fig_split.png")


def fig_block() -> None:
    c = Canvas(15.5, 14.1)
    x0, x1 = 0.7, 15.45
    w3, w4, h = 4.65, 3.45, 0.98
    xs3 = [x0 + 0.15 + w3 / 2 + i * (w3 + 0.3) for i in range(3)]
    xs4 = [x0 + 0.15 + w4 / 2 + i * (w4 + 0.25) for i in range(4)]

    def band(top, title, rows):
        height = 0.5 + rows * 1.18 + 0.05
        c.rect((x0 + x1) / 2, top - height / 2, x1 - x0, height, fill="none", rounded=False, lw=0.6, ls=(0, (3, 2)))
        c.text(x0 + 0.15, top - 0.25, title, ha="left", style="italic")
        return [top - 0.5 - 0.56 - i * 1.18 for i in range(rows)], top - height

    ys, bottom = band(14.05, "User interfaces (built in this project)", 1)
    for x, (n, s) in zip(xs3, [("Dashboard", "Streamlit pages"), ("Command line", "scripts/*.py"),
                               ("Web API", "FastAPI, local only")]):
        c.box(x, ys[0], w3, h, n, s)
    c.line([(7.75, bottom - 0.02), (7.75, bottom - 0.43)], "start, stop and read Sessions", (7.95, bottom - 0.22),
           ha="left", va="center")
    ys, bottom = band(bottom - 0.45, "Session layer: the autonomous loop (built in this project)", 3)
    rows = [[("Session loop", "runs the Experiments"), ("Critic", "14 kinds of Diagnosis"),
             ("Strategy_Engine", "12 Strategies")],
            [("Judge", "corrected t-test"), ("Assessor", "refits on 15 folds"),
             ("Resource_Manager", "budgets, stop rules")],
            [("Planner", "the first Setup"), ("Experiment_Memory", "the learned order"),
             ("Data_Steps", "fitted in each fold")]]
    for y, row in zip(ys, rows):
        for x, (n, s) in zip(xs3, row):
            c.box(x, y, w3, h, n, s)
    session_mid = (ys[0] + ys[-1]) / 2
    c.line([(7.75, bottom - 0.02), (7.75, bottom - 0.43)], "search, refit, profile, explain",
           (7.95, bottom - 0.22), ha="left", va="center")
    ys, bottom = band(bottom - 0.45, "Experiment layer: one search and its evaluation (built in this project)", 2)
    rows = [[("AutoML runner", "one FLAML search"), ("Profiler", "and task detector"),
             ("Evaluator", "metrics, importance")],
            [("Tracker", "records, MLflow runs"), ("Explainer", "SHAP values"), ("Narrator", "local summaries")]]
    for y, row in zip(ys, rows):
        for x, (n, s) in zip(xs3, row):
            c.box(x, y, w3, h, n, s)
    c.line([(7.75, bottom - 0.02), (7.75, bottom - 0.43)], "search, train, score", (7.95, bottom - 0.22),
           ha="left", va="center")
    ys, bottom = band(bottom - 0.45, "Libraries used unchanged", 1)
    for x, (n, s) in zip(xs4, [("FLAML", "search engine"), ("scikit-learn", "imbalanced-learn"),
                               ("LightGBM", "and XGBoost"), ("SHAP", "and Ollama")]):
        c.box(x, ys[0], w4, h, n, s, fill=GREY)
    ys, bottom = band(bottom - 0.45, "Storage (files on this machine)", 1)
    for x, (n, s) in zip(xs4, [("JSON records", "experiments/"), ("Model files", "models/"),
                               ("MLflow runs", "mlruns/"), ("Memory file", "memory.jsonl")]):
        c.box(x, ys[0], w4, h, n, s, fill=LIGHT)
    c.line([(x0, session_mid), (0.3, session_mid), (0.3, ys[0]), (x0, ys[0])], "records, models and runs",
           (0.48, (session_mid + ys[0]) / 2), dashed=True, rotation=90, ha="center", va="center")
    c.save("fig_block.png")


def fig_dfd0() -> None:
    c = Canvas(15.0, 6.2)
    c.entity(1.9, 3.1, 3.0, 1.3, "User")
    c.circle(8.0, 3.1, 1.7, "0", "AutoPilot-ML")
    c.entity(13.3, 5.45, 3.0, 0.9, "OpenML")
    c.entity(13.2, 0.75, 3.2, 0.9, "Ollama server")
    c.line([(3.4, 3.65), (6.39, 3.65)], "dataset (CSV),\ntarget, budget\nand settings", (4.85, 3.8))
    c.line([(6.39, 2.55), (3.4, 2.55)], "models, test scores,\nDiagnoses, Verdicts,\nexplanations", (4.85, 2.4),
           va="top")
    c.line([(11.8, 5.45), (9.2, 4.3)], "public datasets for the\nbenchmark (once)", (9.7, 5.15))
    c.line([(9.2, 1.9), (11.6, 1.0)], "facts of a finished\nSession record", (11.3, 2.05))
    c.line([(11.6, 0.5), (8.58, 1.5)], "summary text, labelled\nas generated", (8.6, 0.85), va="top")
    c.save("fig_dfd0.png")


def fig_dfd1() -> None:
    c = Canvas(15.5, 11.2)
    r = 1.12
    xl, xm, xr = 2.3, 7.6, 12.9
    y1, y2, y3, y4 = 10.15, 6.95, 3.55, 0.8
    xb, yb, rb = 11.9, y1 - 0.15, 1.05  # process 5.0
    c.entity(xl, y1, 2.8, 0.9, "User")
    c.store(7.0, y1, 4.0, "D1", "Dataset (CSV)")
    c.circle(xb, yb, rb, "5.0", "Manage\nbudget")
    c.circle(xl, y2, r, "1.0", "Prepare\ndata")
    c.circle(xm, y2, r, "2.0", "Run\nExperiment")
    c.circle(xr, y2, r, "3.0", "Judge")
    c.circle(xl, y3, r, "6.0", "Final\nevaluation")
    c.store(xm, y3, 4.4, "D2", "Records, models")
    c.circle(xr, y3, r, "4.0", "Diagnose,\npropose")
    c.store(12.3, y4, 5.0, "D3", "Experiment_Memory")
    d, db = 0.707 * r, 0.707 * rb

    def along(p0, p1, t, dx=0.0, dy=0.0):
        return (p0[0] + t * (p1[0] - p0[0]) + dx, p0[1] + t * (p1[1] - p0[1]) + dy)

    c.line([(xl, y1 - 0.45), (xl, y2 + r)], "dataset,\ntarget,\nbudget", (xl - 0.2, (y1 - 0.45 + y2 + r) / 2),
           ha="right", va="center")
    p0, p1 = (5.0, y1), (xl + d, y2 + d)
    c.line([p0, p1], "rows", along(p0, p1, 0.45, dx=0.2), ha="left", va="center")
    c.line([(xl + r, y2), (xm - r, y2)], "Training_Part,\nfolds, first Setup", ((xl + xm) / 2, y2 + 0.15))
    c.line([(xm + r, y2), (xr - r, y2)], "fold scores of\nboth Setups", ((xm + xr) / 2, y2 + 0.15))
    c.line([(xr, y2 - r), (xr, y3 + r)], "kept Setup", (xr + 0.2, (y2 + y3) / 2), ha="left", va="center")
    p0, p1 = (xr - d, y3 + d), (xm + d, y2 - d)
    c.line([p0, p1], "changed Setup", along(p0, p1, 0.4, dy=-0.3), va="top")
    p0, p1 = (xr - d, y2 + d), (xb + 0.2, yb - rb + 0.02)
    c.line([p0, p1], "step cost", along(p0, p1, 0.5, dx=0.25), ha="left", va="center")
    p0, p1 = (xb - db, yb - db), (xm + d, y2 + d)
    c.line([p0, p1], "search budget", along(p0, p1, 0.55, dx=-0.2, dy=0.3), va="bottom")
    c.line([(xm, y2 - r), (xm, y3 + 0.375)], "models and\nrecords", (xm - 0.2, (y2 - r + y3 + 0.375) / 2),
           ha="right", va="center")
    c.line([(xl, y2 - r), (xl, y3 + r)], "Test_Part", (xl + 0.2, (y2 + y3) / 2), ha="left", va="center")
    c.line([(xm - 2.2, y3), (xl + r, y3)], "first and\nfinal model", ((xm - 2.2 + xl + r) / 2, y3 + 0.15))
    c.line([(xl - r, y3), (0.4, y3), (0.4, y1), (xl - 1.4, y1)], "test scores and explanations",
           (0.68, (y3 + y1) / 2), rotation=90, va="center")
    c.line([(xr + r, y2), (15.15, y2), (15.15, y4), (14.8, y4)], "Verdict records", (14.92, (y2 + y4) / 2 - 0.4),
           rotation=90, va="center")
    c.line([(xr, y4 + 0.375), (xr, y3 - r)], "past effects", (xr - 0.2, (y4 + 0.375 + y3 - r) / 2), ha="right",
           va="center")
    c.save("fig_dfd1.png")


def fig_flow() -> None:
    c = Canvas(12.0, 19.4)
    x, w = 5.3, 8.0
    c.terminal(x, 18.75, 6.8, 0.9, "Start: dataset, target, budget")
    c.box(x, 17.35, w, 1.2, "Split and make the folds", "Training_Part, Test_Part, 15 Session_Folds")
    c.box(x, 15.8, w, 1.2, "Search the first Setup", "35% of the time, Training_Part only")
    c.box(x, 14.25, w, 1.2, "Assess and diagnose", "refits on the folds, then the Critic")
    c.box(x, 12.7, w, 1.2, "Propose changes not tried yet", "rule table, ordered by the selector")
    c.diamond(x, 10.95, 3.9, 1.7, "Stop?")
    c.box(x, 9.2, w, 1.2, "Search the candidate", "the next budget, warm started")
    c.box(x, 7.65, w, 1.2, "Assess the candidate", "on the same 15 folds")
    c.diamond(x, 5.85, 4.2, 1.8, "Judge:\nkept?")
    c.box(2.9, 3.8, 4.4, 1.2, "Accept the candidate", "then diagnose it")
    c.box(8.3, 3.8, 3.6, 1.2, "Keep the", "Accepted_Setup")
    c.box(x, 2.15, w, 1.0, "Write the step's record and the memory")
    for y0, y1 in ((18.3, 17.95), (16.75, 16.4), (15.2, 14.85), (13.65, 13.3), (12.1, 11.8), (8.6, 8.25),
                   (7.05, 6.75)):
        c.line([(x, y0), (x, y1)])
    c.line([(x, 10.1), (x, 9.8)], "no", (x + 0.2, 9.95), ha="left", va="center")
    c.line([(x - 2.1, 5.85), (2.9, 5.85), (2.9, 4.4)], "yes", (3.05, 6.0), ha="left")
    c.line([(x + 2.1, 5.85), (8.3, 5.85), (8.3, 4.4)], "no", (7.55, 6.0), ha="left")
    c.line([(2.9, 3.2), (2.9, 2.65)])
    c.line([(8.3, 3.2), (8.3, 2.65)])
    c.line([(x - w / 2, 2.15), (0.7, 2.15), (0.7, 12.7), (x - w / 2, 12.7)], "next step", (0.42, 7.5),
           rotation=90, va="center")
    c.line([(x + 1.95, 10.95), (10.9, 10.95), (10.9, 1.05), (x + 2.6, 1.05)], "yes", (7.5, 11.1), ha="left")
    c.terminal(x, 1.05, 5.2, 0.9, "Final_Evaluation, then end")
    c.save("fig_flow.png")


def fig_judge() -> None:
    c = Canvas(13.0, 12.6)
    x = 4.6
    c.terminal(x, 11.95, 8.4, 0.9, "Fold differences d(j), j = 1 to 15")
    c.diamond(x, 10.2, 4.2, 1.7, "All d(j)\nequal?")
    c.rect(10.55, 10.45, 4.7, 2.3)
    c.text(10.55, 11.2, "Decide on the mean", NAME)
    c.text(10.55, 10.15, "keep if the mean > 0;\nelse keep_simpler if simpler\nand mean \u2265 \u22120.005;"
           "\nelse discard")
    c.box(x, 8.3, 8.2, 1.2, "Corrected resampled t statistic", "then p(better) and p(worse), Student t")
    c.diamond(x, 6.35, 5.8, 1.9, "mean > 0 and\np(better) \u2264 0.20?")
    c.diamond(x, 3.85, 7.0, 2.2, "simpler, mean \u2265 \u22120.005\nand p(worse) > 0.20?")
    c.terminal(11.0, 6.35, 2.8, 0.9, "keep")
    c.terminal(11.0, 3.85, 2.8, 0.9, "keep_simpler")
    c.terminal(x, 1.75, 3.0, 0.9, "discard")
    c.line([(x, 11.5), (x, 11.05)])
    c.line([(x + 2.1, 10.2), (8.2, 10.2)], "yes", (7.4, 10.35))
    c.line([(x, 9.35), (x, 8.9)], "no", (x + 0.2, 9.12), ha="left", va="center")
    c.line([(x, 7.7), (x, 7.3)])
    c.line([(x + 2.9, 6.35), (9.6, 6.35)], "yes", (8.55, 6.5))
    c.line([(x, 5.4), (x, 4.95)], "no", (x + 0.2, 5.17), ha="left", va="center")
    c.line([(x + 3.5, 3.85), (9.6, 3.85)], "yes", (8.75, 4.0))
    c.line([(x, 2.75), (x, 2.2)], "no", (x + 0.2, 2.47), ha="left", va="center")
    c.text(0.3, 0.55, "alpha 0.20, margin 0.005, minimum gain 0; a mean within 10\u207b\u00b9\u00b2 is no gain",
           ha="left")
    c.save("fig_judge.png")


# ----------------------------------------------------------------------------- charts


def per_dataset() -> pd.DataFrame:
    return pd.read_csv(RES / "analysis_per_dataset.csv")


def fig_scores() -> None:
    d = per_dataset()
    systems = ["flaml", "autopilot_full", "autopilot_rules", "autogluon", "manual"]
    styles = {"flaml": ("-", "o"), "autopilot_full": ("-", "s"), "autopilot_rules": ("--", "^"),
              "autogluon": (":", "D"), "manual": ("-.", "v")}
    colours = {"flaml": "#000000", "autopilot_full": "#1f4e79", "autopilot_rules": "#5b9bd5",
               "autogluon": "#7f7f7f", "manual": "#a5a5a5"}
    fig, axes = plt.subplots(2, 3, figsize=(15.5 * CM, 11.4 * CM))
    for ax, ds in zip(axes.flat, DATASETS):
        sub = d[d.dataset == ds]
        for s in systems:
            row = sub[sub.system == s].sort_values("budget_seconds")
            ls, mk = styles[s]
            ax.plot(row.budget_seconds, row.mean_test_score, ls, marker=mk, ms=4, color=colours[s],
                    label=SYSTEM_NAMES[s])
        metric = "F1" if sub.metric.iloc[0] == "f1" else "R\u00b2"
        ax.set_title(ds, fontsize=NAME)
        ax.set_xticks([120, 300])
        ax.set_xlim(90, 330)
        ax.set_ylabel(f"test {metric}", fontsize=TEXT)
        ax.tick_params(length=2)
        ax.grid(axis="y", lw=0.3, color="#bfbfbf")
    for ax in axes[1]:
        ax.set_xlabel("budget (s)", fontsize=TEXT)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=TEXT,
               handlelength=2.4, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.11, 1, 1), h_pad=0.9, w_pad=0.6)
    fig.savefig(OUT / "fig_scores.png", dpi=300)
    plt.close(fig)


def fig_diff() -> None:
    d = per_dataset()
    p = d.pivot_table(index=["budget_seconds", "dataset"], columns="system", values="mean_test_score")
    diff = (p["autopilot_full"] - p["flaml"]).unstack(0)
    pairs = pd.read_csv(RES / "analysis_pairs.csv")
    fig, ax = plt.subplots(figsize=(15.5 * CM, 7.6 * CM))
    ys = np.arange(len(DATASETS))[::-1]
    ax.barh(ys + 0.18, diff.loc[DATASETS, 120], height=0.34, color="#a5a5a5", edgecolor="black", lw=0.5,
            label="budget 120 s")
    ax.barh(ys - 0.18, diff.loc[DATASETS, 300], height=0.34, color="#1f4e79", edgecolor="black", lw=0.5,
            label="budget 300 s")
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(-0.005, color="black", lw=0.8, ls="--")
    ax.set_ylim(-0.85, len(DATASETS) - 0.4)
    ax.text(-0.0054, -0.58, "margin \u22120.005", ha="right", va="center", fontsize=TICK)
    ax.set_yticks(ys)
    ax.set_yticklabels(DATASETS, fontsize=TEXT)
    ax.set_xlabel("AutoPilot-ML full minus FLAML alone (mean test score over 5 seeds)", fontsize=TEXT)
    for budget in (120, 300):
        row = pairs[(pairs.family == "main") & (pairs.budget_seconds == budget) & (pairs.system_a == "autopilot_full")
                    & (pairs.system_b == "flaml")].iloc[0]
        print(f"  median difference at {budget} s: {row.median_difference:.4f} "
              f"[{row.ci_low:.4f}, {row.ci_high:.4f}]")
    ax.legend(loc="lower right", frameon=False, fontsize=TEXT)
    ax.grid(axis="x", lw=0.3, color="#bfbfbf")
    ax.tick_params(length=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_diff.png", dpi=300)
    plt.close(fig)


def cd_panel(ax, ranks: pd.DataFrame, cd: float, title: str) -> None:
    """One critical-difference diagram in the style of Demsar (2006)."""
    ranks = ranks.sort_values("average_rank")
    k = len(ranks)
    ax.set_xlim(1, k)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ya = 0.62
    ax.plot([1, k], [ya, ya], color="black", lw=0.8)
    for t in range(1, k + 1):
        ax.plot([t, t], [ya, ya + 0.05], color="black", lw=0.8)
        ax.text(t, ya + 0.08, str(t), ha="center", va="bottom", fontsize=TICK)
    ax.plot([1, 1 + cd], [0.93, 0.93], color="black", lw=1.4)
    for xx in (1, 1 + cd):
        ax.plot([xx, xx], [0.9, 0.96], color="black", lw=0.8)
    ax.text(1 + cd + 0.1, 0.93, f"CD = {cd:.2f}", ha="left", va="center", fontsize=TICK)
    ax.text(0.85, 0.93, title, ha="right", va="center", fontsize=TEXT, style="italic", clip_on=False)
    lo, hi = ranks.average_rank.min(), ranks.average_rank.max()
    if hi - lo < cd:
        ax.plot([lo - 0.04, hi + 0.04], [ya - 0.09, ya - 0.09], color="black", lw=2.4, solid_capstyle="butt")
    half = (k + 1) // 2
    for i, (_, row) in enumerate(ranks.iterrows()):
        rk = row.average_rank
        name = f"{SHORT_NAMES[row.system]} ({rk:.2f})"
        j = i if i < half else k - 1 - i
        y = ya - 0.25 - 0.17 * j
        if i < half:
            ax.plot([rk, rk, 0.92], [ya, y, y], color="black", lw=0.6, clip_on=False)
            ax.text(0.88, y, name, ha="right", va="center", fontsize=TEXT, clip_on=False)
        else:
            ax.plot([rk, rk, k + 0.08], [ya, y, y], color="black", lw=0.6, clip_on=False)
            ax.text(k + 0.12, y, name, ha="left", va="center", fontsize=TEXT, clip_on=False)


def fig_cd() -> None:
    ranks = pd.read_csv(RES / "analysis_ranks.csv")
    tests = pd.read_csv(RES / "analysis_tests.csv")
    fig, axes = plt.subplots(3, 1, figsize=(15.5 * CM, 12.0 * CM))
    for ax, (family, budget, title) in zip(axes, [("main", 120, "main systems, 120 s"),
                                                  ("main", 300, "main systems, 300 s"),
                                                  ("ablations", 300, "ablations, 300 s")]):
        sub = ranks[(ranks.family == family) & (ranks.budget_seconds == budget)]
        cd = tests[(tests.family == family) & (tests.budget_seconds == budget)].nemenyi_cd.iloc[0]
        cd_panel(ax, sub, cd, title)
    fig.subplots_adjust(left=0.27, right=0.67, top=0.98, bottom=0.02, hspace=0.18)
    fig.savefig(OUT / "fig_cd.png", dpi=300)
    plt.close(fig)


def fig_time() -> None:
    """Mean wall time per run of the five main systems at the 300 s budget."""
    timing = pd.read_csv(RES / "analysis_timing.csv")
    order = ["flaml", "autopilot_rules", "autopilot_full", "autogluon", "manual"]
    colours = {"flaml": "#000000", "autopilot_full": "#1f4e79", "autopilot_rules": "#5b9bd5",
               "autogluon": "#7f7f7f", "manual": "#a5a5a5"}
    t300 = timing[timing.budget_seconds == 300].set_index("system").loc[order]
    fig, ax = plt.subplots(figsize=(15.5 * CM, 6.0 * CM))
    ys = np.arange(len(order))[::-1]
    ax.barh(ys, t300.mean_wall_seconds, color=[colours[s] for s in order], edgecolor="black", lw=0.5,
            height=0.6)
    for yy, v in zip(ys, t300.mean_wall_seconds):
        ax.text(v + 5, yy, f"{v:.1f} s", ha="left", va="center", fontsize=TICK)
    ax.axvline(300, color="black", lw=0.8, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([SYSTEM_NAMES[s] for s in order], fontsize=TICK)
    ax.set_xlabel("mean wall time per run in seconds (budget 300 s, dashed line)", fontsize=TEXT)
    ax.set_xlim(0, 365)
    ax.tick_params(length=2)
    fig.subplots_adjust(left=0.24, right=0.97, top=0.96, bottom=0.21)
    fig.savefig(OUT / "fig_time.png", dpi=300)
    plt.close(fig)


FIGURES = {f.__name__: f for f in (fig_loop, fig_flaml, fig_split, fig_block, fig_dfd0, fig_dfd1, fig_flow,
                                    fig_judge, fig_scores, fig_diff, fig_cd, fig_time)}


def main() -> None:
    names = sys.argv[1:] or list(FIGURES)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        FIGURES[name]()
        print("drew", name)


if __name__ == "__main__":
    main()
