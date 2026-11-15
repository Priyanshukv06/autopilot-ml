"""Build the 5-minute project presentation (.pptx) in the department's prescribed slide order.

Run from autopilot-ml/ with the automl environment (needs python-pptx 1.0.2 and matplotlib):
    python scripts/report/build_slides.py            # .pptx, then PowerPoint adds animations and a PDF copy
    python scripts/report/build_slides.py --no-office

Slides: title, then the nine prescribed sections in order (10 slides). Every slide has speaker notes
(about 600 words, 4.5 minutes in total) for a 5-minute talk. Shapes named "anim:NN:effect" get an entrance
animation from ppt_finalize.ps1 (fade or wipe, automatic after the previous one), so each slide needs one
click. Slides 5 and 7 show the report's loop and block diagrams; slide 6 draws the report's flowchart
(Figure 6) with PowerPoint shapes, slightly simplified, so that its text stays legible when projected.
All numbers are the measured ones of the final report (docs/imp/final-report/).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.oxml.ns import qn  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = ROOT / "docs" / "imp" / "final-report"
OUT_DIR = ROOT / "docs" / "imp" / "presentation"
FIG_DIR = OUT_DIR / "figures"
OUT_PPTX = OUT_DIR / "AutoPilot-ML-Presentation.pptx"
RESULTS = ROOT / "results" / "benchmark-v2"
TEMPLATE = REPORT / "template" / "Minor Project format - final 2026.docx"

FONT = "Calibri"
DARK, MID, LIGHT, PALE = RGBColor(0x1F, 0x4E, 0x79), RGBColor(0x5B, 0x9B, 0xD5), RGBColor(0xDD, 0xEB, 0xF7), \
    RGBColor(0xF3, 0xF7, 0xFB)
TEXT, GREY, WHITE, ARROW = RGBColor(0x22, 0x22, 0x22), RGBColor(0x59, 0x59, 0x59), RGBColor(0xFF, 0xFF, 0xFF), \
    RGBColor(0x40, 0x40, 0x40)
W, H = 13.333, 7.5
TOTAL = 10


# ----------------------------------------------------------------------------- helpers


def runs(paragraph, text: str, size: float, color=TEXT, bold=False, italic=False) -> None:
    """Add text to a paragraph; **...** marks bold parts."""
    for k, part in enumerate(re.split(r"\*\*", text)):
        if not part:
            continue
        r = paragraph.add_run()
        r.text = part
        f = r.font
        f.name, f.size, f.color.rgb = FONT, Pt(size), color
        f.bold = bold or k % 2 == 1
        f.italic = italic


def bullet(paragraph, indent: float = 0.26) -> None:
    ppr = paragraph._p.get_or_add_pPr()
    ppr.set("marL", str(int(Inches(indent))))
    ppr.set("indent", str(-int(Inches(indent))))
    for tag in ("a:buNone", "a:buChar", "a:buAutoNum"):
        for el in ppr.findall(qn(tag)):
            ppr.remove(el)
    ppr.append(ppr.makeelement(qn("a:buChar"), {"char": "\u2022"}))


def box(slide, x, y, w, h, paras=(), *, name=None, fill=None, line=None, shape=MSO_SHAPE.RECTANGLE,
        anchor=MSO_ANCHOR.TOP, margin=0.12, align=PP_ALIGN.LEFT, radius=0.08, wrap=True, line_width=1.0):
    """A shape with paragraphs. paras: (text, size, options) with options color, bold, italic, bullet,
    align, after (points). radius sets the corner of rounded rectangles and the point of chevrons."""
    if fill is None and line is None and shape == MSO_SHAPE.RECTANGLE:
        sp = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    else:
        sp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
        if shape in (MSO_SHAPE.ROUNDED_RECTANGLE, MSO_SHAPE.CHEVRON, MSO_SHAPE.PENTAGON):
            sp.adjustments[0] = radius
        if fill is None:
            sp.fill.background()
        else:
            sp.fill.solid()
            sp.fill.fore_color.rgb = fill
        if line is None:
            sp.line.fill.background()
        else:
            sp.line.color.rgb = line
            sp.line.width = Pt(line_width)
        sp.shadow.inherit = False
    tf = sp.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    for side in ("margin_left", "margin_right"):
        setattr(tf, side, Inches(margin))
    tf.margin_top = tf.margin_bottom = Inches(0.06)
    for i, (text, size, opts) in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = opts.get("align", align)
        if opts.get("bullet"):
            bullet(p)
        p.space_after = Pt(opts.get("after", 4))
        runs(p, text, size, opts.get("color", TEXT), opts.get("bold", False), opts.get("italic", False))
    if name:
        sp.name = name
    return sp


def picture(slide, path: Path, x, y, w=None, h=None, name=None):
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w) if w else None,
                                   Inches(h) if h else None)
    if name:
        pic.name = name
    return pic


def frame(slide, title: str, number: int) -> None:
    """Title bar, accent line, footer and slide number."""
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(W), Inches(0.95))
    bar.fill.solid()
    bar.fill.fore_color.rgb = DARK
    bar.line.fill.background()
    bar.shadow.inherit = False
    accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(0.95), Inches(W), Inches(0.06))
    accent.fill.solid()
    accent.fill.fore_color.rgb = MID
    accent.line.fill.background()
    accent.shadow.inherit = False
    box(slide, 0.5, 0.13, 11.6, 0.72, [(title, 28, {"color": WHITE, "bold": True})], anchor=MSO_ANCHOR.MIDDLE,
        margin=0.02)
    box(slide, 0.5, 7.07, 9.5, 0.3, [("AutoPilot-ML  |  Minor Project  |  Department of CSE, IIIT Bhopal", 11,
                                      {"color": GREY})], margin=0.02)
    box(slide, 11.33, 7.07, 1.5, 0.3, [(f"{number} / {TOTAL}", 11, {"color": GREY, "align": PP_ALIGN.RIGHT})],
        margin=0.02)


def notes(slide, text: str) -> None:
    slide.notes_slide.notes_text_frame.text = text


def card(slide, x, y, w, h, head: str, body: str, *, name=None, head_size=16, body_size=13.5, fill=PALE,
         anchor=MSO_ANCHOR.TOP):
    return box(slide, x, y, w, h, [(head, head_size, {"color": DARK, "bold": True, "after": 2}),
                                   (body, body_size, {"color": TEXT})],
               name=name, fill=fill, line=LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, margin=0.14, anchor=anchor)


def arrow(parent, points, *, head=True, color=ARROW, width=1.25):
    """A line through points (inches) of a slide or group, with an arrowhead at the last point."""
    emu = [(int(Inches(x)), int(Inches(y))) for x, y in points]
    if len(emu) == 2:
        (x0, y0), (x1, y1) = emu
        sp = parent.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x0, y0, x1, y1)
    else:
        builder = parent.shapes.build_freeform(emu[0][0], emu[0][1], scale=1.0)
        builder.add_line_segments(emu[1:], close=False)
        sp = builder.convert_to_shape()
        sp.fill.background()
        sp.shadow.inherit = False
    sp.line.color.rgb = color
    sp.line.width = Pt(width)
    if head:
        ln = sp.line._get_or_add_ln()
        ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    return sp


def tag(parent, x, y, w, text, *, align=PP_ALIGN.CENTER, size=12, h=0.26):
    """A small grey label next to an arrow."""
    return box(parent, x, y, w, h, [(text, size, {"color": GREY, "italic": True, "align": align})], margin=0.0,
               anchor=MSO_ANCHOR.MIDDLE, wrap=False)


def step(parent, x, y, w, h, paras, *, shape=MSO_SHAPE.ROUNDED_RECTANGLE, fill=PALE, color=DARK, size=12.5):
    """A flowchart node: paras is a list of lines; **...** marks bold parts."""
    return box(parent, x, y, w, h, [(text, size, {"color": color, "align": PP_ALIGN.CENTER, "after": 0})
                                    for text in paras],
               fill=fill, line=MID if fill != DARK else DARK, shape=shape, anchor=MSO_ANCHOR.MIDDLE, margin=0.04,
               radius=0.18, wrap=False)


# ----------------------------------------------------------------------------- figures for the slides


def make_figures() -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    logo = FIG_DIR / "iiit_bhopal_logo.png"
    with zipfile.ZipFile(TEMPLATE) as z:
        logo.write_bytes(z.read("word/media/image1.png"))
    plt.rcParams.update({"font.family": FONT, "font.size": 13, "axes.linewidth": 0.8})
    ranks = pd.read_csv(RESULTS / "analysis_ranks.csv")
    ranks = ranks[ranks.family == "main"].pivot_table(index="system", columns="budget_seconds",
                                                      values="average_rank")
    # slope chart of the average ranks (report Table 9) from 2 to 5 minutes, AutoPilot-ML full highlighted
    names = {"autopilot_full": "AutoPilot-ML full", "flaml": "FLAML alone", "autopilot_rules": "AutoPilot-ML rules",
             "autogluon": "AutoGluon", "manual": "LightGBM baseline"}
    style = {"autopilot_full": ("#1F4E79", 3.6, 9), "autopilot_rules": ("#5B9BD5", 2.0, 6)}
    fig, ax = plt.subplots(figsize=(6.6, 3.65), dpi=220)
    for s in ("manual", "autogluon", "flaml", "autopilot_rules", "autopilot_full"):  # highlighted line on top
        a, b = ranks.loc[s, 120], ranks.loc[s, 300]
        colour, width, size = style.get(s, ("#A6A6A6", 1.8, 5))
        ax.plot([0, 1], [a, b], color=colour, lw=width, marker="o", ms=size, zorder=3 if s in style else 2)
        ax.text(-0.05, a, f"{a:.2f}", ha="right", va="center", fontsize=11.5,
                color=colour if s in style else "#595959", fontweight="bold" if s == "autopilot_full" else None)
    by_value: dict[float, list[str]] = {}
    for s in names:
        by_value.setdefault(round(ranks.loc[s, 300], 2), []).append(s)
    for value, systems in by_value.items():
        colour = style.get(systems[0], ("#595959",))[0]
        ax.text(1.05, value, f"{value:.2f}  " + ", ".join(names[s] for s in systems), ha="left", va="center",
                fontsize=11.5, color=colour, fontweight="bold" if "autopilot_full" in systems else None)
    for x, text in ((0, "2 minutes"), (1, "5 minutes")):
        ax.plot([x, x], [1.32, 3.9], color="#D9D9D9", lw=1, zorder=1)
        ax.text(x, 1.02, text, ha="center", va="top", fontsize=12.5, color="#222222", fontweight="bold")
    gain = ranks.loc["autopilot_full", 120] - ranks.loc["autopilot_full", 300]
    ax.text(0.5, 4.08, f"AutoPilot-ML full: the largest gain of the five systems ({gain:.2f} ranks)",
            ha="center", va="center", fontsize=11.5, color="#1F4E79", style="italic")
    ax.set_xlim(-0.36, 2.15)
    ax.set_ylim(4.25, 1.0)
    ax.axis("off")
    ax.set_title("Average rank over the 6 datasets (1 is best)", fontsize=14, color="#1F4E79")
    fig.tight_layout()
    chart = FIG_DIR / "slide_ranks.png"
    fig.savefig(chart)
    plt.close(fig)
    return {"logo": logo, "ranks": chart}


# ----------------------------------------------------------------------------- slides


REFERENCES = [
    "C. Wang, Q. Wu, M. Weimer, and E. Zhu, \u201cFLAML: A fast and lightweight AutoML library,\u201d in Proc. "
    "Machine Learning and Systems (MLSys), vol. 3, 2021, pp. 434\u2013447.",
    "Microsoft, \u201cFLAML: A fast library for automated machine learning and tuning,\u201d source code, "
    "version 2.7.0, MIT licence. Accessed: Nov. 2026. [Online]. Available: https://github.com/microsoft/FLAML",
    "M. Feurer, A. Klein, K. Eggensperger, J. Springenberg, M. Blum, and F. Hutter, \u201cEfficient and robust "
    "automated machine learning,\u201d in Advances in Neural Information Processing Systems 28, 2015, "
    "pp. 2962\u20132970.",
    "R. S. Olson, N. Bartley, R. J. Urbanowicz, and J. H. Moore, \u201cEvaluation of a tree-based pipeline "
    "optimization tool for automating data science,\u201d in Proc. GECCO, 2016, pp. 485\u2013492.",
    "N. Erickson, J. Mueller, A. Shirkov, H. Zhang, P. Larroy, M. Li, and A. Smola, \u201cAutoGluon-Tabular: "
    "Robust and accurate AutoML for structured data,\u201d arXiv:2003.06505, 2020.",
    "N. Hollmann, S. M\u00fcller, and F. Hutter, \u201cLarge language models for automated data science: "
    "Introducing CAAFE for context-aware automated feature engineering,\u201d in Advances in Neural "
    "Information Processing Systems 36, 2023.",
    "C. Nadeau and Y. Bengio, \u201cInference for the generalization error,\u201d Machine Learning, vol. 52, "
    "no. 3, pp. 239\u2013281, 2003.",
    "W. R. Thompson, \u201cOn the likelihood that one unknown probability exceeds another in view of the "
    "evidence of two samples,\u201d Biometrika, vol. 25, no. 3/4, pp. 285\u2013294, 1933.",
    "P. Auer, N. Cesa-Bianchi, and P. Fischer, \u201cFinite-time analysis of the multiarmed bandit problem,"
    "\u201d Machine Learning, vol. 47, no. 2\u20133, pp. 235\u2013256, 2002.",
    "J. Vanschoren, J. N. van Rijn, B. Bischl, and L. Torgo, \u201cOpenML: Networked science in machine "
    "learning,\u201d ACM SIGKDD Explorations Newsletter, vol. 15, no. 2, pp. 49\u201360, 2013.",
]


def slide_title(prs, figs) -> None:
    s = prs.slides.add_slide(prs.slide_layouts[6])
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(6.55), Inches(W), Inches(0.95))
    band.fill.solid()
    band.fill.fore_color.rgb = DARK
    band.line.fill.background()
    band.shadow.inherit = False
    box(s, 0.5, 6.62, 12.33, 0.8, [("Department of Computer Science and Engineering, Indian Institute of "
                                    "Information Technology Bhopal  |  November 2026", 15,
                                    {"color": WHITE, "align": PP_ALIGN.CENTER})], anchor=MSO_ANCHOR.MIDDLE)
    logo = picture(s, figs["logo"], 0, 0.55, h=1.4)
    logo.left = int((Inches(W) - logo.width) / 2)
    box(s, 0.5, 2.1, 12.33, 0.85, [("AutoPilot-ML", 44, {"color": DARK, "bold": True, "align": PP_ALIGN.CENTER})],
        name="anim:01:fade", anchor=MSO_ANCHOR.MIDDLE)
    box(s, 0.5, 2.95, 12.33, 1.0,
        [("An Autonomous, Self-Optimizing Machine Learning Experimentation Framework", 24,
          {"color": TEXT, "align": PP_ALIGN.CENTER, "after": 6}),
         ("Minor Project Presentation", 18, {"color": MID, "bold": True, "align": PP_ALIGN.CENTER})],
        name="anim:02:fade")
    box(s, 2.0, 4.3, 9.33, 2.1,
        [("Presented by", 14, {"color": GREY, "align": PP_ALIGN.CENTER, "after": 2}),
         ("Priyanshu Verma (23U02016)     Salil Kushwah (23U02003)", 20,
          {"color": TEXT, "bold": True, "align": PP_ALIGN.CENTER, "after": 12}),
         ("Under the supervision of", 14, {"color": GREY, "align": PP_ALIGN.CENTER, "after": 2}),
         ("[Name of supervisor], [Designation of supervisor]", 18, {"color": TEXT, "align": PP_ALIGN.CENTER})],
        name="anim:03:fade")
    notes(s, "Time: about 15 seconds.\n\nGood morning. We are Priyanshu Verma and Salil Kushwah. Our minor project"
             " is AutoPilot-ML, an autonomous, self-optimizing machine learning experimentation framework, built "
             "on top of the open-source AutoML library FLAML.")


def slide_intro(prs) -> None:
    s = prs.slides.add_slide(prs.slide_layouts[6])
    frame(s, "Introduction", 2)
    box(s, 0.5, 1.4, 6.6, 5.45, [
        ("Most data used for prediction is **tabular**; gradient-boosted trees such as LightGBM and XGBoost "
         "are among the most accurate models for it.", 20, {"bullet": True, "after": 16}),
        ("**AutoML** libraries such as FLAML [1] search automatically for a model and its hyperparameters "
         "within a time budget.", 20, {"bullet": True, "after": 16}),
        ("The experimentation around the search is still **manual**: finding why a model is weak, choosing a "
         "change, judging whether it really helped, and deciding when to stop.", 20,
         {"bullet": True, "after": 16}),
        ("**AutoPilot-ML** automates this loop, offline on a laptop, with FLAML used unchanged as its search "
         "engine [1], [2].", 20, {"bullet": True}),
    ], name="anim:01:fade")
    # right: conventional AutoML against the AutoPilot-ML loop
    box(s, 7.45, 1.45, 5.4, 0.4, [("Conventional AutoML", 16, {"color": GREY, "bold": True})], margin=0.02)
    grp = s.shapes.add_group_shape()
    for k, label in enumerate(("Data", "Model search", "Model")):
        shape = MSO_SHAPE.PENTAGON if k == 0 else MSO_SHAPE.CHEVRON
        box(grp, 7.45 + k * 1.77, 1.92, 1.9, 0.8, [(label, 15, {"align": PP_ALIGN.CENTER})], fill=RGBColor(
            0xD9, 0xD9, 0xD9), shape=shape, anchor=MSO_ANCHOR.MIDDLE, margin=0.02, radius=0.3, wrap=False)
    grp.name = "anim:02:wipe"
    box(s, 7.45, 3.05, 5.4, 0.4, [("AutoPilot-ML", 16, {"color": DARK, "bold": True})], margin=0.02)
    grp2 = s.shapes.add_group_shape()
    for k, label in enumerate(("Diagnose", "Propose", "Search", "Judge")):
        shape = MSO_SHAPE.PENTAGON if k == 0 else MSO_SHAPE.CHEVRON
        box(grp2, 7.45 + k * 1.33, 3.52, 1.45, 0.8, [(label, 14, {"color": WHITE, "bold": True,
                                                                   "align": PP_ALIGN.CENTER})],
            fill=DARK, shape=shape, anchor=MSO_ANCHOR.MIDDLE, margin=0.02, radius=0.3, wrap=False)
    box(grp2, 7.45, 4.4, 5.4, 0.4, [("repeated until the time, the experiments or the proposals run out", 13,
                                     {"color": GREY, "italic": True, "align": PP_ALIGN.CENTER})], margin=0.02)
    grp2.name = "anim:03:wipe"
    box(s, 7.45, 5.05, 5.4, 1.6, [("Every decision is measured on 15 fixed folds of the training data and "
                                   "recorded; the test data is used only once, at the end.", 16,
                                   {"color": DARK})],
        name="anim:04:fade", fill=LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, anchor=MSO_ANCHOR.MIDDLE,
        margin=0.22)
    notes(s, "Time: about 30 seconds.\n\nMost data used for prediction is tabular, and gradient-boosted trees are "
             "among the most accurate models for it. AutoML libraries such as FLAML automate the search for a "
             "model and its settings, but the experimentation around that search is still manual: finding why a "
             "model is weak, choosing a change, judging whether it helped, and deciding when to stop. AutoPilot-ML"
             " automates this loop, offline, with FLAML as its search engine.")


def slide_literature(prs) -> None:
    s = prs.slides.add_slide(prs.slide_layouts[6])
    frame(s, "Literature Review / Background", 3)
    header = ["System", "Model search", "Changes the data", "Explains weaknesses", "Learns across datasets",
              "Budget across experiments"]
    rows = [
        ["Auto-sklearn [3]", "Yes", "Partly", "No", "Yes", "No"],
        ["TPOT [4]", "Yes", "Partly", "No", "No", "No"],
        ["AutoGluon-Tabular [5]", "Partly", "Partly", "No", "No", "No"],
        ["FLAML [1]", "Yes", "Partly", "No", "Partly", "Partly"],
        ["LLM agents [6]", "Partly", "Yes", "Partly", "No", "No"],
        ["AutoPilot-ML (this work)", "Yes", "Yes", "Yes", "Yes", "Yes"],
    ]
    shape = s.shapes.add_table(len(rows) + 1, len(header), Inches(0.5), Inches(1.3), Inches(12.33), Inches(3.6))
    shape.name = "anim:01:fade"
    table = shape.table
    widths = [2.83, 1.9, 1.9, 1.9, 1.9, 1.9]
    for c, w in enumerate(widths):
        table.columns[c].width = Inches(w)
    for r in range(len(rows) + 1):
        table.rows[r].height = Inches(0.5)
        for c in range(len(header)):
            cell = table.cell(r, c)
            text = header[c] if r == 0 else rows[r - 1][c]
            ours = r == len(rows)
            cell.fill.solid()
            cell.fill.fore_color.rgb = DARK if r == 0 else (LIGHT if ours else (PALE if r % 2 else WHITE))
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.08)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            colour = WHITE if r == 0 else (DARK if text == "Yes" or ours else (GREY if text == "No" else TEXT))
            runs(p, text, 14, colour, bold=r == 0 or ours or text == "Yes")
    box(s, 0.5, 4.95, 12.33, 0.5, [("Partly: only built-in preprocessing or operators, strong defaults instead of "
                                    "a search, defaults mined offline, decisions within one search, or code and "
                                    "text written by a language model without a measured check.", 12,
                                    {"color": GREY, "italic": True})], margin=0.02)
    box(s, 0.5, 5.55, 12.33, 1.35, [
        ("**Research gap:** no existing system diagnoses a model, changes the data, judges each change "
         "statistically and records the evidence, all offline and repeatable.", 16, {"color": DARK, "after": 6}),
        ("Building blocks: corrected resampled t-test [7]; Thompson sampling [8] and UCB1 [9]; OpenML "
         "datasets for the benchmark [10].", 14, {"color": TEXT}),
    ], name="anim:02:fade", fill=LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, anchor=MSO_ANCHOR.MIDDLE, margin=0.2)
    notes(s, "Time: about 25 seconds.\n\nAuto-sklearn, TPOT, AutoGluon and FLAML search models well, and some "
             "reuse settings across datasets. None of them explains why a model is weak or decides on its own to "
             "change the data. Language-model agents can propose changes, but without a measured, repeatable rule."
             " This is the gap we address, building on the corrected resampled t-test and on the bandit methods "
             "Thompson sampling and UCB1.")


def slide_problem(prs) -> None:
    s = prs.slides.add_slide(prs.slide_layouts[6])
    frame(s, "Problem Statement", 4)
    box(s, 0.5, 1.3, 12.33, 1.5, [
        ("Problem", 15, {"color": MID, "bold": True, "after": 2}),
        ("Given a tabular dataset D with a target y, a task T, a metric m and a compute budget B, find a "
         "pipeline p with the best score m(p) on unseen test data, while using as little of B as possible.", 19,
         {"color": TEXT}),
    ], name="anim:01:fade", fill=LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, anchor=MSO_ANCHOR.MIDDLE, margin=0.25)
    box(s, 0.5, 2.98, 12.33, 1.0, [
        ("Research question: can an autonomous experimentation layer make conventional AutoML more efficient "
         "and more effective, by diagnosing weaknesses, selecting strategies, learning from earlier experiments "
         "and allocating the budget adaptively?", 16, {"color": DARK, "italic": True}),
    ], name="anim:02:fade", margin=0.1)
    objectives = [
        ("Diagnose", "Find the weaknesses of a model automatically, with measured evidence"),
        ("Improve", "Propose changes and accept them only on statistical evidence"),
        ("Manage", "Allocate the time budget and learn from earlier experiments"),
        ("Evaluate", "Compare with FLAML, AutoGluon and a baseline on public data"),
    ]
    box(s, 0.5, 4.1, 12.33, 0.42, [("Objectives", 16, {"color": DARK, "bold": True})], margin=0.02)
    for k, (head, body) in enumerate(objectives):
        card(s, 0.5 + k * 3.13, 4.58, 2.93, 2.25, f"{k + 1}. {head}", body, name=f"anim:{k + 3:02d}:fade",
             head_size=19, body_size=17, anchor=MSO_ANCHOR.MIDDLE)
    notes(s, "Time: about 30 seconds.\n\nGiven a tabular dataset, a metric and a time budget, the system must find"
             " the pipeline with the best score on unseen test data, using as little of the budget as possible. "
             "Our objectives were to diagnose weaknesses, to accept changes only on statistical evidence, to "
             "manage the budget and learn from earlier experiments, and to evaluate the system against strong "
             "systems on public data.")


def slide_method(prs) -> None:
    s = prs.slides.add_slide(prs.slide_layouts[6])
    frame(s, "Proposed Methodology / Proposed Work", 5)
    picture(s, REPORT / "figures" / "fig_loop.png", 0.5, 1.3, w=7.25, name="anim:01:wipe")
    box(s, 0.5, 5.12, 7.25, 1.7, [
        ("Key principles", 15, {"color": DARK, "bold": True, "after": 3}),
        ("The test part (20% of the rows) stays locked until the final evaluation.", 14,
         {"bullet": True, "after": 2}),
        ("Every decision uses the same 15 folds of the training part (3 repeats of 5 folds).", 14,
         {"bullet": True, "after": 2}),
        ("FLAML performs the search inside each step and is not modified.", 14, {"bullet": True}),
    ], name="anim:07:fade", fill=PALE, line=LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, margin=0.18,
        anchor=MSO_ANCHOR.MIDDLE)
    parts = [
        ("Critic", "Diagnoses weaknesses from measured evidence: 14\u00a0kinds"),
        ("Strategy Engine", "Proposes changes that answer them: 12\u00a0strategies"),
        ("Judge", "Corrected t-test on 15 folds: keep, keep_simpler or discard"),
        ("Resource Manager", "Divides the time budget and decides when to stop"),
        ("Experiment Memory", "Records every change with its effect and cost"),
    ]
    for k, (head, body) in enumerate(parts):
        card(s, 8.05, 1.3 + k * 1.12, 4.78, 1.02, head, body, name=f"anim:{k + 2:02d}:fade", head_size=16,
             body_size=14, anchor=MSO_ANCHOR.MIDDLE)
    notes(s, "Time: about 35 seconds.\n\nThe figure shows the loop. FLAML searches a setup, and the Assessor "
             "refits the best configuration on 15 fixed folds. The Judge compares it with the accepted setup. If "
             "the change is kept, the Critic diagnoses the new setup, and the Strategy Engine proposes the next "
             "change, helped by the Experiment Memory. The Resource Manager gives every search its budget and "
             "decides when to stop. The test part stays locked until the end.")


def slide_flow(prs) -> None:
    s = prs.slides.add_slide(prs.slide_layouts[6])
    frame(s, "System Diagram / Proposed Algorithm / Flowchart", 6)
    # flowchart of one session (Figure 6 of the report, drawn for the slide): main column, then the loop
    cx, x0, w0, hh = 3.1, 0.95, 4.3, 0.42  # centre, left edge and width of the main column, node height
    y = {"start": 1.2, "split": 1.8, "first": 2.4, "assess": 3.0, "propose": 3.6, "stop": 4.2, "search": 4.94,
         "judge": 5.54, "write": 6.4}
    setup = s.shapes.add_group_shape()
    step(setup, x0, y["start"], w0, hh, ["Start: dataset, target, time budget"],
         shape=MSO_SHAPE.FLOWCHART_TERMINATOR, fill=DARK, color=WHITE)
    step(setup, x0, y["split"], w0, hh, ["**Split**: training part, test part, 15 folds"])
    step(setup, x0, y["first"], w0, hh, ["**Search the first setup** with 35% of the time"])
    step(setup, x0, y["assess"], w0, hh, ["**Assess** it, then the **Critic** diagnoses it"])
    for a, b in (("start", "split"), ("split", "first"), ("first", "assess")):
        arrow(setup, [(cx, y[a] + hh), (cx, y[b])])
    setup.name = "anim:01:fade"
    loop = s.shapes.add_group_shape()
    arrow(loop, [(cx, y["assess"] + hh), (cx, y["propose"])])
    step(loop, x0, y["propose"], w0, hh, ["**Propose** the changes not tried yet"])
    arrow(loop, [(cx, y["propose"] + hh), (cx, y["stop"])])
    step(loop, cx - 0.65, y["stop"], 1.3, 0.56, ["**Stop?**"], shape=MSO_SHAPE.FLOWCHART_DECISION, fill=WHITE)
    step(loop, 0.5, y["stop"] - 0.02, 1.6, 0.6, ["**Final evaluation**", "on the test part"],
         shape=MSO_SHAPE.FLOWCHART_TERMINATOR, fill=DARK, color=WHITE, size=12)
    arrow(loop, [(cx - 0.65, y["stop"] + 0.28), (2.1, y["stop"] + 0.28)])
    tag(loop, 2.1, y["stop"] - 0.04, 0.35, "yes")
    arrow(loop, [(cx, y["stop"] + 0.56), (cx, y["search"])])
    tag(loop, cx + 0.12, y["stop"] + 0.5, 0.3, "no", align=PP_ALIGN.LEFT, h=0.24)
    step(loop, x0, y["search"], w0, hh, ["**Search and assess** the candidate"])
    arrow(loop, [(cx, y["search"] + hh), (cx, y["judge"])])
    step(loop, cx - 0.75, y["judge"], 1.5, 0.58, ["**Judge**: kept?"], shape=MSO_SHAPE.FLOWCHART_DECISION,
         fill=WHITE)
    mid = y["judge"] + 0.29
    step(loop, 0.5, mid - 0.35, 1.45, 0.7, ["**Accept** it", "and diagnose it"], size=12)
    step(loop, 4.25, mid - 0.35, 1.45, 0.7, ["**Keep** the", "accepted setup"], size=12)
    arrow(loop, [(cx - 0.75, mid), (1.95, mid)])
    arrow(loop, [(cx + 0.75, mid), (4.25, mid)])
    tag(loop, 1.97, mid - 0.29, 0.36, "yes")
    tag(loop, cx + 0.77, mid - 0.29, 0.36, "no")
    step(loop, 0.5, y["write"], 5.2, hh, ["**Write** the step's record and the memory"])
    arrow(loop, [(1.225, mid + 0.35), (1.225, y["write"])])
    arrow(loop, [(4.975, mid + 0.35), (4.975, y["write"])])
    arrow(loop, [(5.7, y["write"] + hh / 2), (6.05, y["write"] + hh / 2), (6.05, y["propose"] + hh / 2),
                 (x0 + w0, y["propose"] + hh / 2)])
    tag(loop, 5.3, y["search"] + 0.08, 0.7, "next step", align=PP_ALIGN.RIGHT)
    loop.name = "anim:02:fade"
    box(s, 6.6, 1.2, 6.23, 2.75, [
        ("Judge: the decision rule for each change", 17, {"color": DARK, "bold": True, "after": 8}),
        ("t = mean(d) / \u221a((1/J + 1/4) \u00b7 var(d))", 18, {"color": TEXT, "align": PP_ALIGN.CENTER,
                                                                "after": 4}),
        ("d: the J = 15 fold differences, candidate minus accepted setup; p(better) and p(worse) from "
         "Student\u2019s t with 14 degrees of freedom", 12, {"color": GREY, "italic": True, "after": 8}),
        ("**keep**: mean(d) > 0 and p(better) \u2264 0.20", 15, {"bullet": True, "after": 3}),
        ("**keep_simpler**: simpler, mean(d) \u2265 \u22120.005 and p(worse) > 0.20", 15,
         {"bullet": True, "after": 3}),
        ("**discard**: in every other case", 15, {"bullet": True}),
    ], name="anim:03:fade", fill=PALE, line=LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, margin=0.22,
        anchor=MSO_ANCHOR.MIDDLE)
    box(s, 6.6, 4.12, 6.23, 2.73, [
        ("Resource Manager: budget and stopping", 17, {"color": DARK, "bold": True, "after": 8}),
        ("The first search receives 35% of the total time.", 15, {"bullet": True, "after": 4}),
        ("Each later search gets the time left, minus a reserve, shared over the experiments still allowed, "
         "and at least 5 seconds.", 15, {"bullet": True, "after": 4}),
        ("A session stops when the time, the experiments or the proposals run out, after three discards in a "
         "row, or on request.", 15, {"bullet": True}),
    ], name="anim:04:fade", fill=PALE, line=LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, margin=0.22,
        anchor=MSO_ANCHOR.MIDDLE)
    notes(s, "Time: about 30 seconds.\n\nThis flowchart is one session. After the split, the first setup is "
             "searched with 35 percent of the time, assessed and diagnosed. Each step then proposes a change, "
             "searches it and assesses it on the same folds. The Judge's corrected t-test keeps a change that is "
             "better, keeps a simpler one that is not worse, and discards the rest. When the session stops, the "
             "test part is scored once.")


def slide_implementation(prs) -> None:
    s = prs.slides.add_slide(prs.slide_layouts[6])
    frame(s, "Implementation", 7)
    picture(s, REPORT / "figures" / "fig_block.png", 0.5, 1.25, w=6.45, name="anim:01:fade")
    box(s, 7.3, 1.25, 5.53, 0.42, [("Technology stack", 16, {"color": DARK, "bold": True})], margin=0.02)
    grp = s.shapes.add_group_shape()
    stack = ["Python 3.11", "FLAML 2.7.0", "scikit-learn 1.9", "LightGBM, XGBoost", "SHAP 0.51", "MLflow 3.17",
             "Streamlit 1.65", "FastAPI 0.142", "Docker", "pytest 9.1"]
    for k, name in enumerate(stack):
        x, y = 7.3 + (k % 2) * 2.82, 1.75 + (k // 2) * 0.55
        box(grp, x, y, 2.7, 0.45, [(name, 14, {"color": DARK, "align": PP_ALIGN.CENTER})], fill=LIGHT,
            shape=MSO_SHAPE.ROUNDED_RECTANGLE, anchor=MSO_ANCHOR.MIDDLE, margin=0.05, radius=0.3)
    grp.name = "anim:02:fade"
    facts = [
        ("20,237", "lines of Python in 97 files"),
        ("586", "automated tests, all passing"),
        ("4", "ways to use it: dashboard, command line, local web API and Docker image"),
    ]
    for k, (num, text) in enumerate(facts):
        box(s, 7.3, 4.6 + k * 0.62, 5.53, 0.58, [(f"**{num}**  {text}", 15, {"color": TEXT})],
            name=f"anim:{k + 3:02d}:fade", margin=0.04, anchor=MSO_ANCHOR.MIDDLE)
    box(s, 7.3, 6.45, 5.53, 0.45, [("Runs offline on a laptop CPU (AMD Ryzen 9, no GPU).", 13,
                                    {"color": GREY, "italic": True})], name="anim:06:fade", margin=0.04)
    notes(s, "Time: about 25 seconds.\n\nThe diagram shows the layers, from the user interfaces through the "
             "autonomous loop down to the libraries, which are used unchanged. The code has about twenty thousand "
             "lines of Python. It can be used through a dashboard, a command line, a local web API or a Docker "
             "image, all 586 automated tests pass, and everything runs offline on a laptop.")


def slide_results(prs, figs) -> None:
    s = prs.slides.add_slide(prs.slide_layouts[6])
    frame(s, "Results, Comparison, and Analysis", 8)
    box(s, 0.5, 1.1, 12.33, 0.38, [("Benchmark: 390 runs on 6 OpenML datasets [10] with 5 seeds and budgets of 2 "
                                    "and 5 minutes; the cards give the results at 5 minutes.", 13,
                                    {"color": GREY, "italic": True})], margin=0.02)
    kpis = [
        ("29% less time", "212 s against 301 s per run, at the same score level as FLAML alone"),
        ("5 of 6 datasets", "improved from 2 to 5 minutes, the most of all systems "
                            "(FLAML\u00a0alone:\u00a02\u00a0of\u00a06)"),
        ("0.6218 F1", "the best of all systems on the imbalanced pc4 dataset (FLAML:\u00a00.5897)"),
        ("4 / 6 and 5 / 6", "datasets with a higher score than AutoGluon and the LightGBM baseline"),
    ]
    for k, (num, text) in enumerate(kpis):
        box(s, 0.5 + k * 3.13, 1.55, 2.93, 1.38, [(num, 24, {"color": DARK, "bold": True, "align": PP_ALIGN.CENTER,
                                                              "after": 2}),
                                                   (text, 12.5, {"color": TEXT, "align": PP_ALIGN.CENTER})],
            name=f"anim:{k + 1:02d}:fade", fill=PALE, line=LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
            anchor=MSO_ANCHOR.MIDDLE, margin=0.12)
    picture(s, figs["ranks"], 0.5, 3.1, w=6.6, name="anim:05:wipe")
    card(s, 7.35, 3.12, 5.48, 1.72, "Worked example: Titanic",
         "The Critic found overfitting and a misleading, high-cardinality Ticket column. The Judge kept "
         "dropping Ticket (p(better)\u00a0=\u00a00.010), and the test F1 rose from 0.6838 to 0.7244.",
         name="anim:06:fade", head_size=16, body_size=14, anchor=MSO_ANCHOR.MIDDLE)
    card(s, 7.35, 5.0, 5.48, 1.75, "Reliability on data with known answers",
         "Critic: no false alarm in its six data-level kinds of diagnosis. Judge: kept none of 18 harmful "
         "changes and found 10 of 12 helpful ones.", name="anim:07:fade", head_size=16, body_size=14,
         anchor=MSO_ANCHOR.MIDDLE)
    notes(s, "Time: about 50 seconds.\n\nIn the benchmark, at five minutes, AutoPilot-ML matched the score level "
             "of FLAML with 29 percent less time, ranked second of five systems, scored higher than AutoGluon on "
             "four datasets and than the LightGBM baseline on five, and gave the best score on pc4. The chart "
             "shows that it gains the most from more time, moving from fourth to second place. On Titanic, it found by "
             "itself that the Ticket column was misleading, and the test F1 rose from 0.684 to 0.724. And on data "
             "with known answers, the Critic raised no false data-level alarm, and the Judge kept none of 18 "
             "harmful changes.")


def slide_conclusion(prs) -> None:
    s = prs.slides.add_slide(prs.slide_layouts[6])
    frame(s, "Conclusion and Future Work", 9)
    box(s, 0.5, 1.3, 6.0, 4.6, [
        ("Conclusion", 20, {"color": DARK, "bold": True, "after": 10}),
        ("A complete autonomous experimentation layer around FLAML, used unchanged: 14 kinds of diagnosis, "
         "12 strategies, a statistical Judge, a Resource Manager and an Experiment Memory.", 17,
         {"bullet": True, "after": 10}),
        ("Reliable on data with known answers: no false data-level alarm from the Critic, and none of 18 "
         "harmful changes kept by the Judge.", 17, {"bullet": True, "after": 10}),
        ("At 5 minutes: the score level of FLAML with 29% less time, second of five systems, and the best score "
         "on pc4.", 17, {"bullet": True, "after": 10}),
        ("Every decision explained, recorded and repeatable; fully offline, with 586 automated tests.", 17,
         {"bullet": True}),
    ], name="anim:01:fade", fill=PALE, line=LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, margin=0.28, radius=0.05)
    box(s, 6.83, 1.3, 6.0, 4.6, [
        ("Future work", 20, {"color": DARK, "bold": True, "after": 10}),
        ("A larger first search for short budgets, so that the gains seen at 5 minutes also appear at "
         "2 minutes.", 17, {"bullet": True, "after": 10}),
        ("Use the time saved (about 88 seconds of a 5-minute run) for a final, longer search of the accepted "
         "setup.", 17, {"bullet": True, "after": 10}),
        ("A larger memory and benchmark, with more datasets and longer budgets.", 17,
         {"bullet": True, "after": 10}),
        ("New kinds of diagnosis, such as outliers, skewed columns and data drift.", 17,
         {"bullet": True, "after": 10}),
        ("Language models as proposers of new changes, with the Judge kept as the gate.", 17, {"bullet": True}),
    ], name="anim:02:fade", fill=PALE, line=LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, margin=0.28, radius=0.05)
    box(s, 0.5, 6.08, 12.33, 0.78, [("AutoPilot-ML turns one AutoML search into an explained, measured and "
                                     "repeatable experimentation loop.", 19,
                                     {"color": WHITE, "bold": True, "align": PP_ALIGN.CENTER})],
        name="anim:03:fade", fill=DARK, shape=MSO_SHAPE.ROUNDED_RECTANGLE, anchor=MSO_ANCHOR.MIDDLE,
        margin=0.2, radius=0.2)
    notes(s, "Time: about 30 seconds.\n\nTo conclude, we built a complete, tested and offline experimentation "
             "layer whose every decision is explained and repeatable, and which, at five minutes, matches the "
             "score level of FLAML in less time. Next steps include a larger first search for short budgets, using"
             " the saved time, a larger memory and benchmark, new diagnoses, and language models as proposers, "
             "with the Judge as the gate.")


def slide_references(prs) -> None:
    s = prs.slides.add_slide(prs.slide_layouts[6])
    frame(s, "References", 10)
    box(s, 0.5, 1.2, 12.33, 5.1, [(f"[{k}]  {ref}", 14, {"color": TEXT, "after": 6})
                                   for k, ref in enumerate(REFERENCES, 1)], margin=0.05)
    box(s, 0.5, 6.38, 12.33, 0.6, [("Thank you. We welcome your questions.", 20,
                                    {"color": DARK, "bold": True, "align": PP_ALIGN.CENTER})],
        name="anim:01:fade", anchor=MSO_ANCHOR.MIDDLE)
    notes(s, "Time: about 5 seconds.\n\nThese are our main references. Thank you; we welcome your questions.")


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-office", action="store_true", help="do not open PowerPoint (no animations, no PDF)")
    args = ap.parse_args()
    if OUT_PPTX.exists():
        try:
            with open(OUT_PPTX, "ab"):
                pass
        except PermissionError:
            raise SystemExit(f"{OUT_PPTX} is open in PowerPoint; close it first")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figs = make_figures()
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    slide_title(prs, figs)
    slide_intro(prs)
    slide_literature(prs)
    slide_problem(prs)
    slide_method(prs)
    slide_flow(prs)
    slide_implementation(prs)
    slide_results(prs, figs)
    slide_conclusion(prs)
    slide_references(prs)
    prs.core_properties.title = "AutoPilot-ML: minor project presentation"
    prs.core_properties.author = "Priyanshu Verma, Salil Kushwah"
    prs.save(OUT_PPTX)
    print(f"written {OUT_PPTX}: {len(prs.slides)} slides")
    if not args.no_office:
        script = HERE / "ppt_finalize.ps1"
        out = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
                              "-InPath", str(OUT_PPTX), "-PdfPath", str(OUT_PPTX.with_suffix(".pdf"))],
                             capture_output=True, text=True, timeout=600)
        if out.returncode != 0:
            raise SystemExit("PowerPoint step failed:\n" + out.stdout[-2000:] + out.stderr[-2000:])
        print("PowerPoint:", out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "done")


if __name__ == "__main__":
    main()
