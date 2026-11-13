# Final report: sources and build

The final project report is built from the files in this folder by `scripts/report/build_report.py`, in the department's format (`template/Minor Project format - final 2026.docx`).

## Files

- `front.toml`: names, title, supervisor, dates and other details for the front pages.
- `chapters/*.md`: the text, one file per part, in reading order. The syntax is described at the top of `scripts/report/build_report.py`.
- `references.txt`: the reference list. Only the entries cited in the text are printed, numbered in order of first citation (IEEE style).
- `figures/`: the figures, made by `scripts/report/report_figures.py` from `results/` (charts) or drawn by it (diagrams).
- Output: `AutoPilot-ML-Final-Report.docx` and `AutoPilot-ML-Final-Report.pdf`.

## Build

From `autopilot-ml/`, with the `automl` environment (needs `python-docx==1.2.0`) and Microsoft Word installed:

```
python scripts/report/report_figures.py
python scripts/report/build_report.py
```

The second command writes the .docx, asks Word to fill in the page numbers of the contents, figure and table lists, saves it, exports the PDF and runs the format checks.

## Format rules applied

- A4, 2.54 cm margins on all four sides in every section; page numbers at the bottom centre from the Abstract onwards (the template's numbering), none on the cover and front pages.
- Times New Roman everywhere: chapter headings 16 pt bold, section headings 14 pt bold, both left aligned; body text 12 pt justified; table text 11 pt; captions 11 pt centred (figure captions below the figure, table captions above the table, no full stop); references 10 pt in IEEE style.
- Line spacing 1.15 everywhere; spacing between paragraphs comes from paragraph spacing, never from empty lines.
- Figures, tables and equations numbered in one sequence each through the report, each referred to in the text.
- No underlining, no em dashes, no emojis.

## Status

- [ ] Builder and checks
- [ ] Figures
- [ ] Front pages and abstract
- [ ] Chapters 1 to 9 and references
- [ ] Build, Word update, PDF, visual check
- [ ] Report notes work log and commit
