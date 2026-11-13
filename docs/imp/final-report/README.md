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

## Status (13 November 2026)

- [x] Builder and checks
- [x] Figures
- [x] Front pages and abstract
- [x] Chapters 1 to 9 and references
- [x] Build, Word update, PDF, visual check: 52 pages, 9 figures, 9 tables, 9 equations, 3 algorithms, 77 references, 0 format problems
- [x] Report notes work log and commit

The report is the condensed version: it leads with what was delivered, keeps only the main tables and figures, and has no dates or timeline in the chapters. Every chapter is inside the page range the template gives (for example 5.0 pages of implementation and 5.1 of results against "5-10 pages"). The secondary studies left out of it (the ablation variants, time-to-target, the Judge agreement study and the hypothesis outcomes) stay in `docs/imp/report-notes.md` and `results/`. The figure functions that are no longer used (`fig_split`, `fig_judge`, `fig_diff`, `fig_cd`) and their images are kept in case a figure is wanted back.

Still to do by the team: the guide's name and designation in `front.toml` (then build again), the signatures, the plagiarism report from the institute's tool, and a page "List of publications" only if a paper is published or submitted.

## Deliberate choices where the template and the written rules differ

- Chapter and section headings are left aligned, as the written rules say; the template's sample chapter titles are centred. Front-page titles (Certificate, Abstract and the others) stay centred as in the template. To centre the chapter headings, set `HEADING_ALIGN` in the builder.
- All sections have 2.54 cm margins on every side (the template's cover page had 3 cm on the left), because the rules ask for equal margins and one layout throughout.
- Page numbering starts at 1 on the Abstract, as in the template; the pages before it carry no number.
- Equations use Word's equation font (Cambria Math), because Word only draws brackets and roots correctly in a math font. The code listing uses Courier New, so that the code stays readable.
- Grammar fixes in the template's declaration text ("is presented in partial fulfillment", "for the award of"), "Bachelor of Technology" in the certificate as on the cover, and the academic year 2026-27 for a report dated November 2026.
