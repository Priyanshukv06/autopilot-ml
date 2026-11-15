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

- A4, 2.54 cm margins on all four sides in every section; page numbers at the bottom centre of every page except the cover: i to vii on the front pages (certificate to list of tables) and 1 onwards from the Abstract (the template's numbering).
- Times New Roman everywhere, also in the code listing: chapter headings 16 pt bold and all section headings 14 pt bold, left aligned; body text 12 pt justified; table text 11 pt, justified or centred; captions 11 pt centred (figure captions below the figure, table captions above the table, no full stop); references 10 pt in IEEE style. Figures are drawn at their printed width, with component names and labels at 12 pt and other text at 10 to 11 pt.
- Line spacing 1.15 everywhere; spacing between paragraphs comes from paragraph spacing, never from empty lines.
- Figures, tables and equations numbered in one sequence each; every figure and table is introduced by number in the paragraph next to it ("Table 3 lists ..."), with what it shows and what it means.
- Every paragraph of chapters 1 and 2 has a citation; 34 references, all cited, in IEEE style (page ranges and citation ranges with an en dash, access dates on online sources).
- No underlining, no em dashes, no emojis; component names (Critic, Judge, Strategy Engine and the others) are the only capitalised terms; a negative number is printed with a minus sign that never breaks away from its digits.
- Long tables may break across pages with the header row repeated; short tables, figures, equations and algorithms stay whole. The layout leaves no blank space larger than about 2 cm inside a chapter.

## Status (15 November 2026)

- [x] Builder and checks
- [x] Figures
- [x] Front pages and abstract, with all names filled in: MAJOR PROJECT REPORT on the cover, the supervisor Dr. Gagan Vishwakarma (Assistant Professor, Department of CSE, IIIT Bhopal) and the Director Dr. Ashutosh Kumar Singh
- [x] Chapters 1 to 9 and references
- [x] Build, Word update, PDF, visual check: 47 pages (8 front pages and 39 numbered pages), 9 figures, 9 tables, 9 equations, 3 algorithms, 34 references, 0 format problems
- [x] Report notes work log and commit

The report is the condensed version: it leads with what was delivered, keeps only the main tables and figures, and has no dates or timeline in the chapters. Chapter lengths against the template's ranges: introduction 2.3 pages, literature review 6.4, problem and objectives 1.4, methodology 7.0, algorithms 1.4, diagrams 2.9, implementation 5.0, results 5.0, conclusion 2.0 and references 1.7 (shorter than the template's 2 to 5 pages because the list was cut to at most 35 entries). The secondary studies left out of it (the ablation variants, time-to-target, the Judge agreement study and the hypothesis outcomes) stay in `docs/imp/report-notes.md` and `results/`. The figure functions that are no longer used (`fig_split`, `fig_judge`, `fig_diff`, `fig_cd`) and their images are kept in case a figure is wanted back.

Still to do by the team: the signatures and the certificate's date on the printed copy, the plagiarism report from the institute's tool (the template asks for content plagiarism below 12%, placed after the references), and a page "List of publications" only if a paper is published, accepted or submitted. The template file keeps the department's name "Minor Project format"; that name is not printed anywhere in the report.

## Deliberate choices where the template and the written rules differ

- Chapter and section headings are left aligned, as the written rules say; the template's sample chapter titles are centred. Front-page titles (Certificate, Abstract and the others) stay centred as in the template. To centre the chapter headings, set `HEADING_ALIGN` in the builder.
- All sections have 2.54 cm margins on every side (the template's cover page had 3 cm on the left), because the rules ask for equal margins and one layout throughout.
- The front pages carry roman page numbers, because the written rules ask for a page number on every page; the template left them unnumbered. Arabic numbering still starts at 1 on the Abstract, as in the template.
- Equations use Word's equation font (Cambria Math): in Times New Roman, Word draws roots, primes and floor brackets wrongly (tested on 14 November 2026). All other text, including the code listing, is in Times New Roman.
- Grammar fixes in the template's declaration text ("is presented in partial fulfillment", "for the award of"), "Bachelor of Technology" in the certificate as on the cover, and the academic year 2026-27 for a report dated November 2026.
