"""Build the final project report (.docx) in the department's format.

Run from autopilot-ml/ with the automl environment:
    python scripts/report/build_report.py            # .docx, then Word fills page numbers, saves, exports PDF
    python scripts/report/build_report.py --no-word  # .docx only

Sources in docs/imp/final-report/: front.toml, chapters/*.md (read in name order),
references.txt and figures/. Needs python-docx (1.2.0); the Word step needs Microsoft Word.

Chapter file syntax. Blocks are separated by blank lines.
  # Title                       chapter (numbered, printed in capitals; "# Title {-}" is unnumbered)
  ## Title / ### Title          section / subsection (numbered)
  - text                        bullet item, one per line (continuation lines are joined)
  1. text                       numbered item
  ::figure ID | PATH | WIDTH_CM | Caption
  ::table ID | Caption | widths=3,2,2 | align=jcc | width=CM     pipe rows (first row = header), then ::end
  ::equation ID                 an equation from equations.py
  ::algorithm ID | Title        lines "Input: ..." / "Output: ..." or "LEVEL text" (LEVEL 1 to 6), then ::end
  ::code ID | Caption           code lines, then ::end (numbered as a figure)
  anything else                 a paragraph
Inline: **bold**, *italic*, `code`, citations [@key] or [@k1; @k2], and @fig:ID, @tab:ID, @eq:ID,
@alg:ID, which print the number of that item.
The files 00-abstract.md, 00-acknowledgement.md and 00-area-of-work.md hold the text of those pages;
{name} fields in them come from front.toml.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import re
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.text.paragraph import Paragraph

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from equations import EQUATIONS  # noqa: E402
from omml import omath_para  # noqa: E402

ROOT = HERE.parents[1]
SRC = ROOT / "docs" / "imp" / "final-report"
OUT_DOCX = SRC / "AutoPilot-ML-Final-Report.docx"
OUT_PDF = SRC / "AutoPilot-ML-Final-Report.pdf"
TEMPLATE = SRC / "template" / "Minor Project format - final 2026.docx"

FONT = "Times New Roman"
CODE_FONT = "Courier New"
MARGIN_CM = 2.54
TEXT_WIDTH_CM = 21.0 - 2 * MARGIN_CM
LINE = 1.15
BODY_PT, TABLE_PT, CAPTION_PT, REF_PT, CODE_PT = 12, 11, 11, 10, 10
H1_PT, H2_PT, H3_PT = 16, 14, 12
HEADING_ALIGN = WD_ALIGN_PARAGRAPH.LEFT  # the department's rule: headings left aligned
JUSTIFY, CENTER = WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.CENTER
LEFT, RIGHT = WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT
ALIGNS = {"j": JUSTIFY, "c": CENTER, "l": LEFT, "r": RIGHT}


# ----------------------------------------------------------------------------- source parsing


@dataclass
class Block:
    kind: str
    text: str = ""
    level: int = 0
    items: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    opts: dict = field(default_factory=dict)
    source: str = ""


DIRECTIVE = re.compile(r"^::(\w+)\s*(.*)$")


def parse_blocks(text: str, source: str) -> list[Block]:
    lines = text.splitlines()
    blocks: list[Block] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("<!--"):
            i += 1
            continue
        m = DIRECTIVE.match(stripped)
        if m:
            name, rest = m.group(1), m.group(2)
            parts = [p.strip() for p in rest.split("|")] if rest else []
            body: list[str] = []
            i += 1
            if name in ("table", "algorithm", "code"):
                while i < len(lines) and lines[i].strip() != "::end":
                    body.append(lines[i])
                    i += 1
                if i >= len(lines):
                    raise SystemExit(f"{source}: ::{name} {rest} has no ::end")
                i += 1
            blocks.append(make_directive(name, parts, body, source))
            continue
        chunk = []
        while i < len(lines) and lines[i].strip() and not DIRECTIVE.match(lines[i].strip()):
            chunk.append(lines[i])
            i += 1
        blocks.extend(text_blocks(chunk, source))
    return blocks


def text_blocks(chunk: list[str], source: str) -> list[Block]:
    first = chunk[0].strip()
    h = re.match(r"^(#{1,3})\s+(.*)$", first)
    if h:
        title, numbered = h.group(2).strip(), True
        if title.endswith("{-}"):
            title, numbered = title[:-3].strip(), False
        out = [Block("heading", text=title, level=len(h.group(1)), opts={"numbered": numbered}, source=source)]
        if len(chunk) > 1:
            out.extend(text_blocks(chunk[1:], source))
        return out
    for kind, pattern in (("bullets", r"^- (.*)$"), ("numbers", r"^\d+\. (.*)$")):
        if re.match(pattern, first):
            items: list[str] = []
            for ln in chunk:
                m = re.match(pattern, ln.strip())
                if m and not ln.startswith("  "):
                    items.append(m.group(1).strip())
                else:
                    items[-1] += " " + ln.strip()
            return [Block(kind, items=items, source=source)]
    return [Block("para", text=" ".join(ln.strip() for ln in chunk), source=source)]


def make_directive(name: str, parts: list[str], body: list[str], source: str) -> Block:
    if name == "figure":
        if len(parts) != 4:
            raise SystemExit(f"{source}: ::figure needs ID | PATH | WIDTH | Caption: {parts}")
        return Block("figure", text=parts[3], opts={"id": parts[0], "path": parts[1], "width": float(parts[2])},
                     source=source)
    if name == "table":
        opts = {"id": parts[0]}
        for p in parts[2:]:
            key, _, val = p.partition("=")
            opts[key.strip()] = val.strip()
        rows = []
        for ln in body:
            s = ln.strip()
            if not s or re.match(r"^\|?\s*:?-{3,}", s):
                continue
            rows.append([c.strip() for c in s.strip("|").split("|")])
        return Block("table", text=parts[1], rows=rows, opts=opts, source=source)
    if name == "equation":
        if parts[0] not in EQUATIONS:
            raise SystemExit(f"{source}: unknown equation {parts[0]}")
        return Block("equation", opts={"id": parts[0]}, source=source)
    if name in ("algorithm", "code"):
        return Block(name, text=parts[1], items=body, opts={"id": parts[0]}, source=source)
    raise SystemExit(f"{source}: unknown directive ::{name}")


def load_references(path: Path) -> dict[str, str]:
    refs: dict[str, str] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        key, sep, text = line.partition("|")
        if not sep:
            raise SystemExit(f"references.txt line {n}: expected 'key | text'")
        key = key.strip()
        if key in refs:
            raise SystemExit(f"references.txt: duplicate key {key}")
        refs[key] = text.strip()
    return refs


# ----------------------------------------------------------------------------- numbering


CITE = re.compile(r"\[@([^\]]+)\]")
XREF = re.compile(r"@(fig|tab|eq|alg):[\w-]+")


class Numbers:
    def __init__(self, blocks: list[Block], refs: dict[str, str]):
        self.items = {"fig": {}, "tab": {}, "eq": {}, "alg": {}}
        self.refs = refs
        self.cite_order: list[str] = []
        self.used: set[str] = set()
        for b in blocks:
            kind = {"figure": "fig", "code": "fig", "table": "tab", "equation": "eq", "algorithm": "alg"}.get(b.kind)
            if kind:
                ident = b.opts["id"]
                if not ident.startswith(kind + ":"):
                    raise SystemExit(f"{b.source}: {b.kind} ID {ident} must start with {kind}:")
                if ident in self.items[kind]:
                    raise SystemExit(f"{b.source}: duplicate ID {ident}")
                self.items[kind][ident] = len(self.items[kind]) + 1
            for text in block_texts(b):
                for m in CITE.finditer(text):
                    for key in cite_keys(m.group(1)):
                        if key not in refs:
                            raise SystemExit(f"{b.source}: citation of unknown reference {key}")
                        if key not in self.cite_order:
                            self.cite_order.append(key)

    def number(self, ident: str) -> int:
        kind = ident.split(":")[0]
        if ident not in self.items[kind]:
            raise SystemExit(f"unknown cross-reference {ident}")
        return self.items[kind][ident]


def cite_keys(inner: str) -> list[str]:
    return [k.strip().lstrip("@").strip() for k in inner.split(";") if k.strip()]


def block_texts(b: Block) -> list[str]:
    if b.kind in ("para", "heading"):
        return [b.text]
    if b.kind in ("bullets", "numbers"):
        return list(b.items)
    if b.kind == "table":
        return [b.text] + [c for row in b.rows for c in row]
    if b.kind in ("figure", "code"):
        return [b.text]
    if b.kind == "algorithm":
        return [b.text] + list(b.items)
    return []


def format_cites(numbers: list[int]) -> str:
    numbers = sorted(set(numbers))
    groups, start, prev = [], numbers[0], numbers[0]
    for n in numbers[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        if prev - start >= 2:
            groups.append(f"[{start}]-[{prev}]")
        else:
            groups.extend(f"[{k}]" for k in range(start, prev + 1))
        if n is not None:
            start = prev = n
    return ", ".join(groups)


def smart_quotes(s: str) -> str:
    s = re.sub(r'(^|[\s(\[{/])"', "\\1\u201c", s)
    s = s.replace('"', "\u201d")
    s = re.sub(r"(^|[\s(\[{/])'", "\\1\u2018", s)
    return s.replace("'", "\u2019")


def resolve(text: str, nums: Numbers) -> str:
    def cite(m):
        return format_cites([nums.cite_order.index(k) + 1 for k in cite_keys(m.group(1))])

    def xref(m):
        nums.used.add(m.group(0)[1:])
        return str(nums.number(m.group(0)[1:]))

    text = CITE.sub(cite, text)
    text = XREF.sub(xref, text)
    return smart_quotes(text)


# ----------------------------------------------------------------------------- low-level helpers


INLINE = re.compile(r"(\*\*.+?\*\*|\*[^*\s][^*]*?\*|`[^`]+`|\^\{[^}]+\}|_\{[^}]+\})")
_bookmark_ids = itertools.count(1)
TBLPR_ORDER = ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize", "tblStyleColBandSize",
               "tblW", "jc", "tblCellSpacing", "tblInd", "tblBorders", "shd", "tblLayout", "tblCellMar", "tblLook"]


def set_fonts(rpr, font: str = FONT) -> None:
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rfonts.get(qn(attr)) is not None:
            del rfonts.attrib[qn(attr)]
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), font)


def style_run(run, size: float, bold: bool = False, italic: bool = False, font: str = FONT) -> None:
    run.font.size = Pt(size)
    if bold:
        run.font.bold = True
    if italic:
        run.font.italic = True
    set_fonts(run._r.get_or_add_rPr(), font)


def add_runs(par, text: str, size: float = BODY_PT, bold: bool = False, italic: bool = False,
             font: str = FONT) -> None:
    for piece in INLINE.split(text):
        if not piece:
            continue
        b, it, script = bold, italic, None
        if piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
            piece, b = piece[2:-2], True
        elif piece.startswith("`") and piece.endswith("`"):
            piece = piece[1:-1]
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
            piece, it = piece[1:-1], True
        elif piece[:2] in ("^{", "_{") and piece.endswith("}"):
            piece, script = piece[2:-1], piece[0]
        run = par.add_run(piece)
        style_run(run, size, b, it, font)
        if script == "^":
            run.font.superscript = True
        elif script == "_":
            run.font.subscript = True


def fmt(par, align=JUSTIFY, before: float = 0, after: float = 6, keep_next: bool = False,
        left: float | None = None, first: float | None = None, tab: float | None = None,
        page_break: bool = False, line: float = LINE) -> None:
    pf = par.paragraph_format
    pf.alignment = align
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line
    pf.widow_control = True
    if keep_next:
        pf.keep_with_next = True
    if page_break:
        pf.page_break_before = True
    if left is not None:
        pf.left_indent = Cm(left)
    if first is not None:
        pf.first_line_indent = Cm(first)
    if tab is not None:
        pf.tab_stops.add_tab_stop(Cm(tab))


def add_bookmark(par, name: str) -> None:
    ident = str(next(_bookmark_ids))
    start = OxmlElement("w:bookmarkStart")
    start.set(qn("w:id"), ident)
    start.set(qn("w:name"), name)
    end = OxmlElement("w:bookmarkEnd")
    end.set(qn("w:id"), ident)
    ppr = par._p.pPr
    if ppr is not None:
        ppr.addnext(start)
    else:
        par._p.insert(0, start)
    par._p.append(end)


def add_field(par, instr: str, cached: str, size: float, bold: bool = False) -> None:
    def fld(kind):
        run = par.add_run()
        style_run(run, size, bold)
        el = OxmlElement("w:fldChar")
        el.set(qn("w:fldCharType"), kind)
        run._r.append(el)

    fld("begin")
    run = par.add_run()
    style_run(run, size, bold)
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = f" {instr} "
    run._r.append(it)
    fld("separate")
    style_run(par.add_run(cached), size, bold)
    fld("end")


def tblpr_set(table, el) -> None:
    tblpr = table._tbl.tblPr
    name = el.tag.split("}")[1]
    old = tblpr.find(qn("w:" + name))
    if old is not None:
        tblpr.remove(old)
    pos = TBLPR_ORDER.index(name)
    for i, child in enumerate(list(tblpr)):
        child_name = child.tag.split("}")[1]
        if child_name in TBLPR_ORDER and TBLPR_ORDER.index(child_name) > pos:
            child.addprevious(el)
            return
    tblpr.append(el)


def set_borders(table, top=4, bottom=4, left=4, right=4, inside_h=4, inside_v=4) -> None:
    borders = OxmlElement("w:tblBorders")
    for name, size in (("top", top), ("left", left), ("bottom", bottom), ("right", right),
                       ("insideH", inside_h), ("insideV", inside_v)):
        el = OxmlElement(f"w:{name}")
        if size:
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), str(size))
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), "000000")
        else:
            el.set(qn("w:val"), "nil")
        borders.append(el)
    tblpr_set(table, borders)


def setup_table(table, widths_cm: list[float], align=WD_TABLE_ALIGNMENT.CENTER, margin_lr: float = 0.12,
                margin_tb: float = 0.05) -> None:
    table.alignment = align
    table.autofit = False
    total = OxmlElement("w:tblW")
    total.set(qn("w:w"), str(int(sum(widths_cm) * 567)))
    total.set(qn("w:type"), "dxa")
    tblpr_set(table, total)
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblpr_set(table, layout)
    mar = OxmlElement("w:tblCellMar")
    for side, val in (("top", margin_tb), ("left", margin_lr), ("bottom", margin_tb), ("right", margin_lr)):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), str(int(val * 567)))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tblpr_set(table, mar)
    grid = table._tbl.tblGrid
    for col, width in zip(grid.findall(qn("w:gridCol")), widths_cm):
        col.set(qn("w:w"), str(int(width * 567)))
    for row in table.rows:
        for cell, width in zip(row.cells, widths_cm):
            cell.width = Cm(width)


def row_flags(row, header: bool = False) -> None:
    trpr = row._tr.get_or_add_trPr()
    cant = OxmlElement("w:cantSplit")
    trpr.append(cant)
    if header:
        hdr = OxmlElement("w:tblHeader")
        trpr.append(hdr)


def cell_par(cell, align=JUSTIFY, before: float = 0, after: float = 0, first: bool = True):
    par = cell.paragraphs[0] if first else cell.add_paragraph()
    fmt(par, align, before, after)
    return par


# ----------------------------------------------------------------------------- the document


class Report:
    def __init__(self, front: dict, blocks: list[Block], special: dict[str, list[Block]], nums: Numbers):
        self.front = front
        self.blocks = blocks
        self.special = special
        self.nums = nums
        self.doc = Document()
        self.pending_before = 0.0  # extra space before the next paragraph (after a table)
        self.chapter = self.section = self.subsection = 0
        self.toc: list[tuple[str, str, str]] = []  # (S.no, title, bookmark)
        self.figures: list[tuple[int, str]] = []
        self.tables: list[tuple[int, str]] = []
        self.logo = self._extract_logo()

    # -- setup

    def _extract_logo(self) -> Path:
        out = Path(tempfile.gettempdir()) / "autopilot_report_logo.png"
        with zipfile.ZipFile(TEMPLATE) as z:
            out.write_bytes(z.read("word/media/image1.png"))
        return out

    def setup(self) -> None:
        doc = self.doc
        body = doc.element.body
        for p in list(body.iterchildren(qn("w:p"))):
            body.remove(p)
        styles = doc.styles.element
        rpr_default = styles.find(qn("w:docDefaults")).find(qn("w:rPrDefault")).find(qn("w:rPr"))
        set_fonts(rpr_default)
        for tag in ("w:sz", "w:szCs"):
            el = rpr_default.find(qn(tag))
            if el is None:
                el = OxmlElement(tag)
                rpr_default.append(el)
            el.set(qn("w:val"), str(BODY_PT * 2))
        lang = rpr_default.find(qn("w:lang"))
        if lang is None:
            lang = OxmlElement("w:lang")
            rpr_default.append(lang)
        lang.set(qn("w:val"), "en-GB")
        normal = doc.styles["Normal"]
        normal.font.size = Pt(BODY_PT)
        set_fonts(normal.element.get_or_add_rPr())
        pf = normal.paragraph_format
        pf.alignment, pf.line_spacing = JUSTIFY, LINE
        pf.space_before, pf.space_after, pf.widow_control = Pt(0), Pt(6), True
        for name, size, before, after in (("Heading 1", H1_PT, 0, 12), ("Heading 2", H2_PT, 12, 6),
                                          ("Heading 3", H3_PT, 10, 4)):
            st = doc.styles[name]
            st.font.size, st.font.bold, st.font.italic = Pt(size), True, False
            st.font.color.rgb = RGBColor(0, 0, 0)
            set_fonts(st.element.get_or_add_rPr())
            spf = st.paragraph_format
            spf.alignment, spf.line_spacing = HEADING_ALIGN, LINE
            spf.space_before, spf.space_after = Pt(before), Pt(after)
            spf.keep_with_next, spf.page_break_before = True, name == "Heading 1"
        cap = doc.styles["Caption"]
        cap.font.size, cap.font.bold, cap.font.italic = Pt(CAPTION_PT), False, False
        cap.font.color.rgb = RGBColor(0, 0, 0)
        set_fonts(cap.element.get_or_add_rPr())
        cap.paragraph_format.alignment, cap.paragraph_format.line_spacing = CENTER, LINE
        sec = doc.sections[0]
        self._page(sec)
        cp = doc.core_properties
        cp.title = self.front["title"]
        cp.subject = "Final project report"
        cp.author = ", ".join(s["name"] for s in self.front["students"])
        cp.last_modified_by = cp.author
        cp.comments = ""
        cp.keywords = "AutoML, FLAML, autonomous experimentation"
        cp.created = cp.modified = dt.datetime(2026, 11, 13)
        cp.revision = 1

    @staticmethod
    def _page(sec) -> None:
        sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
        for side in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
            setattr(sec, side, Cm(MARGIN_CM))
        sec.header_distance = sec.footer_distance = Cm(1.25)

    # -- paragraph helpers

    def par(self, text: str = "", align=JUSTIFY, size: float = BODY_PT, bold: bool = False, before: float = 0,
            after: float = 6, keep_next: bool = False, page_break: bool = False, style: str | None = None,
            italic: bool = False):
        p = self.doc.add_paragraph(style=style)
        fmt(p, align, before + self.pending_before, after, keep_next, page_break=page_break)
        self.pending_before = 0.0
        if text:
            add_runs(p, text, size, bold, italic)
        return p

    def title(self, text: str, page_break: bool = True, bookmark: str | None = None, before: float = 0):
        p = self.par(text, CENTER, H1_PT, True, before=before, after=18, keep_next=True, page_break=page_break)
        if bookmark:
            add_bookmark(p, bookmark)
        return p

    def spacer(self, page_break: bool = False, keep_next: bool = False):
        """A 2 pt paragraph: Word needs one between two tables, or to carry a page break before a table."""
        p = self.doc.add_paragraph()
        fmt(p, LEFT, 0, 0, keep_next=keep_next, page_break=page_break)
        p.paragraph_format.line_spacing = Pt(2)
        return p

    def before_table(self) -> None:
        """Two tables in a row would merge into one in Word, so separate them."""
        if self.doc.element.body[-2].tag == qn("w:tbl"):
            self.spacer(keep_next=True)

    # -- front pages

    def cover(self) -> None:
        f = self.front
        self.par(f["title"], CENTER, 20, True, after=0)
        self.par(f["report_kind"], CENTER, 14, True, before=24, after=0)
        self.par("Submitted in partial fulfillment for the award of the degree of", CENTER, 14, before=12, after=0)
        self.par(f["degree"], CENTER, 16, True, before=6, after=0)
        self.par(f"({f['department']})", CENTER, 14, True, after=0)
        self.par("Submitted to", CENTER, 14, before=24, after=0)
        self.par(f["institute"], CENTER, 16, True, before=6, after=0)
        self.par(f["institute_city"], CENTER, 16, True, after=0)
        p = self.par(align=CENTER, before=18, after=0)
        p.add_run().add_picture(str(self.logo), width=Cm(5.5))
        self.par("Submitted by", CENTER, 16, True, before=18, after=0)
        for s in f["students"]:
            self.par(f"{s['name']} ({s['scholar']})", CENTER, 14, after=0)
        self.par("Under the supervision of", CENTER, 16, True, before=18, after=0)
        self.par(f["supervisor"], CENTER, 14, after=0)
        self.par(f["supervisor_designation"], CENTER, 14, after=0)
        self.par(f"({f['department']})", CENTER, 14, after=0)
        self.par(f["month_year"], CENTER, 14, True, before=24, after=0)

    def certificate(self) -> None:
        f = self.front
        self.par(f["institute"], CENTER, H1_PT, True, after=0, page_break=True)
        self.par(f["institute_city"], CENTER, H1_PT, True, after=0)
        p = self.par(align=CENTER, before=12, after=0)
        p.add_run().add_picture(str(self.logo), width=Cm(2.9))
        self.title("CERTIFICATE", page_break=False, before=18)
        names = " and ".join(f"{s['name']} ({s['scholar']})" for s in f["students"])
        text = (f"This is to certify that the work embodied in this report entitled **\u201c{f['title']}\u201d** "
                f"has been satisfactorily completed by **{names}**. It is a bonafide piece of work, carried out "
                f"under our guidance in the **{f['department']}**, **Indian Institute of Information Technology, "
                f"Bhopal** for the partial fulfillment of the Bachelor of Technology during the academic year "
                f"{f['academic_year']}.")
        self.par(text, JUSTIFY, 14, after=0)
        self.par("Date:", LEFT, BODY_PT, before=36, after=0)
        left = [(f["supervisor"] + ",", True), (f["supervisor_designation"] + ",", False),
                ("Department of CSE,", False), ("IIIT Bhopal (M.P.)", False)]
        right = [(f["coordinator"] + ",", True), ("Department Coordinator,", False), ("Department of CSE,", False),
                 ("IIIT Bhopal (M.P.)", False)]
        self.signature_table([left, right], before=60)

    def signature_table(self, columns: list[list[tuple[str, bool]]], before: float) -> None:
        table = self.doc.add_table(rows=1, cols=len(columns))
        setup_table(table, [TEXT_WIDTH_CM / len(columns)] * len(columns), margin_lr=0.0)
        set_borders(table, 0, 0, 0, 0, 0, 0)
        for c, lines in enumerate(columns):
            cell = table.cell(0, c)
            for k, (text, bold) in enumerate(lines):
                p = cell_par(cell, LEFT, before=before if k == 0 else 0, first=k == 0)
                add_runs(p, text, BODY_PT, bold)
        self.pending_before = 12

    def declaration(self) -> None:
        f = self.front
        self.spacer(page_break=True, keep_next=True)
        table = self.doc.add_table(rows=1, cols=2)
        setup_table(table, [3.9, TEXT_WIDTH_CM - 3.9], margin_lr=0.1)
        set_borders(table, 0, 0, 0, 0, 0, 0)
        logo_par = cell_par(table.cell(0, 0), CENTER)
        logo_par.add_run().add_picture(str(self.logo), width=Cm(3.5))
        table.cell(0, 0).vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        name_cell = table.cell(0, 1)
        name_cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for k, line in enumerate(("INDIAN INSTITUTE OF", "INFORMATION TECHNOLOGY", f["institute_city"])):
            add_runs(cell_par(name_cell, CENTER, first=k == 0), line, 20, True)
        self.title("DECLARATION", page_break=False, before=18)
        text = (f"We hereby declare that the project report entitled \u201c{f['title']}\u201d is presented in partial "
                f"fulfillment of the requirements for the award of the degree of **Bachelor of Technology in "
                f"Computer Science and Engineering**. It is an authentic documentation of our original work "
                f"carried out under the able guidance of **{f['supervisor']}**. The work has been carried out "
                f"entirely at the Indian Institute of Information Technology, Bhopal. The project work presented "
                f"has not been submitted in part or whole for the award of any degree or professional diploma in "
                f"any other institute or organization.")
        self.par(text, JUSTIFY, BODY_PT, after=12)
        self.par("We, with this, declare that the facts mentioned above are true to the best of our knowledge. "
                 "In case of any unlikely discrepancy that may occur, we will be the ones to take responsibility.",
                 JUSTIFY, BODY_PT)
        table = self.doc.add_table(rows=len(f["students"]), cols=2)
        setup_table(table, [TEXT_WIDTH_CM * 0.6, TEXT_WIDTH_CM * 0.4], margin_lr=0.0)
        set_borders(table, 0, 0, 0, 0, 0, 0)
        for r, s in enumerate(f["students"]):
            before = 54 if r == 0 else 30
            add_runs(cell_par(table.cell(r, 0), LEFT, before=before), f"{s['name']} ({s['scholar']})", BODY_PT)
            add_runs(cell_par(table.cell(r, 1), RIGHT, before=before), "Signature", BODY_PT)
        self.pending_before = 12

    def text_page(self, title: str, key: str, signed: bool = False) -> None:
        self.title(title)
        for b in self.special[key]:
            self.paragraph_block(b)
        if signed:
            for k, s in enumerate(self.front["students"]):
                self.par(f"{s['name']} ({s['scholar']})", LEFT, before=18 if k == 0 else 0, after=0)

    def list_page(self, title: str, header: tuple[str, str, str], rows: list[tuple[str, str, str]]) -> None:
        self.title(title)
        table = self.doc.add_table(rows=len(rows) + 1, cols=3)
        setup_table(table, [1.8, TEXT_WIDTH_CM - 4.3, 2.5], margin_tb=0.1)
        set_borders(table, 0, 0, 0, 0, 0, 0)
        for c, text in enumerate(header):
            add_runs(cell_par(table.cell(0, c), CENTER if c != 1 else LEFT), text, BODY_PT, True)
        row_flags(table.rows[0], header=True)
        for r, (no, text, bookmark) in enumerate(rows, 1):
            add_runs(cell_par(table.cell(r, 0), CENTER), no, BODY_PT)
            add_runs(cell_par(table.cell(r, 1), LEFT), text, BODY_PT)
            p = cell_par(table.cell(r, 2), CENTER)
            if bookmark:
                add_field(p, f"PAGEREF {bookmark} \\h", "1", BODY_PT)
            row_flags(table.rows[r])
        self.pending_before = 12

    # -- body blocks

    def paragraph_block(self, b: Block) -> None:
        if b.kind == "para":
            self.par(resolve(b.text, self.nums))
        elif b.kind in ("bullets", "numbers"):
            self.list_block(b)
        else:
            raise SystemExit(f"{b.source}: only paragraphs and lists are allowed here")

    def list_block(self, b: Block) -> None:
        for k, item in enumerate(b.items, 1):
            label = f"{k}." if b.kind == "numbers" else "\u2022"
            last = k == len(b.items)
            p = self.doc.add_paragraph()
            fmt(p, JUSTIFY, self.pending_before, 6 if last else 3, left=0.75, first=-0.75, tab=0.75)
            self.pending_before = 0.0
            add_runs(p, label + "\t" + resolve(item, self.nums))

    def heading(self, b: Block) -> None:
        if b.level == 1:
            if b.opts["numbered"]:
                self.chapter += 1
                self.section = self.subsection = 0
                number, sno = f"{self.chapter}.", str(self.chapter)
            else:
                number, sno = "", str(self.chapter + 1)
            bookmark = f"_RefCh{len(self.toc) + 1}"
            self.toc.append((sno, b.text, bookmark))
            text, hang, style = b.text.upper(), 1.0, "Heading 1"
        elif b.level == 2:
            self.section += 1
            self.subsection = 0
            number, text, hang, style = f"{self.chapter}.{self.section}", b.text, 1.25, "Heading 2"
            bookmark = None
        else:
            self.subsection += 1
            number = f"{self.chapter}.{self.section}.{self.subsection}"
            text, hang, style, bookmark = b.text, 1.5, "Heading 3", None
        p = self.doc.add_paragraph(style=style)
        before = {1: 0, 2: 12, 3: 10}[b.level]
        p.paragraph_format.space_before = Pt(before + self.pending_before)
        self.pending_before = 0.0
        size = {1: H1_PT, 2: H2_PT, 3: H3_PT}[b.level]
        if number:
            fmt_heading_number(p, hang)
            add_runs(p, number + "\t" + smart_quotes(text), size, True)
        else:
            add_runs(p, smart_quotes(text), size, True)
        if bookmark:
            add_bookmark(p, bookmark)

    def caption(self, label: str, number: int, text: str, before: float, after: float, keep_next: bool,
                bookmark: str) -> None:
        p = self.doc.add_paragraph(style="Caption")
        fmt(p, CENTER, before, after, keep_next)
        style_run(p.add_run(f"{label} "), CAPTION_PT, True)
        add_field(p, f"SEQ {label} \\* ARABIC", str(number), CAPTION_PT, True)
        style_run(p.add_run(": "), CAPTION_PT, True)
        add_runs(p, text, CAPTION_PT)
        add_bookmark(p, bookmark)

    def figure(self, b: Block) -> None:
        n = self.nums.number(b.opts["id"])
        path = SRC / b.opts["path"]
        if not path.exists():
            raise SystemExit(f"{b.source}: missing figure file {path}")
        p = self.doc.add_paragraph()
        fmt(p, CENTER, 6 + self.pending_before, 0, keep_next=True, line=1.0)
        self.pending_before = 0.0
        p.add_run().add_picture(str(path), width=Cm(b.opts["width"]))
        text = resolve(b.text, self.nums)
        self.caption("Figure", n, text, 6, 12, False, f"_RefFig{n}")
        self.figures.append((n, text))

    def code(self, b: Block) -> None:
        n = self.nums.number(b.opts["id"])
        self.before_table()
        table = self.doc.add_table(rows=1, cols=1)
        setup_table(table, [TEXT_WIDTH_CM], margin_lr=0.2, margin_tb=0.1)
        set_borders(table, 4, 4, 4, 4, 0, 0)
        cell = table.cell(0, 0)
        lines = [ln.rstrip() for ln in b.items]
        while lines and not lines[-1]:
            lines.pop()
        for k, line in enumerate(lines):
            p = cell_par(cell, LEFT, first=k == 0)
            p.paragraph_format.keep_with_next = True
            text = line.replace(" ", "\u00a0") if line else "\u00a0"
            style_run(p.add_run(text), CODE_PT, font=CODE_FONT)
        row_flags(table.rows[0])
        text = resolve(b.text, self.nums)
        self.caption("Figure", n, text, 6, 12, False, f"_RefFig{n}")
        self.figures.append((n, text))

    def table(self, b: Block) -> None:
        n = self.nums.number(b.opts["id"])
        rows = b.rows
        ncols = len(rows[0])
        for r in rows:
            if len(r) != ncols:
                raise SystemExit(f"{b.source}: table {b.opts['id']} row has {len(r)} cells, expected {ncols}: {r}")
        text = resolve(b.text, self.nums)
        self.caption("Table", n, text, 12 + self.pending_before, 6, True, f"_RefTab{n}")
        self.pending_before = 0.0
        self.tables.append((n, text))
        width = float(b.opts.get("width", TEXT_WIDTH_CM))
        weights = [float(w) for w in b.opts["widths"].split(",")] if "widths" in b.opts else [1.0] * ncols
        widths = [width * w / sum(weights) for w in weights]
        align = b.opts.get("align", "j" + "c" * (ncols - 1))
        if len(align) != ncols:
            raise SystemExit(f"{b.source}: table {b.opts['id']} align={align} does not match {ncols} columns")
        table = self.doc.add_table(rows=len(rows), cols=ncols)
        setup_table(table, widths)
        set_borders(table)
        for r, row in enumerate(rows):
            for c, cell_text in enumerate(row):
                cell = table.cell(r, c)
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                p = cell_par(cell, CENTER if r == 0 else ALIGNS[align[c]])
                add_runs(p, resolve(cell_text, self.nums), TABLE_PT, bold=r == 0)
            row_flags(table.rows[r], header=r == 0)
        self.pending_before = 12

    def equation(self, b: Block) -> None:
        n = self.nums.number(b.opts["id"])
        self.before_table()
        table = self.doc.add_table(rows=1, cols=3)
        setup_table(table, [1.6, TEXT_WIDTH_CM - 3.2, 1.6], margin_lr=0.0)
        set_borders(table, 0, 0, 0, 0, 0, 0)
        before = 4 + self.pending_before
        self.pending_before = 0.0
        for c in range(3):
            table.cell(0, c).vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        cell_par(table.cell(0, 0), LEFT, before, 4)
        mid = cell_par(table.cell(0, 1), CENTER, before, 4)
        mid._p.append(parse_xml(omath_para(*EQUATIONS[b.opts["id"]])))
        add_runs(cell_par(table.cell(0, 2), RIGHT, before, 4), f"({n})", BODY_PT)
        row_flags(table.rows[0])
        self.pending_before = 8

    def algorithm(self, b: Block) -> None:
        n = self.nums.number(b.opts["id"])
        self.before_table()
        table = self.doc.add_table(rows=2, cols=1)
        setup_table(table, [TEXT_WIDTH_CM], margin_lr=0.1, margin_tb=0.06)
        set_borders(table, 12, 12, 0, 0, 6, 0)
        head = cell_par(table.cell(0, 0), LEFT)
        head.paragraph_format.keep_with_next = True
        style_run(head.add_run(f"Algorithm {n}: "), TABLE_PT, True)
        add_runs(head, resolve(b.text, self.nums), TABLE_PT, True)
        cell = table.cell(1, 0)
        first, line_no = True, 0
        for raw in b.items:
            s = raw.strip()
            if not s:
                continue
            m = re.match(r"^(Input|Output|Data|Result):\s*(.*)$", s)
            p = cell_par(cell, JUSTIFY, first=first)
            first = False
            if m:
                style_run(p.add_run(m.group(1) + ": "), TABLE_PT, True)
                add_runs(p, resolve(m.group(2), self.nums), TABLE_PT)
                continue
            m = re.match(r"^([1-6])\s+(.*)$", s)
            if not m:
                raise SystemExit(f"{b.source}: algorithm line needs a level 1 to 6: {s}")
            line_no += 1
            indent = 0.8 + 0.6 * (int(m.group(1)) - 1)
            fmt(p, JUSTIFY, 0, 0, left=indent, first=-indent, tab=indent)
            style_run(p.add_run(f"{line_no}:"), TABLE_PT)
            add_runs(p, "\t" + resolve(m.group(2), self.nums), TABLE_PT)
        row_flags(table.rows[0])
        row_flags(table.rows[1])  # the whole algorithm moves to the next page rather than splitting
        self.pending_before = 12

    def references(self) -> None:
        for k, key in enumerate(self.nums.cite_order, 1):
            p = self.doc.add_paragraph()
            fmt(p, LEFT, 0, 4, left=1.0, first=-1.0, tab=1.0)
            add_runs(p, f"[{k}]\t" + smart_quotes(self.nums.refs[key]), REF_PT)

    # -- assembly

    def build(self) -> None:
        self.setup()
        self.cover()
        self.certificate()
        self.declaration()
        self.text_page("ACKNOWLEDGEMENT", "acknowledgement", signed=True)
        self.text_page("AREA OF WORK", "area-of-work")
        sec = self.doc.add_section(WD_SECTION.NEW_PAGE)
        # python-docx puts the first section's break into a new empty paragraph; keep it 2 pt high.
        # The contents, figure and table lists are built at the end (when their rows are known)
        # and moved in front of this paragraph.
        holder = Paragraph(self.doc.element.body[-2], self.doc._body)
        fmt(holder, LEFT, 0, 0)
        holder.paragraph_format.line_spacing = Pt(2)
        self._page(sec)
        sec.footer.is_linked_to_previous = False
        fp = sec.footer.paragraphs[0]
        fmt(fp, CENTER, 0, 0)
        add_field(fp, "PAGE", "1", BODY_PT)
        pg = OxmlElement("w:pgNumType")
        pg.set(qn("w:start"), "1")
        sec._sectPr.find(qn("w:pgMar")).addnext(pg)
        self.title("ABSTRACT", page_break=False, bookmark="_RefAbstract")
        for b in self.special["abstract"]:
            self.paragraph_block(b)
        for b in self.blocks:
            getattr(self, {"heading": "heading", "para": "paragraph_block", "bullets": "list_block",
                           "numbers": "list_block", "figure": "figure", "code": "code", "table": "table",
                           "equation": "equation", "algorithm": "algorithm"}[b.kind])(b)
            if b.kind == "heading" and b.level == 1 and not b.opts["numbered"] and b.text.lower() == "references":
                self.references()
        self.front_lists(holder)

    def front_lists(self, holder) -> None:
        """Build the contents, figure and table lists at the end, then move them before the Abstract."""
        body = self.doc.element.body
        start = len(body) - 1  # the final sectPr stays last
        rows = [("", "Certificate", ""), ("", "Declaration", ""), ("", "Acknowledgement", ""),
                ("", "Area of work", ""), ("", "List of figures", ""), ("", "List of tables", ""),
                ("", "Abstract", "_RefAbstract")]
        rows += [(sno, title, bm) for sno, title, bm in self.toc]
        self.list_page("TABLE OF CONTENTS", ("S.no", "Title", "Page No."), rows)
        self.list_page("LIST OF FIGURES", ("Fig", "Description", "Page No."),
                       [(str(n), t, f"_RefFig{n}") for n, t in self.figures])
        self.list_page("LIST OF TABLES", ("Table No.", "Description", "Page No."),
                       [(str(n), t, f"_RefTab{n}") for n, t in self.tables])
        for el in list(body)[start:-1]:
            holder._p.addprevious(el)


def fmt_heading_number(p, hang: float) -> None:
    pf = p.paragraph_format
    pf.left_indent = Cm(hang)
    pf.first_line_indent = Cm(-hang)
    pf.tab_stops.add_tab_stop(Cm(hang))


# ----------------------------------------------------------------------------- checks


def source_checks(blocks: list[Block], nums: Numbers) -> list[str]:
    problems = []
    defined = {k for table in nums.items.values() for k in table}
    for ident in sorted(defined - nums.used):
        problems.append(f"{ident} is never referred to in the text")
    chapter = 0
    for b in blocks:
        if b.kind == "heading" and b.level == 1:
            chapter += 1
        if chapter in (1, 2):
            texts = [b.text] if b.kind == "para" else (b.items if b.kind in ("bullets", "numbers") else [])
            for text in texts:
                if not CITE.search(text):
                    problems.append(f"chapter {chapter} paragraph without a citation: {text[:70]}...")
        for text in block_texts(b):
            if b.kind in ("para", "bullets", "numbers") and text.rstrip()[-1:] not in (".",):
                problems.append(f"{b.source}: paragraph does not end with a full stop: ...{text[-60:]}")
            if b.kind in ("figure", "code", "table") and text == b.text and text.rstrip().endswith("."):
                problems.append(f"{b.source}: caption ends with a full stop: {text}")
            if "\u2014" in text or "\u2013" in text:
                problems.append(f"{b.source}: em or en dash in: {text[:70]}")
            if re.search(r"[\U0001F000-\U0001FAFF\u2600-\u27BF]", text):
                problems.append(f"{b.source}: emoji in: {text[:70]}")
            if re.search(r"\b(e\.g|i\.e|etc)\.", text):
                problems.append(f"{b.source}: abbreviation (e.g., i.e., etc.) in: {text[:70]}")
    return problems


def docx_checks(path: Path) -> list[str]:
    """Format checks on the written file: fonts, sizes, spacing, underline, empty paragraphs."""
    problems = []
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    from lxml import etree

    root = etree.fromstring(xml.encode("utf-8"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    w = "{%s}" % ns["w"]
    fonts = set()
    math_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
    for rf in root.iter(w + "rFonts"):
        in_math = any(a.tag.startswith(math_ns) for a in rf.iterancestors())
        if not in_math:
            fonts.add(rf.get(w + "ascii"))
    for bad in fonts - {FONT, CODE_FONT, None}:
        problems.append(f"font in use outside equations: {bad}")
    for u in root.iter(w + "u"):
        if u.get(w + "val") not in ("none", None):
            problems.append("underlined text found")
    empty = 0
    for p in root.iter(w + "p"):
        sp = p.find(f"{w}pPr/{w}spacing")
        line, rule = (sp.get(w + "line"), sp.get(w + "lineRule")) if sp is not None else (None, None)
        has_picture = p.find(f".//{w}drawing") is not None
        spacer = rule == "exact" and line is not None and int(line) <= 40
        if line is not None and line != "276" and not has_picture and not spacer:
            problems.append(f"line spacing {line} {rule} in: {''.join(t.text or '' for t in p.iter(w + 't'))[:50]}")
        in_cell = p.getparent().tag == w + "tc"
        has_text = "".join(t.text or "" for t in p.iter(w + "t")).strip()
        has_other = has_picture or p.find(".//{*}oMathPara") is not None
        if not has_text and not has_other and not in_cell and not spacer:
            empty += 1
    if empty:
        problems.append(f"{empty} empty paragraphs outside tables")
    for sect in root.iter(w + "sectPr"):
        mar = sect.find(w + "pgMar")
        sizes = {mar.get(w + k) for k in ("top", "bottom", "left", "right")}
        if len(sizes) != 1:
            problems.append(f"unequal margins {sizes}")
        pgsz = sect.find(w + "pgSz")
        if (pgsz.get(w + "w"), pgsz.get(w + "h")) != ("11906", "16838"):
            problems.append(f"page size {pgsz.get(w + 'w')} x {pgsz.get(w + 'h')} is not A4")
    return problems


# ----------------------------------------------------------------------------- main


def word_finalize(docx: Path, pdf: Path) -> None:
    script = HERE / "word_finalize.ps1"
    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
           "-InPath", str(docx), "-PdfPath", str(pdf), "-UpdateFields", "-Save"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        raise SystemExit("Word step failed:\n" + out.stdout[-2000:] + out.stderr[-2000:])
    print("Word:", out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "done")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-word", action="store_true", help="do not open Word (no page numbers, no PDF)")
    args = ap.parse_args()
    front = tomllib.loads((SRC / "front.toml").read_text(encoding="utf-8"))
    refs = load_references(SRC / "references.txt")
    special, blocks = {}, []
    for path in sorted((SRC / "chapters").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if path.name.startswith("00-"):
            text = text.format(**{k: v for k, v in front.items() if isinstance(v, str)})
            special[path.stem[3:]] = parse_blocks(text, path.name)
        else:
            blocks.extend(parse_blocks(text, path.name))
    nums = Numbers(blocks, refs)
    report = Report(front, blocks, special, nums)
    report.build()
    problems = source_checks(blocks, nums)
    unused = sorted(set(refs) - set(nums.cite_order))
    report.doc.save(OUT_DOCX)
    print(f"written {OUT_DOCX.relative_to(ROOT)}: {report.chapter} chapters, {len(report.figures)} figures, "
          f"{len(report.tables)} tables, {len(nums.items['eq'])} equations, {len(nums.items['alg'])} algorithms, "
          f"{len(nums.cite_order)} references cited ({len(unused)} listed but not cited: {', '.join(unused)})")
    if not args.no_word:
        word_finalize(OUT_DOCX, OUT_PDF)
    problems += docx_checks(OUT_DOCX)
    print(f"{len(problems)} format problems")
    for p in problems:
        print("  -", p)


if __name__ == "__main__":
    main()
