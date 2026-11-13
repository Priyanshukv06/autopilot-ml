"""Small helpers that write Office Math (OMML) for the report's equations.

The equations use Word's equation font (Cambria Math), because Word only draws stretchy
brackets, roots and accents correctly in a math font. Variables are written with ``v``
(italic, the math default), names, numbers and labels with ``t`` (upright).
"""

from __future__ import annotations

from xml.sax.saxutils import escape

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
MATH_FONT = "Cambria Math"
HALF_POINTS = 24  # 12 pt, the size of the body text


def _join(parts) -> str:
    if isinstance(parts, str):
        return parts
    return "".join(_join(p) for p in parts)


def _run(text: str, upright: bool) -> str:
    mrpr = '<m:rPr><m:sty m:val="p"/></m:rPr>' if upright else ""
    wrpr = (f'<w:rPr><w:rFonts w:ascii="{MATH_FONT}" w:hAnsi="{MATH_FONT}"/>'
            f'<w:sz w:val="{HALF_POINTS}"/><w:szCs w:val="{HALF_POINTS}"/></w:rPr>')
    return f'<m:r>{mrpr}{wrpr}<m:t xml:space="preserve">{escape(text)}</m:t></m:r>'


def t(text: str) -> str:
    """Upright text: numbers, operators, names and labels."""
    return _run(text, upright=True)


def v(text: str) -> str:
    """A variable: italic."""
    return _run(text, upright=False)


def func(name: str, *body) -> str:
    """A named function such as max or ln, with Word's spacing after the name."""
    return f"<m:func><m:fName>{t(name)}</m:fName><m:e>{_join(body)}</m:e></m:func>"


def frac(num, den) -> str:
    return f"<m:f><m:num>{_join(num)}</m:num><m:den>{_join(den)}</m:den></m:f>"


def sqrt(*parts) -> str:
    return (
        '<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/>'
        f"<m:e>{_join(parts)}</m:e></m:rad>"
    )


def sub(base, lower) -> str:
    return f"<m:sSub><m:e>{_join(base)}</m:e><m:sub>{_join(lower)}</m:sub></m:sSub>"


def sup(base, upper) -> str:
    return f"<m:sSup><m:e>{_join(base)}</m:e><m:sup>{_join(upper)}</m:sup></m:sSup>"


def subsup(base, lower, upper) -> str:
    return (
        f"<m:sSubSup><m:e>{_join(base)}</m:e><m:sub>{_join(lower)}</m:sub>"
        f"<m:sup>{_join(upper)}</m:sup></m:sSubSup>"
    )


def bar(*parts) -> str:
    return f'<m:bar><m:barPr><m:pos m:val="top"/></m:barPr><m:e>{_join(parts)}</m:e></m:bar>'


def hat(*parts) -> str:
    return f'<m:acc><m:accPr><m:chr m:val="\u0302"/></m:accPr><m:e>{_join(parts)}</m:e></m:acc>'


def paren(*parts, left: str = "(", right: str = ")") -> str:
    return (
        f'<m:d><m:dPr><m:begChr m:val="{left}"/><m:endChr m:val="{right}"/></m:dPr>'
        f"<m:e>{_join(parts)}</m:e></m:d>"
    )


def nary(char: str, lower, upper, *body) -> str:
    """A sum or product with limits below and above (upper may be None)."""
    hide = '<m:supHide m:val="1"/>' if upper is None else ""
    return (
        f'<m:nary><m:naryPr><m:chr m:val="{char}"/><m:limLoc m:val="undOvr"/>{hide}</m:naryPr>'
        f"<m:sub>{_join(lower)}</m:sub><m:sup>{_join(upper or '')}</m:sup>"
        f"<m:e>{_join(body)}</m:e></m:nary>"
    )


def omath_para(*parts) -> str:
    """A display equation, centred in its paragraph."""
    return (
        f'<m:oMathPara xmlns:m="{M_NS}" xmlns:w="{W_NS}">'
        '<m:oMathParaPr><m:jc m:val="center"/></m:oMathParaPr>'
        f"<m:oMath>{_join(parts)}</m:oMath></m:oMathPara>"
    )
