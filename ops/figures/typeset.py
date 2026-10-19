"""Text as SVG outlines, for the generated figures (``gen_logo.py``).

``nunito()`` downloads the Nunito variable font (SIL OFL) into ``~/.cache/sprout-logo``
once, and saves a static instance there for each weight.
``pip install -e ".[docs]"`` provides fonttools and uharfbuzz. HarfBuzz shapes the
text, so the measured width is the width that renders.
"""

from __future__ import annotations

import urllib.request
from functools import cache
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

NUNITO_URL = "https://github.com/google/fonts/raw/main/ofl/nunito/Nunito%5Bwght%5D.ttf"
CACHE = Path.home() / ".cache" / "sprout-logo"


def nunito(weight: int) -> Path:
    """A static instance of the Nunito variable font at ``weight``."""
    path = CACHE / f"Nunito-{weight}.ttf"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        variable = CACHE / "Nunito[wght].ttf"
        if not variable.exists():
            with urllib.request.urlopen(NUNITO_URL, timeout=120) as r:
                variable.write_bytes(r.read())
        instantiateVariableFont(TTFont(str(variable)), {"wght": weight}).save(str(path))
    return path


@cache
def _load(font_path: Path) -> tuple[hb.Font, TTFont]:
    return hb.Font(hb.Face(font_path.read_bytes())), TTFont(str(font_path))


def outline(
    text: str, font_path: Path, size: float, x: float, baseline: float, tracking_em: float = 0.0
) -> tuple[str, float]:
    """The SVG path data and the advance width of ``text`` set at ``size`` px.

    The text starts at ``x`` on ``baseline``. ``tracking_em`` adds space after each glyph,
    in em. The width does not include the space after the last glyph.
    """
    font, tt = _load(font_path)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf, {"kern": True, "liga": True})

    glyph_set = tt.getGlyphSet()
    order = tt.getGlyphOrder()
    scale = size / tt["head"].unitsPerEm
    tracking = tracking_em * size

    pen = SVGPathPen(glyph_set, ntos=lambda v: f"{v:.1f}".rstrip("0").rstrip("."))
    cursor = x
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions, strict=True):
        gx = cursor + pos.x_offset * scale
        gy = baseline - pos.y_offset * scale
        glyph_set[order[info.codepoint]].draw(TransformPen(pen, (scale, 0, 0, -scale, gx, gy)))
        cursor += pos.x_advance * scale + tracking
    return pen.getCommands(), cursor - x - tracking


class Glyphs:
    """Text as ``<use>`` references to glyph outlines that :meth:`defs` defines once.

    Each glyph's outline occurs once in the file, so many labels keep the file small.
    Call :meth:`defs` once, after the last call to :meth:`text`.
    """

    def __init__(self) -> None:
        self._defs: dict[str, str] = {}

    def text(
        self, s: str, font_path: Path, size: float, x: float, y: float, fill: str, anchor: str = "start"
    ) -> tuple[str, float]:
        """The SVG group that places ``s`` at ``x`` on baseline ``y``, and its advance width.

        ``anchor`` is ``start``, ``middle`` or ``end``, as in the SVG ``text-anchor`` at ``x``.
        """
        font, tt = _load(font_path)
        buf = hb.Buffer()
        buf.add_str(s)
        buf.guess_segment_properties()
        hb.shape(font, buf, {"kern": True, "liga": True})
        glyph_set = tt.getGlyphSet()
        order = tt.getGlyphOrder()
        upem = tt["head"].unitsPerEm
        scale = size / upem
        tag = font_path.stem.replace("[", "").replace("]", "")

        uses = []
        cursor = 0
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions, strict=True):
            name = order[info.codepoint]
            gid = f"{tag}-{info.codepoint}"
            if gid not in self._defs:
                pen = SVGPathPen(glyph_set, ntos=lambda v: f"{v:.0f}")
                glyph_set[name].draw(pen)
                self._defs[gid] = pen.getCommands()
            if self._defs[gid]:
                uses.append(f'<use href="#{gid}" x="{cursor + pos.x_offset}" y="{pos.y_offset}"/>')
            cursor += pos.x_advance
        w = cursor * scale
        dx = {"start": 0.0, "middle": -w / 2, "end": -w}[anchor]
        svg = (
            f'<g fill="{fill}" transform="translate({x + dx:.1f} {y:.1f}) scale({scale:.5f} {-scale:.5f})">'
            + "".join(uses)
            + "</g>"
        )
        return svg, w

    def defs(self) -> str:
        """The ``<defs>`` element with the outline of each glyph that :meth:`text` placed."""
        return (
            "<defs>"
            + "".join(f'<path id="{gid}" d="{d}"/>' for gid, d in self._defs.items() if d)
            + "</defs>"
        )
