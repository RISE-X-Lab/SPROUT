"""Text as SVG outlines, for the generated figures (gen_logo.py, gen_architecture.py).

Nunito (SIL OFL) is fetched into ``~/.cache`` on first use and instantiated at
the requested weight; ``pip install -e ".[docs]"`` provides fonttools and uharfbuzz. Shaping
goes through HarfBuzz, so the measured width is the width that renders.
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
    """Return (svg path data, advance width) for ``text`` set at ``size`` px."""
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
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        gx = cursor + pos.x_offset * scale
        gy = baseline - pos.y_offset * scale
        glyph_set[order[info.codepoint]].draw(TransformPen(pen, (scale, 0, 0, -scale, gx, gy)))
        cursor += pos.x_advance * scale + tracking
    return pen.getCommands(), cursor - x - tracking


class Glyphs:
    """Text as ``<use>`` references to per-glyph ``<defs>``.

    A figure with hundreds of labels would be a megabyte of repeated outlines;
    defining each glyph once in font units and placing it with ``<use>`` keeps
    the file small. Call :meth:`defs` once, after every label has been placed.
    """

    def __init__(self) -> None:
        self._defs: dict[str, str] = {}

    def text(
        self, s: str, font_path: Path, size: float, x: float, y: float, fill: str, anchor: str = "start"
    ) -> tuple[str, float]:
        """Place ``s`` with its baseline at ``y``; returns (svg, advance width)."""
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
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
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
        return (
            "<defs>"
            + "".join(f'<path id="{gid}" d="{d}"/>' for gid, d in self._defs.items() if d)
            + "</defs>"
        )
