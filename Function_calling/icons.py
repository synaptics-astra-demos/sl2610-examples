"""Lucide icon font (subset) helpers.

The bundled ``fonts/lucide-subset.ttf`` is a 2 KB cut of the Lucide icon
font containing just the glyphs we render. Yocto has no QtSvg and no
emoji font, so we ship our own font and reference codepoints directly.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtGui import QFont, QFontDatabase

_FONT_PATH = Path(__file__).resolve().parent / "fonts" / "lucide-subset.ttf"
_FAMILY: str | None = None

MIC = ""
MIC_OFF = ""
SEND = ""
RESET = ""


def ensure_loaded() -> str:
    global _FAMILY
    if _FAMILY is not None:
        return _FAMILY
    font_id = QFontDatabase.addApplicationFont(str(_FONT_PATH))
    if font_id < 0:
        raise RuntimeError(f"failed to load icon font: {_FONT_PATH}")
    families = QFontDatabase.applicationFontFamilies(font_id)
    if not families:
        raise RuntimeError(f"icon font registered but no families: {_FONT_PATH}")
    _FAMILY = families[0]
    return _FAMILY


def icon_font(pixel_size: int) -> QFont:
    family = ensure_loaded()
    font = QFont(family)
    font.setPixelSize(pixel_size)
    font.setStyleStrategy(QFont.PreferAntialias)
    return font
