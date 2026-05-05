"""Neutral UI theme. Single source of truth for colors + typography + spacing."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication


@dataclass(frozen=True)
class Palette:
    bg_primary: str = "#f8fafc"
    bg_secondary: str = "#ffffff"
    bg_tertiary: str = "#f1f5f9"
    bg_elevated: str = "#ffffff"

    text_primary: str = "#0f172a"
    text_secondary: str = "#475569"
    text_muted: str = "#94a3b8"
    text_inverse: str = "#ffffff"

    accent: str = "#2563eb"
    accent_secondary: str = "#3b82f6"
    accent_light: str = "#dbeafe"
    accent_dark: str = "#1d4ed8"

    border: str = "#e2e8f0"
    border_strong: str = "#cbd5e1"

    success: str = "#16a34a"
    success_light: str = "#dcfce7"
    danger: str = "#dc2626"
    danger_light: str = "#fee2e2"
    warning: str = "#d97706"
    warning_light: str = "#fef3c7"
    info: str = "#0284c7"
    info_light: str = "#e0f2fe"


@dataclass(frozen=True)
class Typography:
    xs: int = 14
    sm: int = 16
    md: int = 18
    lg: int = 22
    xl: int = 28
    family: str = "Inter, Roboto, 'DejaVu Sans', system-ui, sans-serif"
    mono: str = "'JetBrains Mono', 'DejaVu Sans Mono', monospace"


@dataclass(frozen=True)
class Spacing:
    xs: int = 4
    sm: int = 8
    md: int = 12
    lg: int = 20
    xl: int = 32
    xxl: int = 48


PALETTE = Palette()
TYPE = Typography()
SPACE = Spacing()

CHART_COLORS = {
    "cpu":    "#2563eb",
    "memory": "#7c3aed",
    "npu":    "#059669",
    "temp":   "#d97706",
    "power":  "#db2777",
    "infer":  "#0d9488",
}


def _stylesheet(p: Palette) -> str:
    t = TYPE
    return f"""
    QMainWindow, QWidget {{
        background: {p.bg_primary};
        color: {p.text_primary};
        font-family: {t.family};
        font-size: {t.md}px;
    }}

    QFrame#Card,
    MetricsPanel,
    CommandLog {{
        background: {p.bg_secondary};
        border: 1px solid {p.border};
        border-radius: 12px;
    }}

    QFrame#MetricTile {{
        background: {p.bg_secondary};
        border: 1px solid {p.border};
        border-radius: 10px;
    }}

    QLabel#SectionHeader {{
        color: {p.text_muted};
        font-weight: 700;
        font-size: {t.sm}px;
        letter-spacing: 1.2px;
    }}
    QLabel#MetricValue {{
        color: {p.text_primary};
        font-family: {t.mono};
        font-size: {t.xl}px;
        font-weight: 600;
    }}
    QLabel#MetricLabel {{
        color: {p.text_secondary};
        font-size: {t.sm}px;
        font-weight: 500;
    }}
    QLabel#HeaderMeta {{
        color: {p.text_muted};
        font-size: {t.xs}px;
        font-weight: 500;
        letter-spacing: 0.04em;
    }}

    QLineEdit {{
        background: {p.bg_secondary};
        color: {p.text_primary};
        border: 1px solid {p.border_strong};
        border-radius: 10px;
        padding: 14px 16px;
        font-size: {t.md}px;
        selection-background-color: {p.accent_light};
        selection-color: {p.text_primary};
    }}
    QLineEdit:focus {{
        border: 1px solid {p.accent};
    }}

    QPushButton#PrimaryButton {{
        background: {p.accent};
        color: {p.text_inverse};
        border: none;
        border-radius: 10px;
        padding: 14px 28px;
        font-weight: 600;
        font-size: {t.md}px;
        min-height: 56px;
    }}
    QPushButton#PrimaryButton:hover {{
        background: {p.accent_dark};
    }}
    QPushButton#PrimaryButton:disabled {{
        background: {p.border_strong};
        color: {p.text_muted};
    }}

    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {p.border_strong};
        border-radius: 5px;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {p.text_muted};
    }}
    QScrollBar::add-line, QScrollBar::sub-line {{
        height: 0;
    }}
    """


def apply_theme(app: QApplication, palette: Palette = PALETTE) -> Palette:
    app.setStyleSheet(_stylesheet(palette))
    base_font = QFont()
    base_font.setFamily("Inter")
    base_font.setPointSize(12)
    app.setFont(base_font)
    return palette
