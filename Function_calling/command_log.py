"""Bottom-half scrolling command + response log.

Each turn renders as a stack of QFrame bubble widgets:
    YOU bubble (accent-light bg, right margin)
    -> one bubble per tool step (TOOL chip, name, arg chips, result line)
    -> respond() renders as a plain ASSISTANT chat bubble (no side effect)
    Footer = action count + latency.

Auto-scrolls to the newest entry; keeps a fixed ring of recent turns.
"""

from __future__ import annotations

import html as _html
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)

from theme import PALETTE, SPACE, TYPE

MAX_TURNS = 200

P = PALETTE
T = TYPE
S = SPACE


@dataclass(frozen=True)
class ToolStep:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    message: str = ""
    error: str | None = None

    @property
    def is_respond(self) -> bool:
        return self.name == "respond"


@dataclass(frozen=True)
class LogTurn:
    user: str
    steps: tuple[ToolStep, ...]
    latency_ms: float


def _chip(text: str, *, fg: str, bg: str, border: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"background: {bg}; color: {fg}; border: 1px solid {border}; "
        f"border-radius: 999px; padding: 3px 10px; "
        f"font-size: {T.xs}px; font-weight: 700; letter-spacing: 0.06em;"
    )
    lbl.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
    return lbl


class _UserBubble(QFrame):
    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            f"_UserBubble {{ background: {P.accent_light}; border-radius: 10px; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        chip = _chip("YOU", fg=P.accent_dark, bg=P.bg_secondary, border="#bfdbfe")
        layout.addWidget(chip, alignment=Qt.AlignLeft)

        body = QLabel(text)
        body.setWordWrap(True)
        body.setStyleSheet(
            f"background: transparent; font-size: {T.md}px; "
            f"color: {P.text_primary}; line-height: 145%;"
        )
        layout.addWidget(body)


class _ToolBubble(QFrame):
    def __init__(self, step: ToolStep, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        is_err = step.status == "error"
        bg = P.danger_light if is_err else P.bg_tertiary
        border = "#fca5a5" if is_err else P.border
        self.setStyleSheet(
            f"_ToolBubble {{ background: {bg}; "
            f"border: 1px solid {border}; border-radius: 10px; }}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        head = QHBoxLayout()
        head.setSpacing(10)
        accent = P.danger if is_err else P.accent
        chip_border = "#fecaca" if is_err else "#bfdbfe"
        label = "ERROR" if is_err else "TOOL"
        head.addWidget(_chip(label, fg=accent, bg=P.bg_secondary, border=chip_border))

        name_color = P.danger if is_err else P.text_primary
        name_lbl = QLabel(step.name)
        name_lbl.setStyleSheet(
            f"background: transparent; font-size: 13px; font-family: {T.mono}; "
            f"font-weight: 700; color: {name_color};"
        )
        head.addWidget(name_lbl)
        head.addStretch(1)
        layout.addLayout(head)

        if step.args:
            args_row = QHBoxLayout()
            args_row.setSpacing(6)
            chip_border = "#fecaca" if is_err else P.border
            for k, v in step.args.items():
                k_esc = _html.escape(str(k))
                v_esc = _html.escape(_arg_display(v))
                chip_w = QLabel(
                    f'<span style="color:{P.text_secondary}">{k_esc}</span>'
                    f'<span style="color:{P.text_muted}">=</span>'
                    f'<span style="color:{P.text_primary};font-weight:600">{v_esc}</span>'
                )
                chip_w.setTextFormat(Qt.RichText)
                chip_w.setStyleSheet(
                    f"background: {P.bg_secondary}; "
                    f"border: 1px solid {chip_border}; "
                    f"border-radius: 6px; padding: 3px 9px; "
                    f"font-family: {T.mono}; font-size: {T.sm}px;"
                )
                chip_w.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
                args_row.addWidget(chip_w)
            args_row.addStretch(1)
            layout.addLayout(args_row)

        result_text = step.error if is_err else step.message
        if result_text:
            result_color = P.danger if is_err else P.text_secondary
            result_weight = "500" if is_err else "400"
            result_lbl = QLabel(f"-> {result_text}")
            result_lbl.setWordWrap(True)
            result_lbl.setStyleSheet(
                f"background: transparent; font-size: 13px; "
                f"font-family: {T.mono}; color: {result_color}; "
                f"font-weight: {result_weight};"
            )
            layout.addWidget(result_lbl)


class _AssistantBubble(QFrame):
    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            f"_AssistantBubble {{ background: {P.bg_secondary}; "
            f"border: 1px solid {P.border}; border-radius: 10px; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        chip = _chip("ASSISTANT", fg=P.text_secondary, bg=P.bg_tertiary, border=P.border)
        layout.addWidget(chip, alignment=Qt.AlignLeft)

        body = QLabel(text)
        body.setWordWrap(True)
        body.setStyleSheet(
            f"background: transparent; font-size: {T.md}px; "
            f"color: {P.text_primary}; line-height: 150%;"
        )
        layout.addWidget(body)


class _SystemMessage(QLabel):
    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setStyleSheet(
            f"color: {P.text_muted}; font-size: {T.xs}px; "
            f"font-style: italic; padding: 6px 4px; background: transparent;"
        )


class _TurnFooter(QLabel):
    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setStyleSheet(
            f"color: {P.text_muted}; font-size: {T.xs}px; "
            f"font-style: italic; padding: 0 4px; background: transparent;"
        )


class _EmptyState(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(S.lg, 40, S.lg, 40)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("Ready when you are")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"font-size: 20px; font-weight: 600; color: {P.text_secondary}; "
            "background: transparent;"
        )

        sub = QLabel('Try: "turn on the lights" or "beep three times"')
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(
            f"font-size: {T.sm}px; color: {P.text_muted}; "
            f"font-family: {T.mono}; background: transparent;"
        )

        layout.addWidget(title)
        layout.addWidget(sub)


class _ThinkingBubble(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            f"_ThinkingBubble {{ background: {P.bg_tertiary}; "
            f"border: 1px solid {P.border}; border-radius: 10px; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        head = QHBoxLayout()
        head.setSpacing(10)
        head.addWidget(_chip("THINKING", fg=P.text_secondary,
                             bg=P.bg_secondary, border=P.border))
        head.addStretch(1)
        layout.addLayout(head)

        dots_row = QHBoxLayout()
        dots_row.setSpacing(6)
        dots_row.setContentsMargins(2, 4, 0, 2)
        self._dots: list[QLabel] = []
        for _ in range(3):
            dot = QLabel()
            dot.setFixedSize(8, 8)
            dot.setStyleSheet(f"background: {P.text_muted}; border-radius: 4px;")
            self._dots.append(dot)
            dots_row.addWidget(dot)
        dots_row.addStretch(1)
        layout.addLayout(dots_row)

        self._step = 0
        self._timer = QTimer(self)
        self._timer.setInterval(400)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self) -> None:
        self._step = (self._step + 1) % 3
        for i, dot in enumerate(self._dots):
            color = P.text_secondary if i == self._step else P.text_muted
            dot.setStyleSheet(f"background: {color}; border-radius: 4px;")

    def stop_animation(self) -> None:
        self._timer.stop()


def _arg_display(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


class CommandLog(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("CommandLog")
        self.setFrameShape(QFrame.StyledPanel)

        self._turns: deque[LogTurn] = deque(maxlen=MAX_TURNS)
        self._thinking_bubble: _ThinkingBubble | None = None
        self._thinking_wrapper: QWidget | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QWidget()
        header.setStyleSheet("background: transparent;")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(18, 14, 18, 10)
        self._header_label = QLabel("CONVERSATION")
        self._header_label.setObjectName("SectionHeader")
        self._header_meta = QLabel("0 turns")
        self._header_meta.setObjectName("HeaderMeta")
        h_layout.addWidget(self._header_label)
        h_layout.addStretch(1)
        h_layout.addWidget(self._header_meta)

        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {P.border};")

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        self._container = QWidget()
        self._container.setStyleSheet("background: transparent;")
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(16, 16, 16, 16)
        self._layout.setSpacing(6)
        self._layout.addStretch(1)

        self._scroll.setWidget(self._container)

        root.addWidget(header)
        root.addWidget(sep)
        root.addWidget(self._scroll, stretch=1)

        self._empty_state: _EmptyState | None = None
        self.clear()

    def append_user_bubble(self, text: str) -> None:
        self._remove_empty_state()
        bubble = _UserBubble(text)
        self._insert_widget(bubble, margin_right=S.lg)

    def append_turn(self, turn: LogTurn) -> None:
        self._turns.append(turn)
        self._render_steps(turn)
        self._update_header()

    def append_system(self, text: str) -> None:
        self._remove_empty_state()
        self._insert_widget(_SystemMessage(text))

    def append_error(self, text: str) -> None:
        self._remove_empty_state()
        step = ToolStep(name="(backend)", status="error", error=text)
        self._insert_widget(_ToolBubble(step), margin_left=S.lg)

    def show_thinking(self) -> None:
        if self._thinking_wrapper is not None:
            return
        self._remove_empty_state()
        bubble = _ThinkingBubble()
        self._thinking_bubble = bubble
        wrapper = QWidget()
        wrapper.setStyleSheet("background: transparent;")
        w_layout = QVBoxLayout(wrapper)
        w_layout.setContentsMargins(S.lg, 0, 0, 0)
        w_layout.setSpacing(0)
        w_layout.addWidget(bubble)
        self._thinking_wrapper = wrapper
        idx = max(0, self._layout.count() - 1)
        self._layout.insertWidget(idx, wrapper)
        QTimer.singleShot(10, self._scroll_bottom)

    def hide_thinking(self) -> None:
        if self._thinking_wrapper is None:
            return
        if self._thinking_bubble is not None:
            self._thinking_bubble.stop_animation()
        idx = self._layout.indexOf(self._thinking_wrapper)
        if idx >= 0:
            self._layout.takeAt(idx)
        self._thinking_wrapper.hide()
        self._thinking_wrapper.deleteLater()
        self._thinking_wrapper = None
        self._thinking_bubble = None

    def clear(self) -> None:
        self.hide_thinking()
        self._turns.clear()
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            w = item.widget()
            if w:
                w.hide()
                w.deleteLater()
        self._empty_state = _EmptyState()
        self._layout.insertWidget(0, self._empty_state)
        self._update_header()

    def _remove_empty_state(self) -> None:
        if self._empty_state is not None:
            self._layout.removeWidget(self._empty_state)
            self._empty_state.hide()
            self._empty_state.deleteLater()
            self._empty_state = None

    def _insert_widget(self, widget: QWidget, *,
                       margin_left: int = 0, margin_right: int = 0) -> None:
        if margin_left or margin_right:
            wrapper = QWidget()
            wrapper.setStyleSheet("background: transparent;")
            w_layout = QVBoxLayout(wrapper)
            w_layout.setContentsMargins(margin_left, 0, margin_right, 0)
            w_layout.setSpacing(0)
            w_layout.addWidget(widget)
            target = wrapper
        else:
            target = widget
        idx = max(0, self._layout.count() - 1)
        self._layout.insertWidget(idx, target)
        QTimer.singleShot(10, self._scroll_bottom)

    def _render_steps(self, turn: LogTurn) -> None:
        for step in turn.steps:
            if step.is_respond:
                msg = step.args.get("message", "") or step.message
                self._insert_widget(_AssistantBubble(str(msg)), margin_left=S.lg)
            else:
                self._insert_widget(_ToolBubble(step), margin_left=S.lg)

        non_respond = [s for s in turn.steps if not s.is_respond]
        if non_respond:
            n = len(non_respond)
            plural = "s" if n != 1 else ""
            footer = f"{n} action{plural} - {turn.latency_ms:.0f} ms"
        else:
            footer = f"{turn.latency_ms:.0f} ms"
        self._insert_widget(_TurnFooter(footer), margin_left=S.lg)

    def _update_header(self) -> None:
        n = len(self._turns)
        self._header_meta.setText(f"{n} turn{'s' if n != 1 else ''}")

    def _scroll_bottom(self) -> None:
        sb = self._scroll.verticalScrollBar()
        sb.setValue(sb.maximum())
