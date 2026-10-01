# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Qt-binding compatibility shim and shared button styling for the Face ID UI."""

from __future__ import annotations

import locale
import os


def ensure_utf8_locale():
    lang = (os.environ.get("LANG") or "").strip()
    lc_all = (os.environ.get("LC_ALL") or "").strip()
    env_ascii = lang in ("", "C", "POSIX") or lc_all in ("C", "POSIX")
    encoding = locale.getpreferredencoding(False).lower()
    if "utf" in encoding and not env_ascii:
        return

    for candidate in ("en_US.utf8", "en_US.UTF-8", "C.UTF-8"):
        try:
            locale.setlocale(locale.LC_CTYPE, candidate)
            os.environ["LC_ALL"] = candidate
            os.environ["LANG"] = candidate
            return
        except locale.Error:
            continue

    os.environ["LC_ALL"] = "en_US.utf8"
    os.environ["LANG"] = "en_US.utf8"


def resolve_rgb888_format(qt):
    if hasattr(qt.QImage, "Format_RGB888"):
        return qt.QImage.Format_RGB888
    if hasattr(qt.QImage, "Format") and hasattr(qt.QImage.Format, "Format_RGB888"):
        return qt.QImage.Format.Format_RGB888
    raise RuntimeError("QImage RGB888 format is unavailable in this Qt binding")


def import_qt():
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets

        class _Qt:
            QApplication = QtWidgets.QApplication
            QWidget = QtWidgets.QWidget
            QLabel = QtWidgets.QLabel
            QPushButton = QtWidgets.QPushButton
            QLineEdit = QtWidgets.QLineEdit
            QHBoxLayout = QtWidgets.QHBoxLayout
            QVBoxLayout = QtWidgets.QVBoxLayout
            QScrollArea = QtWidgets.QScrollArea
            QSizePolicy = QtWidgets.QSizePolicy
            QStyle = QtWidgets.QStyle
            QIcon = QtGui.QIcon
            QImage = QtGui.QImage
            QPixmap = QtGui.QPixmap
            QSize = QtCore.QSize
            Qt = QtCore.Qt

        return _Qt
    except ModuleNotFoundError:
        pass

    try:
        from PyQt5 import QtCore, QtGui, QtWidgets

        class _Qt:
            QApplication = QtWidgets.QApplication
            QWidget = QtWidgets.QWidget
            QLabel = QtWidgets.QLabel
            QPushButton = QtWidgets.QPushButton
            QLineEdit = QtWidgets.QLineEdit
            QHBoxLayout = QtWidgets.QHBoxLayout
            QVBoxLayout = QtWidgets.QVBoxLayout
            QScrollArea = QtWidgets.QScrollArea
            QSizePolicy = QtWidgets.QSizePolicy
            QStyle = QtWidgets.QStyle
            QIcon = QtGui.QIcon
            QImage = QtGui.QImage
            QPixmap = QtGui.QPixmap
            QSize = QtCore.QSize
            Qt = QtCore.Qt

        return _Qt
    except ModuleNotFoundError:
        pass

    try:
        from PySide6 import QtCore, QtGui, QtWidgets

        class _Qt:
            QApplication = QtWidgets.QApplication
            QWidget = QtWidgets.QWidget
            QLabel = QtWidgets.QLabel
            QPushButton = QtWidgets.QPushButton
            QLineEdit = QtWidgets.QLineEdit
            QHBoxLayout = QtWidgets.QHBoxLayout
            QVBoxLayout = QtWidgets.QVBoxLayout
            QScrollArea = QtWidgets.QScrollArea
            QSizePolicy = QtWidgets.QSizePolicy
            QStyle = QtWidgets.QStyle
            QIcon = QtGui.QIcon
            QImage = QtGui.QImage
            QPixmap = QtGui.QPixmap
            QSize = QtCore.QSize
            Qt = QtCore.Qt

        return _Qt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Qt UI requires PyQt6, PyQt5, or PySide6. Install one of them before using --qt-enrollment-ui."
        ) from exc


def button_style(round_px=6):
    return (
        "QPushButton {"
        "background: #3f76b2; color: #ffffff; border: 1px solid #2f5f93;"
        f"border-radius: {round_px}px; padding: 8px 10px; font-weight: 700;"
        "}"
        "QPushButton:hover { background: #6f9fd4; }"
        "QPushButton:pressed { background: #6f9fd4; }"
        "QPushButton:disabled { background: #2a3950; color: #8aa0bb; border: 1px solid #2a3950; }"
    )


def set_button_icon(qt, app, button, kind):
    style = app.style()
    theme_icon = qt.QIcon()
    if kind == "enroll":
        for name in ("camera-photo", "camera", "media-record"):
            theme_icon = qt.QIcon.fromTheme(name)
            if not theme_icon.isNull():
                break
        fallback = qt.QStyle.StandardPixmap.SP_DialogApplyButton
    elif kind == "register":
        for name in ("system-users", "user-group-new", "contact-new"):
            theme_icon = qt.QIcon.fromTheme(name)
            if not theme_icon.isNull():
                break
        fallback = qt.QStyle.StandardPixmap.SP_FileDialogDetailedView
    else:
        fallback = qt.QStyle.StandardPixmap.SP_DialogOkButton

    button.setIcon(theme_icon if not theme_icon.isNull() else style.standardIcon(fallback))
    button.setIconSize(qt.QSize(16, 16))
