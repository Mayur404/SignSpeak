"""ASL alphabet and digit reference page."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from ...paths import ASL_CHART
from ..widgets import Card, PageHeader, RoundedImage


class GuidePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        root.addWidget(
            PageHeader(
                "ASL alphabet & digits",
                "American Sign Language letter and number reference.",
            )
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("Content")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 8, 8)

        chart_card = Card(padding=18, spacing=10)
        chart = RoundedImage()
        chart.setMinimumHeight(420)
        chart.set_image(str(ASL_CHART))
        chart_card.body.addWidget(chart)
        layout.addWidget(chart_card)
        layout.addStretch(1)

        scroll.setWidget(body)
        root.addWidget(scroll, 1)
