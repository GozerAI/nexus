"""Decisions view."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QTextEdit, QSplitter, QFrame
)
from PySide6.QtCore import Qt, Signal
from nexus.gui.styles import COLORS
from nexus.gui.widgets.dialogs import DecisionDialog


class DecisionsView(QWidget):
    """View for browsing and recording decisions."""

    refresh_requested = Signal()
    record_decision = Signal(dict)
    search_decisions = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self._decisions = []

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Header
        header = QHBoxLayout()
        title = QLabel("Decisions")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()

        new_btn = QPushButton("+ Record Decision")
        new_btn.clicked.connect(self.show_record_dialog)
        header.addWidget(new_btn)

        layout.addLayout(header)

        # Search
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search decisions...")
        self.search_input.returnPressed.connect(self.on_search)
        search_layout.addWidget(self.search_input)

        search_btn = QPushButton("Search")
        search_btn.setObjectName("secondary")
        search_btn.clicked.connect(self.on_search)
        search_layout.addWidget(search_btn)

        layout.addLayout(search_layout)

        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)

        # Decisions table
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Question", "Decision", "Date"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.itemSelectionChanged.connect(self.on_selection_changed)
        table_layout.addWidget(self.table)

        splitter.addWidget(table_container)

        # Detail panel
        detail_container = QFrame()
        detail_container.setObjectName("card")
        detail_layout = QVBoxLayout(detail_container)

        detail_title = QLabel("Decision Details")
        detail_title.setStyleSheet("font-weight: 600; font-size: 14px;")
        detail_layout.addWidget(detail_title)

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['bg_tertiary']};
                border: none;
                border-radius: 8px;
            }}
        """)
        self.detail_text.setPlaceholderText("Select a decision to see details")
        detail_layout.addWidget(self.detail_text)

        splitter.addWidget(detail_container)
        splitter.setSizes([500, 400])

        layout.addWidget(splitter)

    def update_decisions(self, decisions):
        """Update the decisions table."""
        self._decisions = decisions
        self.table.setRowCount(len(decisions))

        for row, dec in enumerate(decisions):
            # Truncate long text for display
            question = dec.question[:50] + "..." if len(dec.question) > 50 else dec.question
            decision = dec.decision[:50] + "..." if len(dec.decision) > 50 else dec.decision

            self.table.setItem(row, 0, QTableWidgetItem(question))
            self.table.setItem(row, 1, QTableWidgetItem(decision))
            self.table.setItem(row, 2, QTableWidgetItem(
                dec.created_at.strftime("%Y-%m-%d")
            ))

    def on_selection_changed(self):
        """Handle selection change to show details."""
        rows = self.table.selectedIndexes()
        if not rows:
            return

        row = rows[0].row()
        if row >= len(self._decisions):
            return

        dec = self._decisions[row]

        alternatives_html = ""
        if dec.alternatives:
            alternatives_html = "<h4>Alternatives Considered:</h4><ul>"
            for alt in dec.alternatives:
                alternatives_html += f"<li>{alt}</li>"
            alternatives_html += "</ul>"

        html = f"""
        <style>
            h3 {{ color: {COLORS['accent_primary']}; margin: 8px 0; }}
            h4 {{ color: {COLORS['text_secondary']}; margin: 12px 0 8px 0; }}
            p {{ margin: 8px 0; }}
            .meta {{ color: {COLORS['text_muted']}; font-size: 11px; }}
        </style>
        <h3>Question</h3>
        <p>{dec.question}</p>

        <h3>Decision</h3>
        <p>{dec.decision}</p>

        <h4>Reasoning:</h4>
        <p>{dec.reasoning}</p>

        {alternatives_html}

        <p class="meta">Recorded: {dec.created_at.strftime("%Y-%m-%d %H:%M")}</p>
        <p class="meta">Project: {dec.project_path or 'None'}</p>
        """

        self.detail_text.setHtml(html)

    def on_search(self):
        query = self.search_input.text().strip()
        self.search_decisions.emit(query)

    def show_record_dialog(self):
        """Show the record decision dialog."""
        dialog = DecisionDialog(parent=self)
        if dialog.exec():
            data = dialog.get_data()
            if data["question"] and data["decision"] and data["reasoning"]:
                self.record_decision.emit(data)
