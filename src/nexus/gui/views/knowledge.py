"""Knowledge graph view."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTableWidget, QTableWidgetItem, QTabWidget,
    QFrame, QComboBox, QTextEdit, QHeaderView, QMessageBox
)
from PySide6.QtCore import Qt, Signal
from nexus.gui.styles import COLORS


class KnowledgeView(QWidget):
    """View for exploring knowledge graph."""

    search_entities = Signal(str, str)  # query, entity_type
    search_facts = Signal(str, str)  # query, topic
    add_fact = Signal(str, str, float)  # statement, topic, confidence

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Header
        header = QHBoxLayout()
        title = QLabel("Knowledge")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()

        add_fact_btn = QPushButton("+ Add Fact")
        add_fact_btn.clicked.connect(self.show_add_fact_dialog)
        header.addWidget(add_fact_btn)

        layout.addLayout(header)

        # Tabs for entities and facts
        tabs = QTabWidget()

        # Entities tab
        entities_tab = QWidget()
        entities_layout = QVBoxLayout(entities_tab)

        # Entity search
        entity_search = QHBoxLayout()
        self.entity_search_input = QLineEdit()
        self.entity_search_input.setPlaceholderText("Search entities...")
        self.entity_search_input.returnPressed.connect(self.on_entity_search)
        entity_search.addWidget(self.entity_search_input)

        self.entity_type_filter = QComboBox()
        self.entity_type_filter.addItems([
            "All Types", "technology", "project", "person",
            "organization", "concept", "file_path", "decision", "task"
        ])
        entity_search.addWidget(self.entity_type_filter)

        entity_search_btn = QPushButton("Search")
        entity_search_btn.clicked.connect(self.on_entity_search)
        entity_search.addWidget(entity_search_btn)

        entities_layout.addLayout(entity_search)

        # Entity table
        self.entity_table = QTableWidget()
        self.entity_table.setColumnCount(3)
        self.entity_table.setHorizontalHeaderLabels(["Name", "Type", "Created"])
        self.entity_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.entity_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.entity_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.entity_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.entity_table.setAlternatingRowColors(True)
        entities_layout.addWidget(self.entity_table)

        tabs.addTab(entities_tab, "Entities")

        # Facts tab
        facts_tab = QWidget()
        facts_layout = QVBoxLayout(facts_tab)

        # Fact search
        fact_search = QHBoxLayout()
        self.fact_search_input = QLineEdit()
        self.fact_search_input.setPlaceholderText("Search facts...")
        self.fact_search_input.returnPressed.connect(self.on_fact_search)
        fact_search.addWidget(self.fact_search_input)

        self.fact_topic_input = QLineEdit()
        self.fact_topic_input.setPlaceholderText("Topic filter...")
        self.fact_topic_input.setMaximumWidth(150)
        fact_search.addWidget(self.fact_topic_input)

        fact_search_btn = QPushButton("Search")
        fact_search_btn.clicked.connect(self.on_fact_search)
        fact_search.addWidget(fact_search_btn)

        facts_layout.addLayout(fact_search)

        # Fact table
        self.fact_table = QTableWidget()
        self.fact_table.setColumnCount(4)
        self.fact_table.setHorizontalHeaderLabels(["Statement", "Topic", "Confidence", "Verified"])
        self.fact_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.fact_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.fact_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.fact_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.fact_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.fact_table.setAlternatingRowColors(True)
        facts_layout.addWidget(self.fact_table)

        tabs.addTab(facts_tab, "Facts")

        # Stats tab
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setPlaceholderText("Loading statistics...")
        stats_layout.addWidget(self.stats_text)

        tabs.addTab(stats_tab, "Statistics")

        layout.addWidget(tabs)

    def on_entity_search(self):
        query = self.entity_search_input.text().strip()
        entity_type = self.entity_type_filter.currentText()
        if entity_type == "All Types":
            entity_type = None
        self.search_entities.emit(query, entity_type)

    def on_fact_search(self):
        query = self.fact_search_input.text().strip()
        topic = self.fact_topic_input.text().strip() or None
        self.search_facts.emit(query, topic)

    def update_entities(self, entities):
        """Update the entities table."""
        self.entity_table.setRowCount(len(entities))

        for row, entity in enumerate(entities):
            self.entity_table.setItem(row, 0, QTableWidgetItem(entity.name))
            self.entity_table.setItem(row, 1, QTableWidgetItem(entity.entity_type.value))
            self.entity_table.setItem(row, 2, QTableWidgetItem(
                entity.created_at.strftime("%Y-%m-%d")
            ))

    def update_facts(self, facts):
        """Update the facts table."""
        self.fact_table.setRowCount(len(facts))

        for row, fact in enumerate(facts):
            self.fact_table.setItem(row, 0, QTableWidgetItem(fact.statement))
            self.fact_table.setItem(row, 1, QTableWidgetItem(fact.topic or ""))
            self.fact_table.setItem(row, 2, QTableWidgetItem(f"{int(fact.confidence * 100)}%"))
            self.fact_table.setItem(row, 3, QTableWidgetItem(str(fact.verification_count)))

    def update_stats(self, stats):
        """Update the statistics display."""
        html = f"""
        <style>
            h3 {{ color: {COLORS['accent_primary']}; }}
            .stat {{ margin: 8px 0; }}
            .value {{ color: {COLORS['accent_primary']}; font-weight: bold; }}
        </style>
        <h3>Knowledge Graph Statistics</h3>
        <p class="stat">Total Entities: <span class="value">{stats.get('total_entities', 0)}</span></p>
        <p class="stat">Total Relations: <span class="value">{stats.get('total_relations', 0)}</span></p>
        <p class="stat">Total Facts: <span class="value">{stats.get('total_facts', 0)}</span></p>
        """

        if stats.get('entities_by_type'):
            html += "<h3>Entities by Type</h3>"
            for etype, count in stats['entities_by_type'].items():
                html += f"<p class='stat'>{etype}: <span class='value'>{count}</span></p>"

        self.stats_text.setHtml(html)

    def show_add_fact_dialog(self):
        """Show dialog to add a new fact."""
        from PySide6.QtWidgets import QDialog, QFormLayout, QDialogButtonBox, QDoubleSpinBox

        dialog = QDialog(self)
        dialog.setWindowTitle("Add Fact")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        statement_input = QTextEdit()
        statement_input.setMaximumHeight(80)
        statement_input.setPlaceholderText("Enter the fact statement...")
        form.addRow("Statement:", statement_input)

        topic_input = QLineEdit()
        topic_input.setPlaceholderText("Optional topic...")
        form.addRow("Topic:", topic_input)

        confidence_input = QDoubleSpinBox()
        confidence_input.setRange(0, 1)
        confidence_input.setSingleStep(0.1)
        confidence_input.setValue(0.5)
        form.addRow("Confidence:", confidence_input)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec():
            statement = statement_input.toPlainText().strip()
            topic = topic_input.text().strip() or None
            confidence = confidence_input.value()

            if statement:
                self.add_fact.emit(statement, topic, confidence)
            else:
                QMessageBox.warning(self, "Error", "Statement is required")
