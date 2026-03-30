"""Dialog windows for Nexus GUI."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QTextEdit, QComboBox, QPushButton, QFormLayout, QDialogButtonBox
)
from PySide6.QtCore import Qt
from nexus.gui.styles import COLORS


class BaseDialog(QDialog):
    """Base dialog with common styling."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(400)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['bg_primary']};
            }}
        """)


class TaskDialog(BaseDialog):
    """Dialog for creating/editing a task."""

    def __init__(self, task=None, parent=None):
        super().__init__("Edit Task" if task else "New Task", parent)
        self.task = task

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Form
        form = QFormLayout()
        form.setSpacing(12)

        # Title
        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("Task title...")
        if task:
            self.title_input.setText(task.title)
        form.addRow("Title:", self.title_input)

        # Description
        self.desc_input = QTextEdit()
        self.desc_input.setPlaceholderText("Task description...")
        self.desc_input.setMaximumHeight(100)
        if task and task.description:
            self.desc_input.setText(task.description)
        form.addRow("Description:", self.desc_input)

        # Priority
        self.priority_combo = QComboBox()
        self.priority_combo.addItems(["critical", "high", "medium", "low", "backlog"])
        if task:
            self.priority_combo.setCurrentText(task.priority.value)
        else:
            self.priority_combo.setCurrentText("medium")
        form.addRow("Priority:", self.priority_combo)

        # Tags
        self.tags_input = QLineEdit()
        self.tags_input.setPlaceholderText("tag1, tag2, tag3...")
        if task and task.tags:
            self.tags_input.setText(", ".join(task.tags))
        form.addRow("Tags:", self.tags_input)

        # Project path
        self.project_input = QLineEdit()
        self.project_input.setPlaceholderText("/path/to/project")
        if task and task.project_path:
            self.project_input.setText(task.project_path)
        form.addRow("Project:", self.project_input)

        layout.addLayout(form)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_data(self):
        """Get the form data."""
        tags = [t.strip() for t in self.tags_input.text().split(",") if t.strip()]
        return {
            "title": self.title_input.text(),
            "description": self.desc_input.toPlainText(),
            "priority": self.priority_combo.currentText(),
            "tags": tags,
            "project_path": self.project_input.text() or None
        }


class GoalDialog(BaseDialog):
    """Dialog for creating/editing a goal."""

    def __init__(self, goal=None, parent=None):
        super().__init__("Edit Goal" if goal else "New Goal", parent)
        self.goal = goal

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        form = QFormLayout()
        form.setSpacing(12)

        # Title
        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("Goal title...")
        if goal:
            self.title_input.setText(goal.title)
        form.addRow("Title:", self.title_input)

        # Description
        self.desc_input = QTextEdit()
        self.desc_input.setPlaceholderText("Goal description...")
        self.desc_input.setMaximumHeight(100)
        if goal and goal.description:
            self.desc_input.setText(goal.description)
        form.addRow("Description:", self.desc_input)

        # Project path
        self.project_input = QLineEdit()
        self.project_input.setPlaceholderText("/path/to/project")
        if goal and goal.project_path:
            self.project_input.setText(goal.project_path)
        form.addRow("Project:", self.project_input)

        layout.addLayout(form)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_data(self):
        return {
            "title": self.title_input.text(),
            "description": self.desc_input.toPlainText(),
            "project_path": self.project_input.text() or None
        }


class DecisionDialog(BaseDialog):
    """Dialog for recording a decision."""

    def __init__(self, parent=None):
        super().__init__("Record Decision", parent)
        self.setMinimumWidth(500)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        form = QFormLayout()
        form.setSpacing(12)

        # Question
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("What was the question or problem?")
        form.addRow("Question:", self.question_input)

        # Decision
        self.decision_input = QTextEdit()
        self.decision_input.setPlaceholderText("What was decided?")
        self.decision_input.setMaximumHeight(80)
        form.addRow("Decision:", self.decision_input)

        # Reasoning
        self.reasoning_input = QTextEdit()
        self.reasoning_input.setPlaceholderText("Why was this decision made?")
        self.reasoning_input.setMaximumHeight(100)
        form.addRow("Reasoning:", self.reasoning_input)

        # Alternatives
        self.alternatives_input = QTextEdit()
        self.alternatives_input.setPlaceholderText("What alternatives were considered? (one per line)")
        self.alternatives_input.setMaximumHeight(80)
        form.addRow("Alternatives:", self.alternatives_input)

        # Project
        self.project_input = QLineEdit()
        self.project_input.setPlaceholderText("/path/to/project")
        form.addRow("Project:", self.project_input)

        layout.addLayout(form)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_data(self):
        alternatives = [a.strip() for a in self.alternatives_input.toPlainText().split("\n") if a.strip()]
        return {
            "question": self.question_input.text(),
            "decision": self.decision_input.toPlainText(),
            "reasoning": self.reasoning_input.toPlainText(),
            "alternatives": alternatives,
            "project_path": self.project_input.text() or None
        }


class SearchDialog(BaseDialog):
    """Dialog for searching memories."""

    def __init__(self, parent=None):
        super().__init__("Search Memories", parent)
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Search input
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search for topics, conversations, or context...")
        self.search_input.returnPressed.connect(self.on_search)
        search_layout.addWidget(self.search_input)

        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self.on_search)
        search_layout.addWidget(self.search_btn)

        layout.addLayout(search_layout)

        # Results area
        self.results_label = QLabel("Enter a search query above")
        self.results_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        self.results_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.results_label)

        self.results_area = QTextEdit()
        self.results_area.setReadOnly(True)
        self.results_area.hide()
        layout.addWidget(self.results_area)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setObjectName("secondary")
        close_btn.clicked.connect(self.reject)
        layout.addWidget(close_btn)

        self._search_callback = None

    def set_search_callback(self, callback):
        """Set the callback for search operations."""
        self._search_callback = callback

    def on_search(self):
        query = self.search_input.text().strip()
        if query and self._search_callback:
            self._search_callback(query)

    def show_results(self, results):
        """Display search results."""
        if not results:
            self.results_area.hide()
            self.results_label.setText("No results found")
            self.results_label.show()
            return

        self.results_label.hide()
        self.results_area.show()

        html = "<style>p { margin: 0 0 12px 0; } .snippet { color: #a6adc8; }</style>"
        for i, hit in enumerate(results, 1):
            relevance = int(hit.relevance * 100)
            html += f"""
            <div style="margin-bottom: 16px; padding: 12px; background: #313244; border-radius: 8px;">
                <b style="color: #89b4fa;">Result {i}</b>
                <span style="color: #6c7086; margin-left: 8px;">({relevance}% relevant)</span>
                <p class="snippet" style="margin-top: 8px;">{hit.snippet}</p>
            </div>
            """

        self.results_area.setHtml(html)
