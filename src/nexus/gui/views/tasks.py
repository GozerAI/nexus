"""Tasks management view."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QGridLayout, QComboBox, QLineEdit, QMessageBox
)
from PySide6.QtCore import Qt, Signal
from nexus.gui.styles import COLORS
from nexus.gui.widgets.cards import TaskCard
from nexus.gui.widgets.dialogs import TaskDialog


class TasksView(QWidget):
    """View for managing tasks."""

    refresh_requested = Signal()
    create_task = Signal(dict)
    update_task = Signal(str, dict)  # task_id, data
    start_task = Signal(str)
    complete_task = Signal(str)
    task_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Header
        header = QHBoxLayout()
        title = QLabel("Tasks")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()

        # Filters
        self.status_filter = QComboBox()
        self.status_filter.addItems(["All Active", "Pending", "In Progress", "Blocked", "Completed"])
        self.status_filter.currentTextChanged.connect(self.on_filter_changed)
        header.addWidget(self.status_filter)

        self.priority_filter = QComboBox()
        self.priority_filter.addItems(["All Priorities", "Critical", "High", "Medium", "Low", "Backlog"])
        self.priority_filter.currentTextChanged.connect(self.on_filter_changed)
        header.addWidget(self.priority_filter)

        new_btn = QPushButton("+ New Task")
        new_btn.clicked.connect(self.show_create_dialog)
        header.addWidget(new_btn)

        layout.addLayout(header)

        # Search
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search tasks...")
        self.search_input.textChanged.connect(self.on_search_changed)
        search_layout.addWidget(self.search_input)
        layout.addLayout(search_layout)

        # Scrollable task list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.container = QWidget()
        self.grid = QGridLayout(self.container)
        self.grid.setSpacing(12)
        self.grid.setAlignment(Qt.AlignTop)

        scroll.setWidget(self.container)
        layout.addWidget(scroll)

        # Empty state
        self.empty_label = QLabel("No tasks found")
        self.empty_label.setStyleSheet(f"color: {COLORS['text_muted']}; padding: 40px;")
        self.empty_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.empty_label)

        self._all_tasks = []

    def update_tasks(self, tasks):
        """Update the task list."""
        self._all_tasks = tasks
        self.filter_and_display()

    def filter_and_display(self):
        """Apply filters and display tasks."""
        tasks = self._all_tasks

        # Status filter
        status = self.status_filter.currentText().lower().replace(" ", "_")
        if status != "all_active":
            tasks = [t for t in tasks if t.status.value == status]

        # Priority filter
        priority = self.priority_filter.currentText().lower()
        if priority != "all priorities":
            tasks = [t for t in tasks if t.priority.value == priority]

        # Search filter
        search = self.search_input.text().lower()
        if search:
            tasks = [t for t in tasks if search in t.title.lower() or
                     (t.description and search in t.description.lower())]

        self.display_tasks(tasks)

    def display_tasks(self, tasks):
        """Display tasks in the grid."""
        # Clear existing
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not tasks:
            self.empty_label.show()
            return

        self.empty_label.hide()

        # Add task cards (2 columns)
        for i, task in enumerate(tasks):
            card = TaskCard(task)
            card.clicked.connect(self.task_selected.emit)
            card.start_clicked.connect(self.start_task.emit)
            card.complete_clicked.connect(self.complete_task.emit)
            self.grid.addWidget(card, i // 2, i % 2)

    def on_filter_changed(self):
        self.filter_and_display()

    def on_search_changed(self):
        self.filter_and_display()

    def show_create_dialog(self):
        """Show the create task dialog."""
        dialog = TaskDialog(parent=self)
        if dialog.exec():
            data = dialog.get_data()
            if data["title"]:
                self.create_task.emit(data)
            else:
                QMessageBox.warning(self, "Error", "Task title is required")
