"""Dashboard view showing overview and focus context."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QGridLayout, QPushButton, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from nexus.gui.styles import COLORS
from nexus.gui.widgets.cards import StatCard, TaskCard, GoalCard


class DashboardView(QWidget):
    """Main dashboard showing system overview."""

    refresh_requested = Signal()
    task_selected = Signal(str)
    goal_selected = Signal(str)
    # Navigation signals when stat cards are clicked
    navigate_to_tasks = Signal()
    navigate_to_goals = Signal()
    navigate_to_memory = Signal()
    navigate_to_decisions = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(24)
        layout.setContentsMargins(24, 24, 24, 24)

        # Header
        header = QHBoxLayout()
        title = QLabel("Dashboard")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setObjectName("secondary")
        refresh_btn.clicked.connect(self.refresh_requested.emit)
        header.addWidget(refresh_btn)

        layout.addLayout(header)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(24)

        # Stats row
        self.stats_layout = QHBoxLayout()
        self.stats_layout.setSpacing(16)

        self.stat_tasks = StatCard("Active Tasks", "0", "")
        self.stat_goals = StatCard("Goals", "0", "")
        self.stat_memory = StatCard("Memories", "0", "")
        self.stat_decisions = StatCard("Decisions", "0", "")

        # Connect stat card clicks to navigation
        self.stat_tasks.clicked.connect(self.navigate_to_tasks.emit)
        self.stat_goals.clicked.connect(self.navigate_to_goals.emit)
        self.stat_memory.clicked.connect(self.navigate_to_memory.emit)
        self.stat_decisions.clicked.connect(self.navigate_to_decisions.emit)

        self.stats_layout.addWidget(self.stat_tasks)
        self.stats_layout.addWidget(self.stat_goals)
        self.stats_layout.addWidget(self.stat_memory)
        self.stats_layout.addWidget(self.stat_decisions)

        content_layout.addLayout(self.stats_layout)

        # Focus context section
        focus_section = QFrame()
        focus_section.setObjectName("card")
        focus_layout = QVBoxLayout(focus_section)

        focus_header = QHBoxLayout()
        focus_title = QLabel("Current Focus")
        focus_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        focus_header.addWidget(focus_title)
        focus_header.addStretch()
        focus_layout.addLayout(focus_header)

        self.focus_label = QLabel("Loading focus context...")
        self.focus_label.setStyleSheet(f"color: {COLORS['text_secondary']}; padding: 8px 0;")
        self.focus_label.setWordWrap(True)
        focus_layout.addWidget(self.focus_label)

        self.suggestion_label = QLabel("")
        self.suggestion_label.setStyleSheet(f"""
            background-color: {COLORS['bg_tertiary']};
            color: {COLORS['accent_primary']};
            padding: 12px;
            border-radius: 8px;
            font-weight: 500;
        """)
        self.suggestion_label.setWordWrap(True)
        self.suggestion_label.hide()
        focus_layout.addWidget(self.suggestion_label)

        content_layout.addWidget(focus_section)

        # Active tasks section
        tasks_section = QVBoxLayout()
        tasks_header = QHBoxLayout()
        tasks_title = QLabel("Active Tasks")
        tasks_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        tasks_header.addWidget(tasks_title)
        tasks_header.addStretch()
        tasks_section.addLayout(tasks_header)

        self.tasks_container = QWidget()
        self.tasks_grid = QGridLayout(self.tasks_container)
        self.tasks_grid.setSpacing(12)
        tasks_section.addWidget(self.tasks_container)

        self.no_tasks_label = QLabel("No active tasks")
        self.no_tasks_label.setStyleSheet(f"color: {COLORS['text_muted']}; padding: 20px;")
        self.no_tasks_label.setAlignment(Qt.AlignCenter)
        tasks_section.addWidget(self.no_tasks_label)

        content_layout.addLayout(tasks_section)

        # Active goals section
        goals_section = QVBoxLayout()
        goals_header = QHBoxLayout()
        goals_title = QLabel("Active Goals")
        goals_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        goals_header.addWidget(goals_title)
        goals_header.addStretch()
        goals_section.addLayout(goals_header)

        self.goals_container = QWidget()
        self.goals_grid = QGridLayout(self.goals_container)
        self.goals_grid.setSpacing(12)
        goals_section.addWidget(self.goals_container)

        self.no_goals_label = QLabel("No active goals")
        self.no_goals_label.setStyleSheet(f"color: {COLORS['text_muted']}; padding: 20px;")
        self.no_goals_label.setAlignment(Qt.AlignCenter)
        goals_section.addWidget(self.no_goals_label)

        content_layout.addLayout(goals_section)
        content_layout.addStretch()

        scroll.setWidget(content)
        layout.addWidget(scroll)

    def update_stats(self, tasks_count, goals_count, memory_count, decisions_count):
        """Update the stat cards."""
        self.stat_tasks.set_value(tasks_count)
        self.stat_goals.set_value(goals_count)
        self.stat_memory.set_value(memory_count)
        self.stat_decisions.set_value(decisions_count)

    def update_focus(self, focus_context):
        """Update the focus context section."""
        if not focus_context:
            self.focus_label.setText("Unable to load focus context")
            self.suggestion_label.hide()
            return

        # Build focus text
        lines = []

        if focus_context.active_tasks:
            task_names = [t.title for t in focus_context.active_tasks[:3]]
            lines.append(f"Working on: {', '.join(task_names)}")

        if focus_context.blocked_tasks:
            lines.append(f"Blocked: {len(focus_context.blocked_tasks)} task(s)")

        if focus_context.blockers_summary:
            lines.append(f"Blockers: {len(focus_context.blockers_summary)}")

        if focus_context.last_session_notes:
            lines.append(f"Last session: {focus_context.last_session_notes[:100]}...")

        if lines:
            self.focus_label.setText("\n".join(lines))
        else:
            self.focus_label.setText("No active work context")

        # Show suggestion
        if focus_context.suggested_action:
            self.suggestion_label.setText(f"Suggested: {focus_context.suggested_action}")
            self.suggestion_label.show()
        else:
            self.suggestion_label.hide()

    def update_tasks(self, tasks):
        """Update the active tasks grid."""
        # Clear existing
        while self.tasks_grid.count():
            item = self.tasks_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not tasks:
            self.no_tasks_label.show()
            return

        self.no_tasks_label.hide()

        # Add task cards (2 columns)
        for i, task in enumerate(tasks[:6]):
            card = TaskCard(task)
            card.clicked.connect(self.task_selected.emit)
            self.tasks_grid.addWidget(card, i // 2, i % 2)

    def update_goals(self, goals):
        """Update the goals grid."""
        # Clear existing
        while self.goals_grid.count():
            item = self.goals_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not goals:
            self.no_goals_label.show()
            return

        self.no_goals_label.hide()

        # Add goal cards (2 columns)
        for i, goal in enumerate(goals[:4]):
            card = GoalCard(goal)
            card.clicked.connect(self.goal_selected.emit)
            self.goals_grid.addWidget(card, i // 2, i % 2)
