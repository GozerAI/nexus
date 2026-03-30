"""Card widgets for displaying information."""

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from nexus.gui.styles import COLORS, STATUS_COLORS, PRIORITY_COLORS


class StatCard(QFrame):
    """Card displaying a single statistic."""

    clicked = Signal()  # Emitted when card is clicked

    def __init__(self, title: str, value: str = "0", icon: str = None, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setMinimumWidth(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Icon and title row
        header = QHBoxLayout()
        if icon:
            icon_label = QLabel(icon)
            icon_label.setStyleSheet(f"font-size: 20px; color: {COLORS['accent_primary']};")
            header.addWidget(icon_label)

        title_label = QLabel(title.upper())
        title_label.setObjectName("stat_label")
        header.addWidget(title_label)
        header.addStretch()
        layout.addLayout(header)

        # Value
        self.value_label = QLabel(str(value))
        self.value_label.setObjectName("stat_value")
        layout.addWidget(self.value_label)

    def set_value(self, value):
        self.value_label.setText(str(value))

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


class TaskCard(QFrame):
    """Card displaying a task."""

    clicked = Signal(str)  # task_id
    start_clicked = Signal(str)
    complete_clicked = Signal(str)

    def __init__(self, task, parent=None):
        super().__init__(parent)
        self.task = task
        self.setObjectName("card")
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Header row with title and priority
        header = QHBoxLayout()

        # Priority badge
        priority_badge = QLabel(task.priority.value.upper())
        priority_color = PRIORITY_COLORS.get(task.priority.value, COLORS['text_muted'])
        priority_badge.setStyleSheet(f"""
            background-color: {priority_color};
            color: {COLORS['bg_primary']};
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 700;
        """)
        header.addWidget(priority_badge)

        # Status badge
        status_badge = QLabel(task.status.value.replace("_", " ").upper())
        status_color = STATUS_COLORS.get(task.status.value, COLORS['text_muted'])
        status_badge.setStyleSheet(f"""
            background-color: transparent;
            color: {status_color};
            padding: 2px 8px;
            border: 1px solid {status_color};
            border-radius: 4px;
            font-size: 10px;
            font-weight: 600;
        """)
        header.addWidget(status_badge)
        header.addStretch()

        layout.addLayout(header)

        # Title
        title_label = QLabel(task.title)
        title_label.setStyleSheet("font-size: 14px; font-weight: 600;")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        # Description (truncated)
        if task.description:
            desc = task.description[:100] + "..." if len(task.description) > 100 else task.description
            desc_label = QLabel(desc)
            desc_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
            desc_label.setWordWrap(True)
            layout.addWidget(desc_label)

        # Tags
        if task.tags:
            tags_layout = QHBoxLayout()
            for tag in task.tags[:3]:
                tag_label = QLabel(f"#{tag}")
                tag_label.setStyleSheet(f"""
                    color: {COLORS['accent_info']};
                    font-size: 11px;
                """)
                tags_layout.addWidget(tag_label)
            tags_layout.addStretch()
            layout.addLayout(tags_layout)

        # Action buttons
        actions = QHBoxLayout()
        actions.setSpacing(8)

        if task.status.value == "pending":
            start_btn = QPushButton("Start")
            start_btn.setObjectName("success")
            start_btn.clicked.connect(lambda: self.start_clicked.emit(task.id))
            actions.addWidget(start_btn)
        elif task.status.value == "in_progress":
            complete_btn = QPushButton("Complete")
            complete_btn.clicked.connect(lambda: self.complete_clicked.emit(task.id))
            actions.addWidget(complete_btn)

        actions.addStretch()
        layout.addLayout(actions)

    def mousePressEvent(self, event):
        self.clicked.emit(self.task.id)
        super().mousePressEvent(event)


class GoalCard(QFrame):
    """Card displaying a goal with progress."""

    clicked = Signal(str)  # goal_id

    def __init__(self, goal, parent=None):
        super().__init__(parent)
        self.goal = goal
        self.setObjectName("card")
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Header
        header = QHBoxLayout()

        # Status badge
        status_badge = QLabel(goal.status.value.upper())
        status_color = STATUS_COLORS.get(goal.status.value, COLORS['text_muted'])
        status_badge.setStyleSheet(f"""
            background-color: transparent;
            color: {status_color};
            padding: 2px 8px;
            border: 1px solid {status_color};
            border-radius: 4px;
            font-size: 10px;
            font-weight: 600;
        """)
        header.addWidget(status_badge)
        header.addStretch()

        # Progress percentage
        progress_pct = int(goal.progress * 100)
        progress_label = QLabel(f"{progress_pct}%")
        progress_label.setStyleSheet(f"color: {COLORS['accent_primary']}; font-weight: 700;")
        header.addWidget(progress_label)

        layout.addLayout(header)

        # Title
        title_label = QLabel(goal.title)
        title_label.setStyleSheet("font-size: 14px; font-weight: 600;")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        # Description
        if goal.description:
            desc = goal.description[:80] + "..." if len(goal.description) > 80 else goal.description
            desc_label = QLabel(desc)
            desc_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
            desc_label.setWordWrap(True)
            layout.addWidget(desc_label)

        # Progress bar
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(progress_pct)
        progress_bar.setTextVisible(False)
        progress_bar.setFixedHeight(6)
        layout.addWidget(progress_bar)

        # Milestone count
        if goal.milestones:
            completed = sum(1 for m in goal.milestones if m.status.value == "completed")
            total = len(goal.milestones)
            milestone_label = QLabel(f"{completed}/{total} milestones")
            milestone_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
            layout.addWidget(milestone_label)

    def mousePressEvent(self, event):
        self.clicked.emit(self.goal.id)
        super().mousePressEvent(event)
