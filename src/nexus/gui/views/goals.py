"""Goals management view."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QGridLayout, QComboBox, QFrame, QProgressBar,
    QMessageBox, QListWidget, QListWidgetItem
)
from PySide6.QtCore import Qt, Signal
from nexus.gui.styles import COLORS, STATUS_COLORS
from nexus.gui.widgets.cards import GoalCard
from nexus.gui.widgets.dialogs import GoalDialog


class GoalsView(QWidget):
    """View for managing goals."""

    refresh_requested = Signal()
    create_goal = Signal(dict)
    goal_selected = Signal(str)
    add_milestone = Signal(str, str, str)  # goal_id, title, description
    complete_milestone = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self._selected_goal = None

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Left panel - Goals list
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(16)
        left_layout.setContentsMargins(24, 24, 12, 24)

        # Header
        header = QHBoxLayout()
        title = QLabel("Goals")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()

        new_btn = QPushButton("+ New Goal")
        new_btn.clicked.connect(self.show_create_dialog)
        header.addWidget(new_btn)

        left_layout.addLayout(header)

        # Filter
        self.status_filter = QComboBox()
        self.status_filter.addItems(["Active", "Completed", "All"])
        self.status_filter.currentTextChanged.connect(self.on_filter_changed)
        left_layout.addWidget(self.status_filter)

        # Goals scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.goals_container = QWidget()
        self.goals_layout = QVBoxLayout(self.goals_container)
        self.goals_layout.setSpacing(12)
        self.goals_layout.setAlignment(Qt.AlignTop)

        scroll.setWidget(self.goals_container)
        left_layout.addWidget(scroll)

        self.empty_label = QLabel("No goals found")
        self.empty_label.setStyleSheet(f"color: {COLORS['text_muted']}; padding: 40px;")
        self.empty_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.empty_label)

        layout.addWidget(left_panel)

        # Right panel - Goal details
        right_panel = QFrame()
        right_panel.setStyleSheet(f"background-color: {COLORS['bg_secondary']}; border-left: 1px solid {COLORS['border']};")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(16)
        right_layout.setContentsMargins(24, 24, 24, 24)

        # Detail header
        self.detail_title = QLabel("Select a goal")
        self.detail_title.setStyleSheet("font-size: 18px; font-weight: 600;")
        right_layout.addWidget(self.detail_title)

        self.detail_desc = QLabel("")
        self.detail_desc.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self.detail_desc.setWordWrap(True)
        right_layout.addWidget(self.detail_desc)

        # Progress
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("0%")
        self.progress_label.setStyleSheet(f"color: {COLORS['accent_primary']}; font-weight: 600;")
        progress_layout.addWidget(self.progress_label)
        right_layout.addLayout(progress_layout)

        # Milestones
        milestones_header = QHBoxLayout()
        milestones_title = QLabel("Milestones")
        milestones_title.setStyleSheet("font-size: 14px; font-weight: 600;")
        milestones_header.addWidget(milestones_title)
        milestones_header.addStretch()

        self.add_milestone_btn = QPushButton("+ Add")
        self.add_milestone_btn.setObjectName("secondary")
        self.add_milestone_btn.clicked.connect(self.show_add_milestone_dialog)
        self.add_milestone_btn.hide()
        milestones_header.addWidget(self.add_milestone_btn)
        right_layout.addLayout(milestones_header)

        self.milestones_list = QListWidget()
        self.milestones_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {COLORS['bg_tertiary']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
            QListWidget::item {{
                padding: 12px;
                border-bottom: 1px solid {COLORS['border']};
            }}
            QListWidget::item:hover {{
                background-color: {COLORS['bg_input']};
            }}
        """)
        self.milestones_list.itemDoubleClicked.connect(self.on_milestone_double_clicked)
        right_layout.addWidget(self.milestones_list)

        right_layout.addStretch()
        layout.addWidget(right_panel)

        self._all_goals = []

    def update_goals(self, goals):
        """Update the goals list."""
        self._all_goals = goals
        self.filter_and_display()

    def filter_and_display(self):
        """Apply filters and display goals."""
        goals = self._all_goals

        status = self.status_filter.currentText().lower()
        if status == "active":
            goals = [g for g in goals if g.status.value == "active"]
        elif status == "completed":
            goals = [g for g in goals if g.status.value == "completed"]

        self.display_goals(goals)

    def display_goals(self, goals):
        """Display goals in the list."""
        # Clear existing
        while self.goals_layout.count():
            item = self.goals_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not goals:
            self.empty_label.show()
            return

        self.empty_label.hide()

        for goal in goals:
            card = GoalCard(goal)
            card.clicked.connect(self.select_goal)
            self.goals_layout.addWidget(card)

    def select_goal(self, goal_id):
        """Select a goal to show details."""
        goal = next((g for g in self._all_goals if g.id == goal_id), None)
        if not goal:
            return

        self._selected_goal = goal

        self.detail_title.setText(goal.title)
        self.detail_desc.setText(goal.description or "No description")

        progress = int(goal.progress * 100)
        self.progress_bar.setValue(progress)
        self.progress_label.setText(f"{progress}%")

        self.add_milestone_btn.show()

        # Update milestones list
        self.milestones_list.clear()
        for milestone in goal.milestones:
            status_icon = "" if milestone.status.value == "completed" else ""
            item = QListWidgetItem(f"{status_icon} {milestone.title}")
            item.setData(Qt.UserRole, milestone.id)
            self.milestones_list.addItem(item)

        self.goal_selected.emit(goal_id)

    def on_filter_changed(self):
        self.filter_and_display()

    def show_create_dialog(self):
        """Show the create goal dialog."""
        dialog = GoalDialog(parent=self)
        if dialog.exec():
            data = dialog.get_data()
            if data["title"]:
                self.create_goal.emit(data)
            else:
                QMessageBox.warning(self, "Error", "Goal title is required")

    def show_add_milestone_dialog(self):
        """Show dialog to add a milestone."""
        if not self._selected_goal:
            return

        from PySide6.QtWidgets import QInputDialog
        title, ok = QInputDialog.getText(self, "Add Milestone", "Milestone title:")
        if ok and title:
            self.add_milestone.emit(self._selected_goal.id, title, "")

    def on_milestone_double_clicked(self, item):
        """Handle milestone double-click to complete."""
        milestone_id = item.data(Qt.UserRole)
        reply = QMessageBox.question(
            self, "Complete Milestone",
            "Mark this milestone as completed?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.complete_milestone.emit(milestone_id)
