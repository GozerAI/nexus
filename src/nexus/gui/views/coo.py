"""
COO View - Chief Operating Officer Dashboard

Provides visibility and control over the autonomous COO:
- Status and state monitoring
- Mode switching (autonomous, supervised, approval, observe)
- Pending approvals queue
- Execution history
- Learning statistics
- Manual task execution triggers
"""

import logging

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QGridLayout, QComboBox, QTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
    QGroupBox, QSplitter, QTabWidget
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

from nexus.gui.styles import COLORS


logger = logging.getLogger(__name__)


class COOView(QWidget):
    """Dashboard for the Autonomous COO."""

    def __init__(self, controller, async_bridge):
        super().__init__()
        self.controller = controller
        self.async_bridge = async_bridge
        self._current_suggestion_id = None
        self._is_visible = False

        self._setup_ui()
        self._setup_refresh_timer()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header with title and controls
        header = self._create_header()
        layout.addWidget(header)

        # Main content with tabs
        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {COLORS['border']};
                background: {COLORS['bg_secondary']};
                border-radius: 8px;
            }}
            QTabBar::tab {{
                background: {COLORS['bg_primary']};
                color: {COLORS['text_secondary']};
                padding: 10px 20px;
                border: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }}
            QTabBar::tab:selected {{
                background: {COLORS['bg_secondary']};
                color: {COLORS['text_primary']};
            }}
        """)

        # Status Tab
        status_tab = self._create_status_tab()
        tabs.addTab(status_tab, "Status")

        # Approvals Tab
        approvals_tab = self._create_approvals_tab()
        tabs.addTab(approvals_tab, "Pending Approvals")

        # Execution History Tab
        history_tab = self._create_history_tab()
        tabs.addTab(history_tab, "Execution History")

        # Learning Tab
        learning_tab = self._create_learning_tab()
        tabs.addTab(learning_tab, "Learning & Analytics")

        layout.addWidget(tabs)

    def _create_header(self) -> QFrame:
        """Create header with controls."""
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['bg_secondary']};
                border-radius: 12px;
                padding: 16px;
            }}
        """)

        layout = QHBoxLayout(frame)

        # Title and status
        title_layout = QVBoxLayout()

        title = QLabel("Autonomous COO")
        title.setFont(QFont("", 18, QFont.Bold))
        title.setStyleSheet(f"color: {COLORS['text_primary']};")
        title_layout.addWidget(title)

        self.status_label = QLabel("Status: Initializing...")
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        title_layout.addWidget(self.status_label)

        layout.addLayout(title_layout)
        layout.addStretch()

        # Mode selector
        mode_layout = QVBoxLayout()
        mode_label = QLabel("Operating Mode:")
        mode_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        mode_layout.addWidget(mode_label)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["autonomous", "supervised", "approval", "observe", "paused"])
        self.mode_combo.setStyleSheet(f"""
            QComboBox {{
                background: {COLORS['bg_primary']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 8px 12px;
                min-width: 150px;
            }}
        """)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)

        layout.addLayout(mode_layout)

        # Start/Stop buttons
        self.start_btn = QPushButton("Start COO")
        self.start_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['accent_success']};
                color: {COLORS['bg_primary']};
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {COLORS['success']};
            }}
        """)
        self.start_btn.clicked.connect(self._on_start_clicked)
        layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop COO")
        self.stop_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['accent_error']};
                color: {COLORS['text_primary']};
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {COLORS['error']};
            }}
        """)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)

        return frame

    def _create_status_tab(self) -> QWidget:
        """Create status overview tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)

        # Metrics grid
        metrics_frame = QFrame()
        metrics_frame.setStyleSheet(f"background: {COLORS['bg_primary']}; border-radius: 8px; padding: 16px;")
        metrics_layout = QGridLayout(metrics_frame)

        # Define metrics
        self.metric_labels = {}
        metrics = [
            ("state", "Current State", "idle"),
            ("uptime", "Uptime", "0s"),
            ("total_executed", "Total Executed", "0"),
            ("success_rate", "Success Rate", "0%"),
            ("pending_approvals", "Pending Approvals", "0"),
            ("current_executions", "Active Executions", "0"),
            ("daily_spend", "Daily Spend", "$0.00"),
            ("learning_effectiveness", "Learning Score", "0%"),
        ]

        for i, (key, label, default) in enumerate(metrics):
            row, col = divmod(i, 4)

            label_widget = QLabel(label)
            label_widget.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
            metrics_layout.addWidget(label_widget, row * 2, col)

            value_widget = QLabel(default)
            value_widget.setFont(QFont("", 16, QFont.Bold))
            value_widget.setStyleSheet(f"color: {COLORS['text_primary']};")
            metrics_layout.addWidget(value_widget, row * 2 + 1, col)

            self.metric_labels[key] = value_widget

        layout.addWidget(metrics_frame)

        # Current suggestion section
        suggestion_group = QGroupBox("Next Suggested Action")
        suggestion_group.setStyleSheet(f"""
            QGroupBox {{
                color: {COLORS['text_primary']};
                font-weight: bold;
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 16px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px;
            }}
        """)
        suggestion_layout = QVBoxLayout(suggestion_group)

        self.suggestion_text = QTextEdit()
        self.suggestion_text.setReadOnly(True)
        self.suggestion_text.setMaximumHeight(150)
        self.suggestion_text.setStyleSheet(f"""
            QTextEdit {{
                background: {COLORS['bg_primary']};
                color: {COLORS['text_primary']};
                border: none;
                border-radius: 6px;
                padding: 12px;
            }}
        """)
        self.suggestion_text.setPlainText("Loading suggestion...")
        suggestion_layout.addWidget(self.suggestion_text)

        # Action buttons
        action_layout = QHBoxLayout()
        self.execute_suggestion_btn = QPushButton("Execute Now")
        self.execute_suggestion_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['accent_primary']};
                color: {COLORS['bg_primary']};
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
            }}
        """)
        self.execute_suggestion_btn.clicked.connect(self._on_execute_suggestion)
        action_layout.addWidget(self.execute_suggestion_btn)

        refresh_btn = QPushButton("Refresh Suggestion")
        refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['bg_secondary']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 8px 16px;
            }}
        """)
        refresh_btn.clicked.connect(self._refresh_suggestion)
        action_layout.addWidget(refresh_btn)

        action_layout.addStretch()
        suggestion_layout.addLayout(action_layout)

        layout.addWidget(suggestion_group)
        layout.addStretch()

        return widget

    def _create_approvals_tab(self) -> QWidget:
        """Create pending approvals tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Approvals table
        self.approvals_table = QTableWidget()
        self.approvals_table.setColumnCount(5)
        self.approvals_table.setHorizontalHeaderLabels([
            "Item", "Type", "Executor", "Confidence", "Actions"
        ])
        self.approvals_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.approvals_table.setStyleSheet(f"""
            QTableWidget {{
                background: {COLORS['bg_secondary']};
                color: {COLORS['text_primary']};
                border: none;
                border-radius: 8px;
            }}
            QHeaderView::section {{
                background: {COLORS['bg_primary']};
                color: {COLORS['text_secondary']};
                padding: 12px;
                border: none;
            }}
        """)

        layout.addWidget(self.approvals_table)

        return widget

    def _create_history_tab(self) -> QWidget:
        """Create execution history tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # History table
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels([
            "Item", "Executor", "Status", "Duration", "Cost", "Time"
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.history_table.setStyleSheet(f"""
            QTableWidget {{
                background: {COLORS['bg_secondary']};
                color: {COLORS['text_primary']};
                border: none;
                border-radius: 8px;
            }}
            QHeaderView::section {{
                background: {COLORS['bg_primary']};
                color: {COLORS['text_secondary']};
                padding: 12px;
                border: none;
            }}
        """)

        layout.addWidget(self.history_table)

        # Refresh button
        refresh_btn = QPushButton("Refresh History")
        refresh_btn.clicked.connect(self._refresh_history)
        layout.addWidget(refresh_btn)

        return widget

    def _create_learning_tab(self) -> QWidget:
        """Create learning analytics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Learning stats
        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"background: {COLORS['bg_primary']}; border-radius: 8px; padding: 16px;")
        stats_layout = QVBoxLayout(stats_frame)

        stats_title = QLabel("Learning Statistics")
        stats_title.setFont(QFont("", 14, QFont.Bold))
        stats_title.setStyleSheet(f"color: {COLORS['text_primary']};")
        stats_layout.addWidget(stats_title)

        self.learning_stats_text = QTextEdit()
        self.learning_stats_text.setReadOnly(True)
        self.learning_stats_text.setStyleSheet(f"""
            QTextEdit {{
                background: {COLORS['bg_secondary']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 12px;
            }}
        """)
        self.learning_stats_text.setPlainText("Loading learning statistics...")
        stats_layout.addWidget(self.learning_stats_text)

        layout.addWidget(stats_frame)

        # Improvement trend
        trend_frame = QFrame()
        trend_frame.setStyleSheet(f"background: {COLORS['bg_primary']}; border-radius: 8px; padding: 16px;")
        trend_layout = QVBoxLayout(trend_frame)

        trend_title = QLabel("Performance Trend")
        trend_title.setFont(QFont("", 14, QFont.Bold))
        trend_title.setStyleSheet(f"color: {COLORS['text_primary']};")
        trend_layout.addWidget(trend_title)

        self.trend_progress = QProgressBar()
        self.trend_progress.setRange(0, 100)
        self.trend_progress.setValue(50)
        self.trend_progress.setStyleSheet(f"""
            QProgressBar {{
                background: {COLORS['bg_secondary']};
                border-radius: 6px;
                height: 20px;
            }}
            QProgressBar::chunk {{
                background: {COLORS['accent_primary']};
                border-radius: 6px;
            }}
        """)
        trend_layout.addWidget(self.trend_progress)

        self.trend_label = QLabel("Baseline performance")
        self.trend_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        trend_layout.addWidget(self.trend_label)

        layout.addWidget(trend_frame)
        layout.addStretch()

        return widget

    def _setup_refresh_timer(self):
        """Setup auto-refresh timer."""
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh_data)
        self.refresh_timer.start(15000)  # Refresh every 15 seconds (reduced frequency)
        self._refresh_pending = False  # Debounce flag

    def _refresh_data(self):
        """Refresh all data from COO."""
        # Only refresh if visible and controller is initialized
        if not self._is_visible:
            return

        if not self.controller.is_initialized:
            # Controller not ready - update UI to show loading state
            self.status_label.setText("Status: Initializing backend...")
            return

        # Debounce: Don't start new refresh if one is pending
        if self._refresh_pending:
            return
        self._refresh_pending = True

        self.async_bridge.run_async(
            self.controller.get_coo_status(),
            on_success=self._on_status_received,
            on_error=self._on_refresh_error
        )

    def _on_status_received(self, status):
        """Handle status received."""
        self._refresh_pending = False
        self._update_status(status)

    def _on_refresh_error(self, error):
        """Handle refresh error."""
        self._refresh_pending = False
        logger.warning("Error refreshing COO status: %s", error)

    def _update_status(self, status: dict):
        """Update UI with COO status."""
        # Update status label
        state = status.get("state", "unknown")
        mode = status.get("mode", "unknown")

        # Show more readable status messages
        state_display = {
            "idle": "Idle (Ready to Start)",
            "observing": "Observing",
            "prioritizing": "Prioritizing Tasks",
            "delegating": "Delegating Work",
            "executing": "Executing",
            "waiting_approval": "Waiting for Approval",
            "learning": "Learning from Outcomes",
            "not_initialized": "Initializing...",
            "not_available": "Not Available",
        }.get(state, state.upper())

        self.status_label.setText(f"Status: {state_display} | Mode: {mode}")

        # Update metrics
        if "state" in self.metric_labels:
            self.metric_labels["state"].setText(state.capitalize())

        if "uptime" in self.metric_labels:
            uptime = status.get("uptime_seconds", 0)
            if uptime > 3600:
                uptime_str = f"{uptime/3600:.1f}h"
            elif uptime > 60:
                uptime_str = f"{uptime/60:.1f}m"
            else:
                uptime_str = f"{uptime:.0f}s"
            self.metric_labels["uptime"].setText(uptime_str)

        if "total_executed" in self.metric_labels:
            self.metric_labels["total_executed"].setText(str(status.get("total_tasks_executed", 0)))

        if "success_rate" in self.metric_labels:
            total = status.get("total_tasks_executed", 0)
            successful = status.get("successful_executions", 0)
            rate = (successful / total * 100) if total > 0 else 0
            self.metric_labels["success_rate"].setText(f"{rate:.0f}%")

        if "pending_approvals" in self.metric_labels:
            self.metric_labels["pending_approvals"].setText(str(status.get("pending_approvals", 0)))

        if "current_executions" in self.metric_labels:
            current = status.get("current_executions", [])
            self.metric_labels["current_executions"].setText(str(len(current)))

        if "daily_spend" in self.metric_labels:
            spend = status.get("daily_spend_usd", 0)
            self.metric_labels["daily_spend"].setText(f"${spend:.2f}")

        if "learning_effectiveness" in self.metric_labels:
            effectiveness = status.get("learning_effectiveness", 0)
            self.metric_labels["learning_effectiveness"].setText(f"{effectiveness*100:.0f}%")

        # Update button states
        is_running = state not in ["idle", "not_initialized", "not_available"]
        self.start_btn.setEnabled(not is_running)
        self.stop_btn.setEnabled(is_running)

        # Update mode combo without triggering change event
        self.mode_combo.blockSignals(True)
        idx = self.mode_combo.findText(mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)
        self.mode_combo.blockSignals(False)

    def _on_mode_changed(self, mode: str):
        """Handle mode change."""
        self.async_bridge.run_async(
            self.controller.set_coo_mode(mode),
            on_success=lambda _: self._refresh_data(),
            on_error=lambda e: logger.warning("Error setting mode: %s", e)
        )

    def _on_start_clicked(self):
        """Handle start button click."""
        from PySide6.QtWidgets import QMessageBox

        # Check if controller is initialized first
        if not self.controller._initialized:
            QMessageBox.warning(
                self,
                "COO Not Ready",
                "The intelligence layer is still initializing.\n\n"
                "Please wait a few seconds and try again.\n\n"
                "If this persists, check the console for errors."
            )
            return

        # Check if COO exists
        if not self.controller._intel:
            QMessageBox.critical(
                self,
                "COO Error",
                "Intelligence layer not available.\n\n"
                "The backend may have failed to initialize."
            )
            return

        coo = self.controller._intel.get_coo()
        if not coo:
            QMessageBox.critical(
                self,
                "COO Error",
                "COO component not initialized.\n\n"
                "Check the console for initialization errors."
            )
            return

        # Show immediate feedback
        self.start_btn.setEnabled(False)
        self.start_btn.setText("Starting...")
        self.status_label.setText("Status: Starting COO...")

        logger.info("[COO] Starting COO - controller initialized: %s", self.controller._initialized)
        logger.info("[COO] Intel: %s", self.controller._intel)
        logger.info("[COO] COO instance: %s", coo)

        self.async_bridge.run_async(
            self.controller.start_coo(),
            on_success=lambda r: self._on_start_success(r),
            on_error=lambda e: self._on_start_error(e)
        )

    def _on_start_success(self, result):
        """Handle successful COO start."""
        from PySide6.QtWidgets import QMessageBox

        logger.info("[COO] Start result: %s", result)

        # Reset button
        self.start_btn.setText("Start COO")

        if result:
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_label.setText("Status: Running")
            logger.info("[COO] COO started successfully!")
        else:
            self.start_btn.setEnabled(True)
            self.status_label.setText("Status: Failed to start")
            QMessageBox.warning(
                self,
                "COO Start Failed",
                "The COO returned False when starting.\n\n"
                "This usually means the COO components failed to initialize.\n"
                "Check the console for detailed error messages."
            )

        self._refresh_data()

    def _on_start_error(self, error):
        """Handle COO start error."""
        from PySide6.QtWidgets import QMessageBox

        logger.warning("[COO] Start error: %s", error)

        # Reset button
        self.start_btn.setText("Start COO")
        self.start_btn.setEnabled(True)
        self.status_label.setText(f"Status: Error")

        QMessageBox.critical(
            self,
            "COO Error",
            f"Failed to start COO:\n\n{error}\n\n"
            "Check the console for more details."
        )

    def _on_stop_clicked(self):
        """Handle stop button click."""
        self.async_bridge.run_async(
            self.controller.stop_coo(),
            on_success=lambda _: self._refresh_data(),
            on_error=lambda e: logger.warning("Error stopping COO: %s", e)
        )

    def _refresh_suggestion(self):
        """Refresh next action suggestion."""
        if not self.controller.is_initialized:
            self.suggestion_text.setPlainText("Waiting for backend initialization...")
            self.execute_suggestion_btn.setEnabled(False)
            return

        self.async_bridge.run_async(
            self.controller.get_coo_suggestions(),
            on_success=self._update_suggestion,
            on_error=lambda e: self.suggestion_text.setPlainText(f"Error: {e}")
        )

    def _update_suggestion(self, data: dict):
        """Update suggestion display."""
        suggestion = data.get("suggestion")
        reason = data.get("reason", "No suggestions available")
        context = data.get("context_summary", {})

        if not suggestion:
            # Show a more informative message when no suggestion
            if reason == "Not initialized":
                message = "COO is initializing... Please wait."
            elif reason == "COO not available":
                message = "COO system is not available. Check configuration."
            elif "No actionable items" in reason:
                goals = context.get("active_goals", 0)
                tasks = context.get("pending_tasks", 0)
                message = f"""No actionable items to suggest.

Current State:
- Active Goals: {goals}
- Pending Tasks: {tasks}
- Blockers: {context.get('blockers', 0)}

Create some tasks or goals to see suggestions here."""
            else:
                message = reason

            self.suggestion_text.setPlainText(message)
            self.execute_suggestion_btn.setEnabled(False)
            return

        decision = data.get("decision", {})

        # PrioritizedItem has an 'item' attribute containing the actual task/goal
        item = getattr(suggestion, 'item', suggestion)
        title = getattr(item, 'title', str(item)[:100])
        score = getattr(suggestion, 'score', None)
        score_value = score.total_score if score else 0

        text = f"""Suggested Action: {title}

Priority Score: {score_value:.2f}
Executor: {decision.get('executor', 'unknown')}
Confidence: {decision.get('confidence', 0):.0%}
Reason: {decision.get('reason', 'N/A')}

Context:
- Active Goals: {context.get('active_goals', 0)}
- Pending Tasks: {context.get('pending_tasks', 0)}
- Blockers: {context.get('blockers', 0)}
"""
        self.suggestion_text.setPlainText(text)
        self.execute_suggestion_btn.setEnabled(True)
        self._current_suggestion_id = getattr(suggestion, 'id', None)

    def _on_execute_suggestion(self):
        """Execute the current suggestion."""
        if hasattr(self, '_current_suggestion_id') and self._current_suggestion_id:
            self.async_bridge.run_async(
                self.controller.execute_task_now(self._current_suggestion_id),
                on_success=lambda _: self._refresh_data(),
                on_error=lambda e: logger.warning("Error executing: %s", e)
            )

    def _refresh_history(self):
        """Refresh execution history."""
        self.async_bridge.run_async(
            self.controller.get_coo_execution_history(50),
            on_success=self._update_history,
            on_error=lambda e: logger.warning("Error loading history: %s", e)
        )

    def _update_history(self, history: list):
        """Update history table."""
        self.history_table.setRowCount(len(history))

        for row, record in enumerate(history):
            self.history_table.setItem(row, 0, QTableWidgetItem(record.get("item_id", "")[:20]))
            self.history_table.setItem(row, 1, QTableWidgetItem(record.get("executor", "")))

            status = record.get("status", "unknown")
            status_item = QTableWidgetItem(status)
            if status == "completed":
                status_item.setForeground(Qt.green)
            elif status == "failed":
                status_item.setForeground(Qt.red)
            self.history_table.setItem(row, 2, status_item)

            duration = record.get("duration_minutes", 0)
            self.history_table.setItem(row, 3, QTableWidgetItem(f"{duration:.1f}m"))

            cost = record.get("cost_usd", 0)
            self.history_table.setItem(row, 4, QTableWidgetItem(f"${cost:.3f}"))

            completed = record.get("completed_at", "")
            if completed:
                time_str = completed.split("T")[1][:8] if "T" in completed else completed
            else:
                time_str = "-"
            self.history_table.setItem(row, 5, QTableWidgetItem(time_str))

    def showEvent(self, event):
        """Handle show event."""
        super().showEvent(event)
        self._is_visible = True
        self._refresh_data()
        self._refresh_suggestion()

    def hideEvent(self, event):
        """Handle hide event - stop refreshing when not visible."""
        super().hideEvent(event)
        self._is_visible = False
