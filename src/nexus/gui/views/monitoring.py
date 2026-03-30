"""Monitoring View - Drift detection, performance metrics, cost tracking, and system health."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QGridLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QProgressBar, QGroupBox,
    QTabWidget, QSizePolicy, QComboBox
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QColor
from nexus.gui.styles import COLORS
from datetime import datetime


class MetricCard(QFrame):
    """Card displaying a single metric with trend."""

    def __init__(self, title: str, value: str, unit: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("metricCard")
        self.setStyleSheet(f"""
            QFrame#metricCard {{
                background-color: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 10px;
                padding: 16px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(16, 16, 16, 16)

        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px; text-transform: uppercase;")
        layout.addWidget(title_label)

        # Value
        value_layout = QHBoxLayout()
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 28px; font-weight: bold;")
        value_layout.addWidget(self.value_label)

        if unit:
            unit_label = QLabel(unit)
            unit_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 14px;")
            value_layout.addWidget(unit_label)

        value_layout.addStretch()
        layout.addLayout(value_layout)

        # Trend indicator
        self.trend_label = QLabel("")
        self.trend_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px;")
        layout.addWidget(self.trend_label)

    def update_value(self, value: str, trend: str = "", trend_positive: bool = True):
        """Update the metric value and trend."""
        self.value_label.setText(value)
        if trend:
            color = COLORS['accent_success'] if trend_positive else COLORS['accent_error']
            arrow = "↑" if trend_positive else "↓"
            self.trend_label.setText(f"{arrow} {trend}")
            self.trend_label.setStyleSheet(f"color: {color}; font-size: 12px;")


class AlertItem(QFrame):
    """A single alert/notification item."""

    dismissed = Signal(str)

    def __init__(self, alert_id: str, severity: str, message: str,
                 model: str, timestamp: str, parent=None):
        super().__init__(parent)
        self.alert_id = alert_id

        severity_colors = {
            "low": COLORS['accent_info'],
            "medium": COLORS['accent_warning'],
            "high": COLORS['accent_warning'],
            "critical": COLORS['accent_error'],
        }
        border_color = severity_colors.get(severity, COLORS['border'])

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_secondary']};
                border: 1px solid {border_color};
                border-left: 4px solid {border_color};
                border-radius: 6px;
                padding: 8px;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        # Content
        content_layout = QVBoxLayout()

        header = QHBoxLayout()
        severity_label = QLabel(severity.upper())
        severity_label.setStyleSheet(f"color: {border_color}; font-weight: bold; font-size: 11px;")
        header.addWidget(severity_label)

        model_label = QLabel(f"• {model}")
        model_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        header.addWidget(model_label)

        header.addStretch()

        time_label = QLabel(timestamp)
        time_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        header.addWidget(time_label)

        content_layout.addLayout(header)

        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 13px;")
        content_layout.addWidget(message_label)

        layout.addLayout(content_layout, 1)

        # Dismiss button
        dismiss_btn = QPushButton("✕")
        dismiss_btn.setFixedSize(24, 24)
        dismiss_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_muted']};
                border: none;
                font-size: 14px;
            }}
            QPushButton:hover {{
                color: {COLORS['text_primary']};
            }}
        """)
        dismiss_btn.clicked.connect(lambda: self.dismissed.emit(self.alert_id))
        layout.addWidget(dismiss_btn)


class MonitoringView(QWidget):
    """View for system monitoring, drift detection, and performance metrics."""

    refresh_requested = Signal()
    alert_dismissed = Signal(str)
    time_range_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.alert_items = []
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(24, 24, 24, 24)

        # Header
        header = QHBoxLayout()
        title = QLabel("System Monitoring")
        title.setObjectName("title")
        header.addWidget(title)

        header.addStretch()

        # Time range selector
        header.addWidget(QLabel("Time Range:"))
        self.time_range = QComboBox()
        self.time_range.addItems(["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days"])
        self.time_range.currentTextChanged.connect(self.time_range_changed.emit)
        header.addWidget(self.time_range)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setObjectName("secondary")
        refresh_btn.clicked.connect(self.refresh_requested.emit)
        header.addWidget(refresh_btn)

        layout.addLayout(header)

        # Metrics cards row
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(16)

        self.requests_card = MetricCard("Total Requests", "0")
        metrics_layout.addWidget(self.requests_card)

        self.latency_card = MetricCard("Avg Latency", "0", "ms")
        metrics_layout.addWidget(self.latency_card)

        self.cost_card = MetricCard("Total Cost", "$0.00")
        metrics_layout.addWidget(self.cost_card)

        self.health_card = MetricCard("System Health", "100%")
        metrics_layout.addWidget(self.health_card)

        layout.addLayout(metrics_layout)

        # Tabs for different monitoring views
        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                background-color: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
            QTabBar::tab {{
                background-color: {COLORS['bg_tertiary']};
                color: {COLORS['text_secondary']};
                padding: 10px 20px;
                margin-right: 4px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }}
            QTabBar::tab:selected {{
                background-color: {COLORS['bg_secondary']};
                color: {COLORS['text_primary']};
            }}
        """)

        # Drift Alerts Tab
        alerts_tab = QWidget()
        alerts_layout = QVBoxLayout(alerts_tab)
        alerts_layout.setContentsMargins(16, 16, 16, 16)

        alerts_header = QHBoxLayout()
        alerts_title = QLabel("Drift Alerts")
        alerts_title.setStyleSheet(f"font-weight: bold; color: {COLORS['text_primary']};")
        alerts_header.addWidget(alerts_title)
        alerts_header.addStretch()
        self.alerts_count = QLabel("0 active alerts")
        self.alerts_count.setStyleSheet(f"color: {COLORS['text_secondary']};")
        alerts_header.addWidget(self.alerts_count)
        alerts_layout.addLayout(alerts_header)

        self.alerts_scroll = QScrollArea()
        self.alerts_scroll.setWidgetResizable(True)
        self.alerts_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.alerts_container = QWidget()
        self.alerts_list_layout = QVBoxLayout(self.alerts_container)
        self.alerts_list_layout.setSpacing(8)
        self.alerts_list_layout.setContentsMargins(0, 0, 0, 0)
        self.alerts_list_layout.addStretch()

        self.alerts_scroll.setWidget(self.alerts_container)
        alerts_layout.addWidget(self.alerts_scroll)

        tabs.addTab(alerts_tab, "Drift Alerts")

        # Performance Tab
        perf_tab = QWidget()
        perf_layout = QVBoxLayout(perf_tab)
        perf_layout.setContentsMargins(16, 16, 16, 16)

        self.perf_table = QTableWidget()
        self.perf_table.setColumnCount(6)
        self.perf_table.setHorizontalHeaderLabels([
            "Model", "Requests", "Avg Latency", "Success Rate", "Errors", "Cost"
        ])
        self.perf_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.perf_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['bg_secondary']};
                border: none;
                gridline-color: {COLORS['border']};
            }}
            QTableWidget::item {{
                padding: 8px;
            }}
            QHeaderView::section {{
                background-color: {COLORS['bg_tertiary']};
                color: {COLORS['text_secondary']};
                padding: 10px;
                border: none;
                font-weight: bold;
            }}
        """)
        perf_layout.addWidget(self.perf_table)

        tabs.addTab(perf_tab, "Performance")

        # Cost Breakdown Tab
        cost_tab = QWidget()
        cost_layout = QVBoxLayout(cost_tab)
        cost_layout.setContentsMargins(16, 16, 16, 16)

        self.cost_table = QTableWidget()
        self.cost_table.setColumnCount(5)
        self.cost_table.setHorizontalHeaderLabels([
            "Model", "Input Tokens", "Output Tokens", "Requests", "Total Cost"
        ])
        self.cost_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cost_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['bg_secondary']};
                border: none;
                gridline-color: {COLORS['border']};
            }}
        """)
        cost_layout.addWidget(self.cost_table)

        # Cost summary
        cost_summary = QHBoxLayout()
        self.budget_label = QLabel("Budget: $0.00 / $100.00")
        self.budget_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        cost_summary.addWidget(self.budget_label)

        cost_summary.addStretch()

        self.budget_bar = QProgressBar()
        self.budget_bar.setRange(0, 100)
        self.budget_bar.setValue(0)
        self.budget_bar.setFixedWidth(200)
        self.budget_bar.setFixedHeight(8)
        self.budget_bar.setTextVisible(False)
        self.budget_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {COLORS['bg_input']};
                border: none;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['accent_success']};
                border-radius: 4px;
            }}
        """)
        cost_summary.addWidget(self.budget_bar)

        cost_layout.addLayout(cost_summary)

        tabs.addTab(cost_tab, "Cost Breakdown")

        # Circuit Breakers Tab
        circuit_tab = QWidget()
        circuit_layout = QVBoxLayout(circuit_tab)
        circuit_layout.setContentsMargins(16, 16, 16, 16)

        self.circuit_table = QTableWidget()
        self.circuit_table.setColumnCount(5)
        self.circuit_table.setHorizontalHeaderLabels([
            "Model", "State", "Failures", "Last Failure", "Recovery In"
        ])
        self.circuit_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.circuit_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['bg_secondary']};
                border: none;
                gridline-color: {COLORS['border']};
            }}
        """)
        circuit_layout.addWidget(self.circuit_table)

        tabs.addTab(circuit_tab, "Circuit Breakers")

        layout.addWidget(tabs, 1)

    def update_metrics(self, total_requests: int, avg_latency: float,
                       total_cost: float, system_health: float,
                       request_trend: str = "", latency_trend: str = "",
                       cost_trend: str = ""):
        """Update the metric cards."""
        self.requests_card.update_value(f"{total_requests:,}", request_trend, True)
        self.latency_card.update_value(f"{avg_latency:.0f}", latency_trend, avg_latency < 1000)
        self.cost_card.update_value(f"${total_cost:.2f}", cost_trend, False)
        self.health_card.update_value(f"{system_health:.0f}%", "", system_health >= 80)

    def update_alerts(self, alerts: list):
        """Update the drift alerts list."""
        # Clear existing alerts
        for item in self.alert_items:
            item.deleteLater()
        self.alert_items.clear()

        # Add new alerts
        for alert in alerts:
            item = AlertItem(
                alert_id=alert.get('id', ''),
                severity=alert.get('severity', 'low'),
                message=alert.get('message', ''),
                model=alert.get('model', ''),
                timestamp=alert.get('timestamp', '')
            )
            item.dismissed.connect(self.alert_dismissed.emit)
            self.alerts_list_layout.insertWidget(
                self.alerts_list_layout.count() - 1, item
            )
            self.alert_items.append(item)

        self.alerts_count.setText(f"{len(alerts)} active alerts")

    def update_performance(self, models: list):
        """Update the performance table."""
        self.perf_table.setRowCount(len(models))

        for i, model in enumerate(models):
            self.perf_table.setItem(i, 0, QTableWidgetItem(model.get('name', '')))
            self.perf_table.setItem(i, 1, QTableWidgetItem(str(model.get('requests', 0))))
            self.perf_table.setItem(i, 2, QTableWidgetItem(f"{model.get('latency', 0):.0f} ms"))
            self.perf_table.setItem(i, 3, QTableWidgetItem(f"{model.get('success_rate', 0):.1f}%"))
            self.perf_table.setItem(i, 4, QTableWidgetItem(str(model.get('errors', 0))))
            self.perf_table.setItem(i, 5, QTableWidgetItem(f"${model.get('cost', 0):.4f}"))

    def update_cost_breakdown(self, models: list, budget_used: float, budget_limit: float):
        """Update the cost breakdown table and budget bar."""
        self.cost_table.setRowCount(len(models))

        for i, model in enumerate(models):
            self.cost_table.setItem(i, 0, QTableWidgetItem(model.get('name', '')))
            self.cost_table.setItem(i, 1, QTableWidgetItem(f"{model.get('input_tokens', 0):,}"))
            self.cost_table.setItem(i, 2, QTableWidgetItem(f"{model.get('output_tokens', 0):,}"))
            self.cost_table.setItem(i, 3, QTableWidgetItem(str(model.get('requests', 0))))
            self.cost_table.setItem(i, 4, QTableWidgetItem(f"${model.get('cost', 0):.4f}"))

        # Update budget bar
        self.budget_label.setText(f"Budget: ${budget_used:.2f} / ${budget_limit:.2f}")
        percentage = (budget_used / budget_limit * 100) if budget_limit > 0 else 0
        self.budget_bar.setValue(int(percentage))

        # Color based on usage
        if percentage >= 90:
            color = COLORS['accent_error']
        elif percentage >= 70:
            color = COLORS['accent_warning']
        else:
            color = COLORS['accent_success']

        self.budget_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {COLORS['bg_input']};
                border: none;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 4px;
            }}
        """)

    def update_circuit_breakers(self, breakers: list):
        """Update the circuit breakers table."""
        self.circuit_table.setRowCount(len(breakers))

        state_colors = {
            "closed": COLORS['accent_success'],
            "open": COLORS['accent_error'],
            "half_open": COLORS['accent_warning'],
        }

        for i, breaker in enumerate(breakers):
            self.circuit_table.setItem(i, 0, QTableWidgetItem(breaker.get('model', '')))

            state = breaker.get('state', 'closed')
            state_item = QTableWidgetItem(state.upper())
            state_item.setForeground(QColor(state_colors.get(state, COLORS['text_primary'])))
            self.circuit_table.setItem(i, 1, state_item)

            self.circuit_table.setItem(i, 2, QTableWidgetItem(str(breaker.get('failures', 0))))
            self.circuit_table.setItem(i, 3, QTableWidgetItem(breaker.get('last_failure', 'N/A')))
            self.circuit_table.setItem(i, 4, QTableWidgetItem(breaker.get('recovery_in', 'N/A')))

    def update_data(self, data: dict):
        """Update all view data from a single data dictionary.

        Args:
            data: Dictionary containing 'alerts', 'metrics', 'cost', 'circuit_breakers' keys
        """
        # Update alerts
        alerts = data.get("alerts", [])
        self.update_alerts(alerts)

        # Update metrics
        metrics = data.get("metrics", {})
        self.update_metrics(
            total_requests=metrics.get("total_requests", 0),
            avg_latency=metrics.get("avg_latency", 0) * 1000,  # Convert to ms
            total_cost=data.get("cost", {}).get("today", 0),
            system_health=metrics.get("success_rate", 100),
        )

        # Update cost breakdown
        cost_data = data.get("cost", {})
        by_provider = cost_data.get("by_provider", {})
        cost_models = [
            {"name": provider, "cost": cost, "requests": 0, "input_tokens": 0, "output_tokens": 0}
            for provider, cost in by_provider.items()
        ]
        budget_used = cost_data.get("today", 0)
        budget_limit = cost_data.get("budget_remaining", 500) + budget_used
        self.update_cost_breakdown(cost_models, budget_used, budget_limit)

        # Update circuit breakers
        circuit_breakers = data.get("circuit_breakers", [])
        breakers = [
            {
                "model": cb.get("model", ""),
                "state": cb.get("status", "closed"),
                "failures": cb.get("failures", 0),
                "last_failure": cb.get("last_failure", "N/A"),
                "recovery_in": "Auto",
            }
            for cb in circuit_breakers
        ]
        self.update_circuit_breakers(breakers)
