"""Models View - Ensemble management, model health, and strategy selection."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QGridLayout, QPushButton, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
    QGroupBox, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QColor
from nexus.gui.styles import COLORS


class ModelCard(QFrame):
    """Card displaying a single model's status and metrics."""

    def __init__(self, model_name: str, provider: str, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.provider = provider
        self.setObjectName("card")
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        self.setStyleSheet(f"""
            QFrame#card {{
                background-color: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 12px;
            }}
            QFrame#card:hover {{
                border-color: {COLORS['accent_primary']};
            }}
        """)

        # Header with model name and status indicator
        header = QHBoxLayout()

        name_label = QLabel(self.model_name)
        name_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['text_primary']};")
        header.addWidget(name_label)

        header.addStretch()

        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet(f"color: {COLORS['accent_success']}; font-size: 18px;")
        header.addWidget(self.status_indicator)

        layout.addLayout(header)

        # Provider
        provider_label = QLabel(f"Provider: {self.provider}")
        provider_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        layout.addWidget(provider_label)

        # Metrics grid
        metrics_layout = QGridLayout()
        metrics_layout.setSpacing(8)

        # Latency
        metrics_layout.addWidget(QLabel("Latency:"), 0, 0)
        self.latency_label = QLabel("-- ms")
        self.latency_label.setStyleSheet(f"color: {COLORS['accent_info']};")
        metrics_layout.addWidget(self.latency_label, 0, 1)

        # Success Rate
        metrics_layout.addWidget(QLabel("Success:"), 1, 0)
        self.success_label = QLabel("--%")
        self.success_label.setStyleSheet(f"color: {COLORS['accent_success']};")
        metrics_layout.addWidget(self.success_label, 1, 1)

        # Requests
        metrics_layout.addWidget(QLabel("Requests:"), 2, 0)
        self.requests_label = QLabel("0")
        self.requests_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        metrics_layout.addWidget(self.requests_label, 2, 1)

        # Cost
        metrics_layout.addWidget(QLabel("Cost:"), 3, 0)
        self.cost_label = QLabel("$0.00")
        self.cost_label.setStyleSheet(f"color: {COLORS['accent_warning']};")
        metrics_layout.addWidget(self.cost_label, 3, 1)

        layout.addLayout(metrics_layout)

        # Health bar
        health_layout = QHBoxLayout()
        health_layout.addWidget(QLabel("Health:"))
        self.health_bar = QProgressBar()
        self.health_bar.setRange(0, 100)
        self.health_bar.setValue(100)
        self.health_bar.setTextVisible(False)
        self.health_bar.setFixedHeight(8)
        self.health_bar.setStyleSheet(f"""
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
        health_layout.addWidget(self.health_bar)
        layout.addLayout(health_layout)

    def update_metrics(self, latency_ms: float, success_rate: float,
                       requests: int, cost: float, health: float, status: str):
        """Update the card with new metrics."""
        self.latency_label.setText(f"{latency_ms:.0f} ms")
        self.success_label.setText(f"{success_rate:.1f}%")
        self.requests_label.setText(str(requests))
        self.cost_label.setText(f"${cost:.4f}")
        self.health_bar.setValue(int(health * 100))

        # Update status indicator
        status_colors = {
            "healthy": COLORS['accent_success'],
            "degraded": COLORS['accent_warning'],
            "unhealthy": COLORS['accent_error'],
            "quarantined": COLORS['accent_error'],
            "unknown": COLORS['text_muted'],
        }
        self.status_indicator.setStyleSheet(
            f"color: {status_colors.get(status, COLORS['text_muted'])}; font-size: 18px;"
        )

        # Update health bar color based on health
        if health >= 0.8:
            bar_color = COLORS['accent_success']
        elif health >= 0.5:
            bar_color = COLORS['accent_warning']
        else:
            bar_color = COLORS['accent_error']

        self.health_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {COLORS['bg_input']};
                border: none;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background-color: {bar_color};
                border-radius: 4px;
            }}
        """)


class ModelsView(QWidget):
    """View for managing and monitoring AI models in the ensemble."""

    refresh_requested = Signal()
    strategy_changed = Signal(str)
    model_quarantine_requested = Signal(str)
    model_release_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_cards = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(24, 24, 24, 24)

        # Header
        header = QHBoxLayout()
        title = QLabel("Models & Ensemble")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setObjectName("secondary")
        refresh_btn.clicked.connect(self.refresh_requested.emit)
        header.addWidget(refresh_btn)

        layout.addLayout(header)

        # Strategy selection
        strategy_frame = QFrame()
        strategy_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 12px;
            }}
        """)
        strategy_layout = QHBoxLayout(strategy_frame)

        strategy_layout.addWidget(QLabel("Ensemble Strategy:"))

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "Weighted Average",
            "Majority Voting",
            "Cascading",
            "Dynamic Selection",
            "Cost Optimized",
            "Synthesized"
        ])
        self.strategy_combo.currentTextChanged.connect(self.strategy_changed.emit)
        self.strategy_combo.setMinimumWidth(200)
        strategy_layout.addWidget(self.strategy_combo)

        strategy_layout.addStretch()

        # Ensemble stats
        self.ensemble_stats = QLabel("Active Models: 0 | Total Requests: 0 | Avg Latency: -- ms")
        self.ensemble_stats.setStyleSheet(f"color: {COLORS['text_secondary']};")
        strategy_layout.addWidget(self.ensemble_stats)

        layout.addWidget(strategy_frame)

        # Scrollable model cards area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: transparent;
                border: none;
            }}
        """)

        self.cards_container = QWidget()
        self.cards_layout = QGridLayout(self.cards_container)
        self.cards_layout.setSpacing(16)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)

        scroll.setWidget(self.cards_container)
        layout.addWidget(scroll, 1)

        # Quarantined models section
        quarantine_group = QGroupBox("Quarantined Models")
        quarantine_group.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {COLORS['accent_error']};
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }}
            QGroupBox::title {{
                color: {COLORS['accent_error']};
            }}
        """)
        quarantine_layout = QVBoxLayout(quarantine_group)

        self.quarantine_table = QTableWidget()
        self.quarantine_table.setColumnCount(3)
        self.quarantine_table.setHorizontalHeaderLabels(["Model", "Reason", "Actions"])
        self.quarantine_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.quarantine_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.quarantine_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.quarantine_table.setColumnWidth(2, 100)
        self.quarantine_table.setMaximumHeight(150)
        self.quarantine_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['bg_secondary']};
                border: none;
                gridline-color: {COLORS['border']};
            }}
            QTableWidget::item {{
                padding: 8px;
            }}
        """)

        quarantine_layout.addWidget(self.quarantine_table)
        layout.addWidget(quarantine_group)

    def update_models(self, models: list):
        """Update model cards with new data."""
        # Clear existing cards
        for card in self.model_cards.values():
            card.deleteLater()
        self.model_cards.clear()

        # Create new cards
        for i, model in enumerate(models):
            card = ModelCard(
                model_name=model.get('name', 'Unknown'),
                provider=model.get('provider', 'Unknown')
            )
            card.update_metrics(
                latency_ms=model.get('latency_ms', 0),
                success_rate=model.get('success_rate', 0),
                requests=model.get('requests', 0),
                cost=model.get('cost', 0),
                health=model.get('health', 1.0),
                status=model.get('status', 'unknown')
            )

            row = i // 3
            col = i % 3
            self.cards_layout.addWidget(card, row, col)
            self.model_cards[model.get('name')] = card

    def update_ensemble_stats(self, active_models: int, total_requests: int, avg_latency: float):
        """Update ensemble statistics display."""
        self.ensemble_stats.setText(
            f"Active Models: {active_models} | Total Requests: {total_requests} | Avg Latency: {avg_latency:.0f} ms"
        )

    def update_quarantined(self, quarantined: list):
        """Update quarantined models table."""
        self.quarantine_table.setRowCount(len(quarantined))

        for i, item in enumerate(quarantined):
            self.quarantine_table.setItem(i, 0, QTableWidgetItem(item.get('model', '')))
            self.quarantine_table.setItem(i, 1, QTableWidgetItem(item.get('reason', '')))

            release_btn = QPushButton("Release")
            release_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['accent_success']};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 4px 8px;
                }}
            """)
            model_name = item.get('model', '')
            release_btn.clicked.connect(lambda checked, m=model_name: self.model_release_requested.emit(m))
            self.quarantine_table.setCellWidget(i, 2, release_btn)

    def set_strategy(self, strategy: str):
        """Set the current strategy in the combo box."""
        index = self.strategy_combo.findText(strategy)
        if index >= 0:
            self.strategy_combo.setCurrentIndex(index)

    def update_data(self, data: dict):
        """Update all view data from a single data dictionary.

        Args:
            data: Dictionary containing 'models', 'strategy', 'health', 'quarantined' keys
        """
        # Update models
        models = data.get("models", [])
        self.update_models(models)

        # Update strategy
        strategy = data.get("strategy", "")
        strategy_map = {
            "weighted_quality": "Weighted Average",
            "lowest_latency": "Dynamic Selection",
            "lowest_cost": "Cost Optimized",
            "round_robin": "Majority Voting",
            "capability_match": "Cascading",
        }
        display_strategy = strategy_map.get(strategy, strategy)
        self.set_strategy(display_strategy)

        # Update ensemble stats
        health_data = data.get("health", {})
        active_count = len([m for m in models if m.get("status") == "active"])
        total_requests = sum(m.get("requests", 0) for m in models)
        avg_latency = sum(m.get("avg_latency", 0) for m in models) / max(len(models), 1) * 1000
        self.update_ensemble_stats(active_count, total_requests, avg_latency)

        # Update quarantined models
        quarantined = data.get("quarantined", [])
        quarantined_items = [{"model": m, "reason": "Auto-quarantined due to health issues"} for m in quarantined]
        self.update_quarantined(quarantined_items)
