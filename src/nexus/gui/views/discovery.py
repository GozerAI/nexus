"""Discovery View - Browse and manage discovered resources (models, datasets, APIs)."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QGridLayout, QPushButton, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget,
    QComboBox, QCheckBox, QGroupBox, QSizePolicy, QProgressBar
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QColor
from nexus.gui.styles import COLORS


class ResourceCard(QFrame):
    """Card displaying a discovered resource."""

    action_requested = Signal(str, str)  # resource_id, action

    def __init__(self, resource_id: str, name: str, resource_type: str,
                 source: str, description: str = "", parent=None):
        super().__init__(parent)
        self.resource_id = resource_id
        self.setObjectName("resourceCard")

        self.setStyleSheet(f"""
            QFrame#resourceCard {{
                background-color: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 10px;
            }}
            QFrame#resourceCard:hover {{
                border-color: {COLORS['accent_primary']};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(16, 12, 16, 12)

        # Header
        header = QHBoxLayout()

        name_label = QLabel(name)
        name_label.setStyleSheet(f"font-weight: bold; color: {COLORS['text_primary']}; font-size: 14px;")
        name_label.setWordWrap(True)
        header.addWidget(name_label, 1)

        type_colors = {
            "model": COLORS['accent_primary'],
            "dataset": COLORS['accent_success'],
            "api": COLORS['accent_purple'],
            "tool": COLORS['accent_warning'],
            "library": COLORS['accent_info'],
        }

        type_badge = QLabel(resource_type.upper())
        type_badge.setStyleSheet(f"""
            background-color: {type_colors.get(resource_type, COLORS['bg_tertiary'])};
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: bold;
        """)
        header.addWidget(type_badge)

        layout.addLayout(header)

        # Source
        source_label = QLabel(f"Source: {source}")
        source_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        layout.addWidget(source_label)

        # Description
        if description:
            desc_label = QLabel(description[:150] + "..." if len(description) > 150 else description)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
            layout.addWidget(desc_label)

        # Actions
        actions = QHBoxLayout()
        actions.addStretch()

        view_btn = QPushButton("Details")
        view_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_tertiary']};
                color: {COLORS['text_primary']};
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_input']};
            }}
        """)
        view_btn.clicked.connect(lambda: self.action_requested.emit(self.resource_id, "view"))
        actions.addWidget(view_btn)

        add_btn = QPushButton("Add")
        add_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_primary']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_secondary']};
            }}
        """)
        add_btn.clicked.connect(lambda: self.action_requested.emit(self.resource_id, "add"))
        actions.addWidget(add_btn)

        layout.addLayout(actions)


class DiscoveryView(QWidget):
    """View for discovering and managing AI resources."""

    search_requested = Signal(str, str)  # query, resource_type
    refresh_requested = Signal()
    scan_requested = Signal(str)  # source
    resource_action = Signal(str, str)  # resource_id, action

    def __init__(self, parent=None):
        super().__init__(parent)
        self.resource_cards = []
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(24, 24, 24, 24)

        # Header
        header = QHBoxLayout()
        title = QLabel("Resource Discovery")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()

        refresh_btn = QPushButton("Refresh All")
        refresh_btn.setObjectName("secondary")
        refresh_btn.clicked.connect(self.refresh_requested.emit)
        header.addWidget(refresh_btn)

        layout.addLayout(header)

        # Search and filter bar
        search_frame = QFrame()
        search_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)
        search_layout = QHBoxLayout(search_frame)
        search_layout.setContentsMargins(16, 12, 16, 12)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search resources...")
        self.search_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {COLORS['bg_input']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 8px 12px;
                color: {COLORS['text_primary']};
            }}
            QLineEdit:focus {{
                border-color: {COLORS['accent_primary']};
            }}
        """)
        self.search_input.returnPressed.connect(self._do_search)
        search_layout.addWidget(self.search_input, 1)

        self.type_filter = QComboBox()
        self.type_filter.addItems(["All Types", "Models", "Datasets", "APIs", "Tools", "Libraries"])
        self.type_filter.setMinimumWidth(120)
        search_layout.addWidget(self.type_filter)

        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self._do_search)
        search_layout.addWidget(search_btn)

        layout.addWidget(search_frame)

        # Tabs for different sources
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

        # All Resources Tab
        all_tab = QWidget()
        all_layout = QVBoxLayout(all_tab)
        all_layout.setContentsMargins(16, 16, 16, 16)

        self.all_scroll = QScrollArea()
        self.all_scroll.setWidgetResizable(True)
        self.all_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.all_container = QWidget()
        self.all_grid = QGridLayout(self.all_container)
        self.all_grid.setSpacing(16)
        self.all_grid.setContentsMargins(0, 0, 0, 0)

        self.all_scroll.setWidget(self.all_container)
        all_layout.addWidget(self.all_scroll)

        tabs.addTab(all_tab, "All Resources")

        # Ollama Tab
        ollama_tab = self._create_source_tab(
            "Ollama",
            "Discover locally running Ollama models",
            "ollama"
        )
        tabs.addTab(ollama_tab, "Ollama")

        # HuggingFace Tab
        hf_tab = self._create_source_tab(
            "HuggingFace",
            "Browse models and datasets from HuggingFace Hub",
            "huggingface"
        )
        tabs.addTab(hf_tab, "HuggingFace")

        # OpenRouter Tab
        openrouter_tab = self._create_source_tab(
            "OpenRouter",
            "Access 100+ models through OpenRouter API",
            "openrouter"
        )
        tabs.addTab(openrouter_tab, "OpenRouter")

        # GitHub Tab
        github_tab = self._create_source_tab(
            "GitHub",
            "Find AI tools and libraries on GitHub",
            "github"
        )
        tabs.addTab(github_tab, "GitHub")

        # Local Tab
        local_tab = self._create_source_tab(
            "Local Machine",
            "Scan local system for AI resources",
            "local"
        )
        tabs.addTab(local_tab, "Local")

        layout.addWidget(tabs, 1)

        # Status bar
        status_layout = QHBoxLayout()

        self.status_label = QLabel("Ready to discover resources")
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        status_layout.addWidget(self.status_label)

        status_layout.addStretch()

        self.progress = QProgressBar()
        self.progress.setFixedWidth(200)
        self.progress.setFixedHeight(6)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: {COLORS['bg_input']};
                border: none;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['accent_primary']};
                border-radius: 3px;
            }}
        """)
        status_layout.addWidget(self.progress)

        layout.addLayout(status_layout)

    def _create_source_tab(self, title: str, description: str, source: str) -> QWidget:
        """Create a tab for a specific discovery source."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Header
        header = QHBoxLayout()

        info = QVBoxLayout()
        title_label = QLabel(title)
        title_label.setStyleSheet(f"font-weight: bold; font-size: 16px; color: {COLORS['text_primary']};")
        info.addWidget(title_label)

        desc_label = QLabel(description)
        desc_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        info.addWidget(desc_label)

        header.addLayout(info)
        header.addStretch()

        scan_btn = QPushButton(f"Scan {title}")
        scan_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_primary']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_secondary']};
            }}
        """)
        scan_btn.clicked.connect(lambda: self.scan_requested.emit(source))
        header.addWidget(scan_btn)

        layout.addLayout(header)

        # Results table
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Name", "Type", "Description", "Actions"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        table.setColumnWidth(1, 80)
        table.setColumnWidth(3, 100)
        table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
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

        # Store reference for updates
        setattr(self, f"{source}_table", table)

        layout.addWidget(table)

        return tab

    def _do_search(self):
        """Trigger a search."""
        query = self.search_input.text()
        type_filter = self.type_filter.currentText().lower().replace(" ", "_")
        if type_filter == "all_types":
            type_filter = ""
        self.search_requested.emit(query, type_filter)

    def update_resources(self, resources: list):
        """Update the all resources grid."""
        # Clear existing cards
        for card in self.resource_cards:
            card.deleteLater()
        self.resource_cards.clear()

        # Create new cards
        for i, resource in enumerate(resources):
            card = ResourceCard(
                resource_id=resource.get('id', ''),
                name=resource.get('name', 'Unknown'),
                resource_type=resource.get('type', 'unknown'),
                source=resource.get('source', 'Unknown'),
                description=resource.get('description', '')
            )
            card.action_requested.connect(self.resource_action.emit)

            row = i // 3
            col = i % 3
            self.all_grid.addWidget(card, row, col)
            self.resource_cards.append(card)

    def update_source_table(self, source: str, items: list):
        """Update a specific source's table."""
        table = getattr(self, f"{source}_table", None)
        if not table:
            return

        table.setRowCount(len(items))

        for i, item in enumerate(items):
            table.setItem(i, 0, QTableWidgetItem(item.get('name', '')))
            table.setItem(i, 1, QTableWidgetItem(item.get('type', '')))
            table.setItem(i, 2, QTableWidgetItem(item.get('description', '')[:100]))

            add_btn = QPushButton("Add")
            add_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['accent_primary']};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 4px 8px;
                }}
            """)
            item_id = item.get('id', '')
            add_btn.clicked.connect(lambda checked, rid=item_id: self.resource_action.emit(rid, "add"))
            table.setCellWidget(i, 3, add_btn)

    def set_status(self, message: str, show_progress: bool = False):
        """Update the status bar."""
        self.status_label.setText(message)
        self.progress.setVisible(show_progress)
        if show_progress:
            self.progress.setRange(0, 0)  # Indeterminate

    def set_progress(self, value: int, maximum: int = 100):
        """Set progress bar value."""
        self.progress.setRange(0, maximum)
        self.progress.setValue(value)
        self.progress.setVisible(value < maximum)

    def update_data(self, data: dict):
        """Update all view data from a single data dictionary.

        Args:
            data: Dictionary containing 'sources' with nested resource data
        """
        sources = data.get("sources", {})

        # Collect all resources for the main grid
        all_resources = []

        # Update each source tab
        for source_name, source_data in sources.items():
            resources = source_data.get("resources", [])

            # Add source to each resource for display
            for r in resources:
                r["source"] = source_name
                all_resources.append(r)

            # Update source-specific table
            self.update_source_table(source_name, resources)

        # Update the all resources grid
        self.update_resources(all_resources)

        # Update status
        total = len(all_resources)
        self.set_status(f"Found {total} resources across {len(sources)} sources")

    def update_results(self, results: list):
        """Update view with search results.

        Args:
            results: List of resource dictionaries from search
        """
        self.update_resources(results)
        self.set_status(f"Search found {len(results)} resources")
