"""Settings view."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox,
    QFormLayout, QScrollArea, QFrame, QComboBox, QMessageBox
)
from PySide6.QtCore import Qt, Signal
from nexus.gui.styles import COLORS


class SettingsView(QWidget):
    """View for configuring Nexus settings."""

    save_settings = Signal(dict)
    reset_settings = Signal()
    test_connection = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Header
        header = QHBoxLayout()
        title = QLabel("Settings")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()

        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self.on_save)
        header.addWidget(save_btn)

        layout.addLayout(header)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: transparent;
                border: none;
            }}
        """)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(16)

        # Storage Settings
        storage_group = self._create_group("Storage")
        storage_form = QFormLayout()

        self.chroma_path = QLineEdit()
        self.chroma_path.setPlaceholderText("data/chroma")
        storage_form.addRow("ChromaDB Path:", self.chroma_path)

        self.sqlite_path = QLineEdit()
        self.sqlite_path.setPlaceholderText("data/sqlite/nexus.db")
        storage_form.addRow("SQLite Path:", self.sqlite_path)

        self.backup_path = QLineEdit()
        self.backup_path.setPlaceholderText("data/backups")
        storage_form.addRow("Backup Path:", self.backup_path)

        self.auto_backup = QCheckBox("Enable automatic backups")
        self.auto_backup.setChecked(True)
        storage_form.addRow("", self.auto_backup)

        self.backup_interval = QSpinBox()
        self.backup_interval.setRange(1, 168)
        self.backup_interval.setValue(24)
        self.backup_interval.setSuffix(" hours")
        storage_form.addRow("Backup Interval:", self.backup_interval)

        storage_group.setLayout(storage_form)
        content_layout.addWidget(storage_group)

        # Embedding Settings
        embed_group = self._create_group("Embeddings")
        embed_form = QFormLayout()

        # Backend selection
        self.embed_backend = QComboBox()
        self.embed_backend.addItems(["auto", "builtin", "ollama"])
        self.embed_backend.setToolTip(
            "auto: Use built-in if available, fallback to Ollama\n"
            "builtin: Use sentence-transformers (no external deps)\n"
            "ollama: Use Ollama for embeddings"
        )
        embed_form.addRow("Backend:", self.embed_backend)

        # Built-in model
        self.builtin_model = QComboBox()
        self.builtin_model.addItems([
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-MiniLM-L6-v2"
        ])
        self.builtin_model.setEditable(True)
        self.builtin_model.setToolTip("Model for built-in embeddings (sentence-transformers)")
        embed_form.addRow("Built-in Model:", self.builtin_model)

        # Ollama settings
        self.ollama_url = QLineEdit()
        self.ollama_url.setPlaceholderText("http://localhost:11434")
        embed_form.addRow("Ollama URL:", self.ollama_url)

        test_btn = QPushButton("Test Connection")
        test_btn.setObjectName("secondary")
        test_btn.clicked.connect(self.on_test_connection)
        embed_form.addRow("", test_btn)

        self.ollama_model = QComboBox()
        self.ollama_model.addItems([
            "nomic-embed-text",
            "mxbai-embed-large",
            "all-minilm",
            "bge-base-en"
        ])
        self.ollama_model.setEditable(True)
        embed_form.addRow("Ollama Model:", self.ollama_model)

        self.ollama_fallback = QComboBox()
        self.ollama_fallback.addItems([
            "mxbai-embed-large",
            "nomic-embed-text",
            "all-minilm",
            "bge-base-en"
        ])
        self.ollama_fallback.setEditable(True)
        embed_form.addRow("Ollama Fallback:", self.ollama_fallback)

        self.auto_start_ollama = QCheckBox("Auto-start Ollama if not running")
        self.auto_start_ollama.setChecked(True)
        embed_form.addRow("", self.auto_start_ollama)

        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 128)
        self.batch_size.setValue(32)
        embed_form.addRow("Batch Size:", self.batch_size)

        self.embed_timeout = QDoubleSpinBox()
        self.embed_timeout.setRange(5.0, 120.0)
        self.embed_timeout.setValue(30.0)
        self.embed_timeout.setSuffix(" sec")
        embed_form.addRow("Timeout:", self.embed_timeout)

        self.cache_enabled = QCheckBox("Enable embedding cache")
        self.cache_enabled.setChecked(True)
        embed_form.addRow("", self.cache_enabled)

        embed_group.setLayout(embed_form)
        content_layout.addWidget(embed_group)

        # Memory Settings
        memory_group = self._create_group("Memory")
        memory_form = QFormLayout()

        self.max_chunk_size = QSpinBox()
        self.max_chunk_size.setRange(100, 5000)
        self.max_chunk_size.setValue(1500)
        memory_form.addRow("Max Chunk Size:", self.max_chunk_size)

        self.chunk_overlap = QSpinBox()
        self.chunk_overlap.setRange(0, 500)
        self.chunk_overlap.setValue(200)
        memory_form.addRow("Chunk Overlap:", self.chunk_overlap)

        self.min_chunk_size = QSpinBox()
        self.min_chunk_size.setRange(10, 500)
        self.min_chunk_size.setValue(100)
        memory_form.addRow("Min Chunk Size:", self.min_chunk_size)

        self.context_messages = QSpinBox()
        self.context_messages.setRange(0, 10)
        self.context_messages.setValue(2)
        memory_form.addRow("Context Messages:", self.context_messages)

        memory_group.setLayout(memory_form)
        content_layout.addWidget(memory_group)

        # Truth Verification Settings
        truth_group = self._create_group("Truth Verification")
        truth_form = QFormLayout()

        self.high_confidence = QDoubleSpinBox()
        self.high_confidence.setRange(0.0, 1.0)
        self.high_confidence.setValue(0.7)
        self.high_confidence.setSingleStep(0.05)
        truth_form.addRow("High Confidence:", self.high_confidence)

        self.medium_confidence = QDoubleSpinBox()
        self.medium_confidence.setRange(0.0, 1.0)
        self.medium_confidence.setValue(0.4)
        self.medium_confidence.setSingleStep(0.05)
        truth_form.addRow("Medium Confidence:", self.medium_confidence)

        self.strict_mode = QCheckBox("Enable strict verification mode")
        truth_form.addRow("", self.strict_mode)

        truth_group.setLayout(truth_form)
        content_layout.addWidget(truth_group)

        # Logging Settings
        log_group = self._create_group("Logging")
        log_form = QFormLayout()

        self.log_level = QComboBox()
        self.log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.log_level.setCurrentText("INFO")
        log_form.addRow("Log Level:", self.log_level)

        self.log_file = QLineEdit()
        self.log_file.setPlaceholderText("logs/nexus.log")
        log_form.addRow("Log File:", self.log_file)

        self.log_max_size = QSpinBox()
        self.log_max_size.setRange(1, 100)
        self.log_max_size.setValue(10)
        self.log_max_size.setSuffix(" MB")
        log_form.addRow("Max Size:", self.log_max_size)

        self.log_backup_count = QSpinBox()
        self.log_backup_count.setRange(1, 20)
        self.log_backup_count.setValue(5)
        log_form.addRow("Backup Count:", self.log_backup_count)

        log_group.setLayout(log_form)
        content_layout.addWidget(log_group)

        # Reset button
        reset_layout = QHBoxLayout()
        reset_layout.addStretch()

        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setObjectName("secondary")
        reset_btn.clicked.connect(self.on_reset)
        reset_layout.addWidget(reset_btn)

        content_layout.addLayout(reset_layout)
        content_layout.addStretch()

        scroll.setWidget(content)
        layout.addWidget(scroll)

    def _create_group(self, title: str) -> QGroupBox:
        """Create a styled group box."""
        group = QGroupBox(title)
        group.setStyleSheet(f"""
            QGroupBox {{
                font-weight: 600;
                font-size: 14px;
                color: {COLORS['text_primary']};
                background-color: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                margin-top: 16px;
                padding: 16px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }}
        """)
        return group

    def load_settings(self, config: dict):
        """Load settings from config dict."""
        storage = config.get("storage", {})
        self.chroma_path.setText(storage.get("chroma_path", "data/chroma"))
        self.sqlite_path.setText(storage.get("sqlite_path", "data/sqlite/nexus.db"))
        self.backup_path.setText(storage.get("backup_path", "data/backups"))
        self.auto_backup.setChecked(storage.get("auto_backup", True))
        self.backup_interval.setValue(storage.get("backup_interval_hours", 24))

        embed = config.get("embedding", {})
        self.embed_backend.setCurrentText(embed.get("backend", "auto"))
        self.builtin_model.setCurrentText(embed.get("builtin_model", "all-MiniLM-L6-v2"))
        self.ollama_url.setText(embed.get("ollama_url", "http://localhost:11434"))
        self.ollama_model.setCurrentText(embed.get("ollama_model", embed.get("primary_model", "nomic-embed-text")))
        self.ollama_fallback.setCurrentText(embed.get("ollama_fallback_model", embed.get("fallback_model", "mxbai-embed-large")))
        self.auto_start_ollama.setChecked(embed.get("auto_start_ollama", True))
        self.batch_size.setValue(embed.get("batch_size", 32))
        self.embed_timeout.setValue(embed.get("timeout_seconds", 30.0))
        self.cache_enabled.setChecked(embed.get("cache_enabled", True))

        memory = config.get("memory", {})
        self.max_chunk_size.setValue(memory.get("max_chunk_size", 1500))
        self.chunk_overlap.setValue(memory.get("chunk_overlap", 200))
        self.min_chunk_size.setValue(memory.get("min_chunk_size", 100))
        self.context_messages.setValue(memory.get("include_context_messages", 2))

        truth = config.get("truth", {})
        self.high_confidence.setValue(truth.get("high_confidence_threshold", 0.7))
        self.medium_confidence.setValue(truth.get("medium_confidence_threshold", 0.4))
        self.strict_mode.setChecked(truth.get("strict_mode", False))

        logging = config.get("logging", {})
        self.log_level.setCurrentText(logging.get("level", "INFO"))
        self.log_file.setText(logging.get("file", "logs/nexus.log"))
        self.log_max_size.setValue(logging.get("max_size_mb", 10))
        self.log_backup_count.setValue(logging.get("backup_count", 5))

    def get_settings(self) -> dict:
        """Get current settings as config dict."""
        return {
            "storage": {
                "chroma_path": self.chroma_path.text() or "data/chroma",
                "sqlite_path": self.sqlite_path.text() or "data/sqlite/nexus.db",
                "backup_path": self.backup_path.text() or "data/backups",
                "auto_backup": self.auto_backup.isChecked(),
                "backup_interval_hours": self.backup_interval.value()
            },
            "embedding": {
                "backend": self.embed_backend.currentText(),
                "builtin_model": self.builtin_model.currentText(),
                "ollama_url": self.ollama_url.text() or "http://localhost:11434",
                "ollama_model": self.ollama_model.currentText(),
                "ollama_fallback_model": self.ollama_fallback.currentText(),
                "auto_start_ollama": self.auto_start_ollama.isChecked(),
                "batch_size": self.batch_size.value(),
                "timeout_seconds": self.embed_timeout.value(),
                "cache_enabled": self.cache_enabled.isChecked()
            },
            "memory": {
                "max_chunk_size": self.max_chunk_size.value(),
                "chunk_overlap": self.chunk_overlap.value(),
                "min_chunk_size": self.min_chunk_size.value(),
                "include_context_messages": self.context_messages.value()
            },
            "truth": {
                "high_confidence_threshold": self.high_confidence.value(),
                "medium_confidence_threshold": self.medium_confidence.value(),
                "strict_mode": self.strict_mode.isChecked()
            },
            "logging": {
                "level": self.log_level.currentText(),
                "file": self.log_file.text() or "logs/nexus.log",
                "max_size_mb": self.log_max_size.value(),
                "backup_count": self.log_backup_count.value()
            }
        }

    def on_save(self):
        """Handle save button click."""
        settings = self.get_settings()
        self.save_settings.emit(settings)

    def on_reset(self):
        """Handle reset button click."""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.reset_settings.emit()

    def on_test_connection(self):
        """Handle test connection button click."""
        self.test_connection.emit()

    def show_connection_result(self, success: bool, message: str):
        """Show connection test result."""
        if success:
            QMessageBox.information(self, "Connection Test", message)
        else:
            QMessageBox.warning(self, "Connection Test", message)
