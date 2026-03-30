"""RAG View - Document management, indexing, and retrieval testing."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QPushButton, QLineEdit, QTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget,
    QComboBox, QFileDialog, QProgressBar, QGroupBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QSplitter,
    QListWidget, QListWidgetItem, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QColor
from nexus.gui.styles import COLORS


class DocumentItem(QFrame):
    """A document item in the document list."""

    delete_requested = Signal(str)
    reindex_requested = Signal(str)

    def __init__(self, doc_id: str, name: str, doc_type: str,
                 chunks: int, size: str, indexed: bool, parent=None):
        super().__init__(parent)
        self.doc_id = doc_id

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 8px;
            }}
            QFrame:hover {{
                border-color: {COLORS['accent_primary']};
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        # Icon based on type
        type_icons = {
            "pdf": "📄",
            "txt": "📝",
            "md": "📑",
            "html": "🌐",
            "json": "📊",
            "csv": "📈",
        }
        icon = QLabel(type_icons.get(doc_type.lower(), "📁"))
        icon.setStyleSheet("font-size: 24px;")
        layout.addWidget(icon)

        # Info
        info = QVBoxLayout()
        info.setSpacing(2)

        name_label = QLabel(name)
        name_label.setStyleSheet(f"font-weight: bold; color: {COLORS['text_primary']};")
        info.addWidget(name_label)

        meta = QLabel(f"{doc_type.upper()} • {size} • {chunks} chunks")
        meta.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        info.addWidget(meta)

        layout.addLayout(info, 1)

        # Status
        status_color = COLORS['accent_success'] if indexed else COLORS['accent_warning']
        status_text = "Indexed" if indexed else "Pending"
        status = QLabel(f"● {status_text}")
        status.setStyleSheet(f"color: {status_color}; font-size: 12px;")
        layout.addWidget(status)

        # Actions
        reindex_btn = QPushButton("↻")
        reindex_btn.setToolTip("Reindex")
        reindex_btn.setFixedSize(28, 28)
        reindex_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_tertiary']};
                color: {COLORS['text_primary']};
                border: none;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_primary']};
            }}
        """)
        reindex_btn.clicked.connect(lambda: self.reindex_requested.emit(self.doc_id))
        layout.addWidget(reindex_btn)

        delete_btn = QPushButton("✕")
        delete_btn.setToolTip("Delete")
        delete_btn.setFixedSize(28, 28)
        delete_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_tertiary']};
                color: {COLORS['accent_error']};
                border: none;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_error']};
                color: white;
            }}
        """)
        delete_btn.clicked.connect(lambda: self.delete_requested.emit(self.doc_id))
        layout.addWidget(delete_btn)


class RetrievalResult(QFrame):
    """A single retrieval result."""

    def __init__(self, content: str, source: str, score: float, parent=None):
        super().__init__(parent)

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        # Header with source and score
        header = QHBoxLayout()

        source_label = QLabel(f"📄 {source}")
        source_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        header.addWidget(source_label)

        header.addStretch()

        score_color = COLORS['accent_success'] if score >= 0.8 else (
            COLORS['accent_warning'] if score >= 0.5 else COLORS['accent_error']
        )
        score_label = QLabel(f"Score: {score:.2f}")
        score_label.setStyleSheet(f"color: {score_color}; font-weight: bold; font-size: 12px;")
        header.addWidget(score_label)

        layout.addLayout(header)

        # Content
        content_label = QLabel(content[:500] + "..." if len(content) > 500 else content)
        content_label.setWordWrap(True)
        content_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 13px;")
        layout.addWidget(content_label)


class RAGView(QWidget):
    """View for RAG document management and retrieval testing."""

    upload_requested = Signal(list)  # file paths
    delete_document = Signal(str)
    reindex_document = Signal(str)
    reindex_all = Signal()
    test_query = Signal(str, dict)  # query, settings
    refresh_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.document_items = []
        self.result_items = []
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(24, 24, 24, 24)

        # Header
        header = QHBoxLayout()
        title = QLabel("RAG System")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setObjectName("secondary")
        refresh_btn.clicked.connect(self.refresh_requested.emit)
        header.addWidget(refresh_btn)

        layout.addLayout(header)

        # Stats bar
        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)
        stats_layout = QHBoxLayout(stats_frame)
        stats_layout.setContentsMargins(20, 12, 20, 12)

        self.docs_count = QLabel("Documents: 0")
        self.docs_count.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: bold;")
        stats_layout.addWidget(self.docs_count)

        stats_layout.addWidget(self._separator())

        self.chunks_count = QLabel("Chunks: 0")
        self.chunks_count.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: bold;")
        stats_layout.addWidget(self.chunks_count)

        stats_layout.addWidget(self._separator())

        self.index_size = QLabel("Index Size: 0 MB")
        self.index_size.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: bold;")
        stats_layout.addWidget(self.index_size)

        stats_layout.addWidget(self._separator())

        self.embedding_model = QLabel("Embeddings: sentence-transformers")
        self.embedding_model.setStyleSheet(f"color: {COLORS['text_secondary']};")
        stats_layout.addWidget(self.embedding_model)

        stats_layout.addStretch()

        layout.addWidget(stats_frame)

        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left: Document Management
        docs_widget = QWidget()
        docs_layout = QVBoxLayout(docs_widget)
        docs_layout.setContentsMargins(0, 0, 0, 0)

        docs_header = QHBoxLayout()
        docs_title = QLabel("Documents")
        docs_title.setStyleSheet(f"font-weight: bold; font-size: 16px; color: {COLORS['text_primary']};")
        docs_header.addWidget(docs_title)
        docs_header.addStretch()

        upload_btn = QPushButton("Upload")
        upload_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_primary']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
            }}
        """)
        upload_btn.clicked.connect(self._open_file_dialog)
        docs_header.addWidget(upload_btn)

        reindex_all_btn = QPushButton("Reindex All")
        reindex_all_btn.setObjectName("secondary")
        reindex_all_btn.clicked.connect(self.reindex_all.emit)
        docs_header.addWidget(reindex_all_btn)

        docs_layout.addLayout(docs_header)

        # Document list
        self.docs_scroll = QScrollArea()
        self.docs_scroll.setWidgetResizable(True)
        self.docs_scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {COLORS['bg_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)

        self.docs_container = QWidget()
        self.docs_list_layout = QVBoxLayout(self.docs_container)
        self.docs_list_layout.setSpacing(8)
        self.docs_list_layout.setContentsMargins(8, 8, 8, 8)
        self.docs_list_layout.addStretch()

        self.docs_scroll.setWidget(self.docs_container)
        docs_layout.addWidget(self.docs_scroll)

        splitter.addWidget(docs_widget)

        # Right: Retrieval Testing
        test_widget = QWidget()
        test_layout = QVBoxLayout(test_widget)
        test_layout.setContentsMargins(0, 0, 0, 0)

        test_title = QLabel("Retrieval Testing")
        test_title.setStyleSheet(f"font-weight: bold; font-size: 16px; color: {COLORS['text_primary']};")
        test_layout.addWidget(test_title)

        # Query input
        query_frame = QFrame()
        query_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)
        query_layout = QVBoxLayout(query_frame)
        query_layout.setContentsMargins(12, 12, 12, 12)

        self.query_input = QTextEdit()
        self.query_input.setPlaceholderText("Enter your query to test retrieval...")
        self.query_input.setMaximumHeight(80)
        self.query_input.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['bg_input']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 8px;
                color: {COLORS['text_primary']};
            }}
        """)
        query_layout.addWidget(self.query_input)

        # Settings row
        settings_row = QHBoxLayout()

        settings_row.addWidget(QLabel("Top K:"))
        self.top_k = QSpinBox()
        self.top_k.setRange(1, 20)
        self.top_k.setValue(5)
        self.top_k.setFixedWidth(60)
        settings_row.addWidget(self.top_k)

        settings_row.addWidget(QLabel("Threshold:"))
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 1.0)
        self.threshold.setValue(0.5)
        self.threshold.setSingleStep(0.1)
        self.threshold.setFixedWidth(70)
        settings_row.addWidget(self.threshold)

        self.hybrid_search = QCheckBox("Hybrid Search")
        self.hybrid_search.setChecked(True)
        settings_row.addWidget(self.hybrid_search)

        self.rerank = QCheckBox("Rerank")
        self.rerank.setChecked(True)
        settings_row.addWidget(self.rerank)

        settings_row.addStretch()

        search_btn = QPushButton("Search")
        search_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_primary']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                font-weight: bold;
            }}
        """)
        search_btn.clicked.connect(self._do_search)
        settings_row.addWidget(search_btn)

        query_layout.addLayout(settings_row)
        test_layout.addWidget(query_frame)

        # Results
        results_label = QLabel("Results")
        results_label.setStyleSheet(f"font-weight: bold; color: {COLORS['text_secondary']}; margin-top: 8px;")
        test_layout.addWidget(results_label)

        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {COLORS['bg_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)

        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setSpacing(8)
        self.results_layout.setContentsMargins(8, 8, 8, 8)
        self.results_layout.addStretch()

        self.results_scroll.setWidget(self.results_container)
        test_layout.addWidget(self.results_scroll)

        splitter.addWidget(test_widget)

        # Set splitter sizes
        splitter.setSizes([400, 600])

        layout.addWidget(splitter, 1)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setFixedHeight(4)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: {COLORS['bg_input']};
                border: none;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['accent_primary']};
            }}
        """)
        layout.addWidget(self.progress)

    def _separator(self) -> QFrame:
        """Create a vertical separator."""
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"color: {COLORS['border']};")
        return sep

    def _open_file_dialog(self):
        """Open file dialog for document upload."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Documents",
            "",
            "Documents (*.pdf *.txt *.md *.html *.json *.csv);;All Files (*)"
        )
        if files:
            self.upload_requested.emit(files)

    def _do_search(self):
        """Execute the test query."""
        query = self.query_input.toPlainText().strip()
        if not query:
            return

        settings = {
            "top_k": self.top_k.value(),
            "threshold": self.threshold.value(),
            "hybrid": self.hybrid_search.isChecked(),
            "rerank": self.rerank.isChecked(),
        }
        self.test_query.emit(query, settings)

    def update_stats(self, docs: int, chunks: int, size_mb: float, embedding: str):
        """Update the stats bar."""
        self.docs_count.setText(f"Documents: {docs}")
        self.chunks_count.setText(f"Chunks: {chunks:,}")
        self.index_size.setText(f"Index Size: {size_mb:.1f} MB")
        self.embedding_model.setText(f"Embeddings: {embedding}")

    def update_documents(self, documents: list):
        """Update the document list."""
        # Clear existing items
        for item in self.document_items:
            item.deleteLater()
        self.document_items.clear()

        # Add new items
        for doc in documents:
            item = DocumentItem(
                doc_id=doc.get('id', ''),
                name=doc.get('name', 'Unknown'),
                doc_type=doc.get('type', 'txt'),
                chunks=doc.get('chunks', 0),
                size=doc.get('size', '0 KB'),
                indexed=doc.get('indexed', False)
            )
            item.delete_requested.connect(self.delete_document.emit)
            item.reindex_requested.connect(self.reindex_document.emit)
            self.docs_list_layout.insertWidget(
                self.docs_list_layout.count() - 1, item
            )
            self.document_items.append(item)

    def update_results(self, results: list):
        """Update the retrieval results."""
        # Clear existing results
        for item in self.result_items:
            item.deleteLater()
        self.result_items.clear()

        # Add new results
        for result in results:
            item = RetrievalResult(
                content=result.get('content', ''),
                source=result.get('source', 'Unknown'),
                score=result.get('score', 0.0)
            )
            self.results_layout.insertWidget(
                self.results_layout.count() - 1, item
            )
            self.result_items.append(item)

    def set_loading(self, loading: bool):
        """Show/hide loading indicator."""
        self.progress.setVisible(loading)
        if loading:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(100)

    def update_data(self, data: dict):
        """Update all view data from a single data dictionary.

        Args:
            data: Dictionary containing 'documents', 'indices', 'stats' keys
        """
        # Update documents
        documents = data.get("documents", [])
        # Convert indexed_at to indexed boolean for display
        for doc in documents:
            doc["indexed"] = doc.get("status") == "indexed"
            doc["size"] = f"{doc.get('chunks', 0) * 0.5:.1f} KB"  # Estimate size
        self.update_documents(documents)

        # Update stats
        stats = data.get("stats", {})
        indices = data.get("indices", [])
        embedding_model = indices[0].get("embedding_model", "unknown") if indices else "unknown"
        self.update_stats(
            docs=stats.get("total_documents", len(documents)),
            chunks=stats.get("total_chunks", 0),
            size_mb=stats.get("index_size_mb", 0),
            embedding=embedding_model
        )

    def show_query_results(self, results: dict):
        """Show query test results.

        Args:
            results: Dictionary with 'query', 'results', 'time_ms' keys
        """
        if "error" in results:
            # Show error message
            self.update_results([{
                "content": f"Error: {results['error']}",
                "source": "Error",
                "score": 0.0
            }])
            return

        # Update results display
        self.update_results(results.get("results", []))

        # Could also display query time: results.get("time_ms")
