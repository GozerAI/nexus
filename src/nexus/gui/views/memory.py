"""Memory search view."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QScrollArea, QFrame, QSplitter
)
from PySide6.QtCore import Qt, Signal
from nexus.gui.styles import COLORS


class MemoryView(QWidget):
    """View for searching and browsing memories."""

    search_requested = Signal(str, int)  # query, limit
    topic_history_requested = Signal(str)
    load_topics = Signal()  # Signal to load topic history

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Header
        header = QHBoxLayout()
        title = QLabel("Memory Search")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        # Search area
        search_layout = QHBoxLayout()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search for topics, conversations, or context...")
        self.search_input.returnPressed.connect(self.on_search)
        search_layout.addWidget(self.search_input)

        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self.on_search)
        search_layout.addWidget(search_btn)

        history_btn = QPushButton("Topic History")
        history_btn.setObjectName("secondary")
        history_btn.clicked.connect(self.on_topic_history)
        search_layout.addWidget(history_btn)

        layout.addLayout(search_layout)

        # Stats bar
        self.stats_label = QLabel("Enter a search query to find relevant memories")
        self.stats_label.setStyleSheet(f"color: {COLORS['text_muted']}; padding: 8px 0;")
        layout.addWidget(self.stats_label)

        # Results area
        splitter = QSplitter(Qt.Horizontal)

        # Results list
        results_container = QWidget()
        results_layout = QVBoxLayout(results_container)
        results_layout.setContentsMargins(0, 0, 0, 0)

        results_title = QLabel("Results")
        results_title.setStyleSheet("font-weight: 600;")
        results_layout.addWidget(results_title)

        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        self.results_layout.setSpacing(12)
        self.results_layout.setAlignment(Qt.AlignTop)
        self.results_scroll.setWidget(self.results_widget)
        results_layout.addWidget(self.results_scroll)

        splitter.addWidget(results_container)

        # Detail view
        detail_container = QFrame()
        detail_container.setObjectName("card")
        detail_layout = QVBoxLayout(detail_container)

        detail_title = QLabel("Details")
        detail_title.setStyleSheet("font-weight: 600;")
        detail_layout.addWidget(detail_title)

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['bg_tertiary']};
                border: none;
                border-radius: 8px;
                padding: 12px;
            }}
        """)
        self.detail_text.setPlaceholderText("Select a result to see details")
        detail_layout.addWidget(self.detail_text)

        splitter.addWidget(detail_container)
        splitter.setSizes([400, 400])

        layout.addWidget(splitter)

    def on_search(self):
        query = self.search_input.text().strip()
        if query:
            self.search_requested.emit(query, 20)  # default limit of 20

    def on_topic_history(self):
        query = self.search_input.text().strip()
        if query:
            self.topic_history_requested.emit(query)
        else:
            # Load all recent topics
            self.load_topics.emit()

    def show_results(self, results, query=""):
        """Display search results."""
        # Clear existing
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not results:
            self.stats_label.setText(f"No results found for '{query}'")
            return

        self.stats_label.setText(f"Found {len(results)} results for '{query}'")

        for i, hit in enumerate(results):
            card = self.create_result_card(i + 1, hit)
            self.results_layout.addWidget(card)

    def create_result_card(self, num, hit):
        """Create a result card widget."""
        card = QFrame()
        card.setObjectName("card")
        card.setCursor(Qt.PointingHandCursor)
        card.setStyleSheet(f"""
            QFrame#card:hover {{
                border-color: {COLORS['accent_primary']};
            }}
        """)

        layout = QVBoxLayout(card)
        layout.setSpacing(8)

        # Header
        header = QHBoxLayout()
        num_label = QLabel(f"#{num}")
        num_label.setStyleSheet(f"color: {COLORS['accent_primary']}; font-weight: 600;")
        header.addWidget(num_label)

        relevance = int(hit.relevance * 100)
        relevance_label = QLabel(f"{relevance}% relevant")
        relevance_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        header.addWidget(relevance_label)
        header.addStretch()
        layout.addLayout(header)

        # Snippet
        snippet = QLabel(hit.snippet)
        snippet.setWordWrap(True)
        snippet.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(snippet)

        # Conversation info
        if hit.conversation_title:
            conv_label = QLabel(f"From: {hit.conversation_title}")
            conv_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
            layout.addWidget(conv_label)

        # Click handler to show details
        def show_detail(event, h=hit):
            self.show_detail(h)
        card.mousePressEvent = show_detail

        return card

    def show_detail(self, hit):
        """Show full details for a hit."""
        html = f"""
        <style>
            h3 {{ color: {COLORS['accent_primary']}; margin-bottom: 8px; }}
            p {{ color: {COLORS['text_secondary']}; margin: 8px 0; }}
            .label {{ color: {COLORS['text_muted']}; font-size: 11px; }}
        </style>
        <h3>Memory Detail</h3>
        <p class="label">Relevance: {int(hit.relevance * 100)}%</p>
        <p class="label">Conversation: {hit.conversation_title or 'Unknown'}</p>
        <hr style="border-color: {COLORS['border']};">
        <p>{hit.chunk.text}</p>
        """
        self.detail_text.setHtml(html)

    def show_topic_history(self, mentions, topic=""):
        """Display topic history."""
        # Clear existing
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not mentions:
            self.stats_label.setText(f"No history found for '{topic}'")
            return

        self.stats_label.setText(f"Found {len(mentions)} mentions of '{topic}' (chronological)")

        for mention in mentions:
            card = QFrame()
            card.setObjectName("card")

            layout = QVBoxLayout(card)
            layout.setSpacing(8)

            # Timestamp
            ts_label = QLabel(mention.timestamp.strftime("%Y-%m-%d %H:%M"))
            ts_label.setStyleSheet(f"color: {COLORS['accent_primary']}; font-weight: 600;")
            layout.addWidget(ts_label)

            # Text
            text_label = QLabel(mention.chunk_text)
            text_label.setWordWrap(True)
            text_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
            layout.addWidget(text_label)

            self.results_layout.addWidget(card)

    def update_topic_history(self, topics):
        """Display recent topic history."""
        # Clear existing
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not topics:
            self.stats_label.setText("No topic history available")
            return

        self.stats_label.setText(f"Recent topic history ({len(topics)} topics)")

        for topic in topics:
            card = QFrame()
            card.setObjectName("card")

            layout = QVBoxLayout(card)
            layout.setSpacing(8)

            # Topic name
            name_label = QLabel(topic.get("topic", "Unknown"))
            name_label.setStyleSheet(f"color: {COLORS['accent_primary']}; font-weight: 600;")
            layout.addWidget(name_label)

            # Count/info
            if "count" in topic:
                count_label = QLabel(f"Mentioned {topic['count']} times")
                count_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
                layout.addWidget(count_label)

            self.results_layout.addWidget(card)
