"""Chat view for interacting with Nexus AI."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QTextEdit, QPushButton, QSizePolicy, QLineEdit
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont
from nexus.gui.styles import COLORS


class ChatMessage(QFrame):
    """A single chat message bubble."""

    # Signal emitted when feedback is given: (message_id, is_positive)
    feedback_given = Signal(str, bool)

    def __init__(self, text: str, is_user: bool = True, message_id: str = None,
                 model_name: str = None, request_id: str = None, parent=None):
        super().__init__(parent)
        self.setObjectName("chatMessage")
        self.message_id = message_id
        self.model_name = model_name
        self.request_id = request_id
        self._feedback_submitted = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        if is_user:
            layout.addStretch()

        # Main content container
        content_container = QVBoxLayout()
        content_container.setSpacing(4)

        bubble = QFrame()
        bubble.setObjectName("userBubble" if is_user else "assistantBubble")
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(12, 8, 12, 8)

        label = QLabel(text)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bubble_layout.addWidget(label)

        bg_color = COLORS['accent_primary'] if is_user else COLORS['bg_tertiary']
        text_color = 'white' if is_user else COLORS['text_primary']

        bubble.setStyleSheet(f"""
            QFrame#userBubble, QFrame#assistantBubble {{
                background-color: {bg_color};
                border-radius: 12px;
                max-width: 600px;
            }}
            QLabel {{
                color: {text_color};
                font-size: 14px;
            }}
        """)

        content_container.addWidget(bubble)

        # Add feedback buttons for AI responses (not user messages)
        if not is_user and message_id:
            self.feedback_frame = QFrame()
            feedback_layout = QHBoxLayout(self.feedback_frame)
            feedback_layout.setContentsMargins(4, 2, 4, 2)
            feedback_layout.setSpacing(8)

            # Thumbs up button
            self.thumbs_up = QPushButton("👍")
            self.thumbs_up.setFixedSize(28, 28)
            self.thumbs_up.setCursor(Qt.PointingHandCursor)
            self.thumbs_up.setToolTip("Good response")
            self.thumbs_up.clicked.connect(lambda: self._submit_feedback(True))
            self.thumbs_up.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    border: 1px solid {COLORS['border']};
                    border-radius: 4px;
                    font-size: 14px;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['success']}30;
                    border-color: {COLORS['success']};
                }}
            """)
            feedback_layout.addWidget(self.thumbs_up)

            # Thumbs down button
            self.thumbs_down = QPushButton("👎")
            self.thumbs_down.setFixedSize(28, 28)
            self.thumbs_down.setCursor(Qt.PointingHandCursor)
            self.thumbs_down.setToolTip("Poor response")
            self.thumbs_down.clicked.connect(lambda: self._submit_feedback(False))
            self.thumbs_down.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    border: 1px solid {COLORS['border']};
                    border-radius: 4px;
                    font-size: 14px;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['error']}30;
                    border-color: {COLORS['error']};
                }}
            """)
            feedback_layout.addWidget(self.thumbs_down)

            # Status label (hidden initially)
            self.feedback_status = QLabel("")
            self.feedback_status.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
            self.feedback_status.hide()
            feedback_layout.addWidget(self.feedback_status)

            feedback_layout.addStretch()
            content_container.addWidget(self.feedback_frame)

        layout.addLayout(content_container)

        if not is_user:
            layout.addStretch()

    def _submit_feedback(self, is_positive: bool):
        """Handle feedback button click."""
        if self._feedback_submitted:
            return

        self._feedback_submitted = True

        # Update button styles to show selection
        if is_positive:
            self.thumbs_up.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['success']}40;
                    border: 2px solid {COLORS['success']};
                    border-radius: 4px;
                    font-size: 14px;
                }}
            """)
            self.feedback_status.setText("Thanks for the feedback!")
        else:
            self.thumbs_down.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['error']}40;
                    border: 2px solid {COLORS['error']};
                    border-radius: 4px;
                    font-size: 14px;
                }}
            """)
            self.feedback_status.setText("Thanks - we'll improve!")

        # Disable both buttons
        self.thumbs_up.setEnabled(False)
        self.thumbs_down.setEnabled(False)
        self.feedback_status.show()

        # Emit signal with message details
        self.feedback_given.emit(self.message_id, is_positive)


class ChatView(QWidget):
    """Chat interface for conversing with Nexus AI."""

    message_sent = Signal(str)
    clear_requested = Signal()
    # Signal: (message_id, model_name, request_id, is_positive)
    feedback_submitted = Signal(str, str, str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.messages = []
        self._message_counter = 0
        self._last_response_info = {}  # Store model info for feedback
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Header
        header = QHBoxLayout()
        title = QLabel("Chat")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()

        clear_btn = QPushButton("Clear Chat")
        clear_btn.setObjectName("secondary")
        clear_btn.clicked.connect(self._clear_chat)
        header.addWidget(clear_btn)

        layout.addLayout(header)

        # Chat area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)

        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setSpacing(12)
        self.chat_layout.setContentsMargins(16, 16, 16, 16)
        self.chat_layout.addStretch()

        self.scroll.setWidget(self.chat_container)
        layout.addWidget(self.scroll, 1)

        # Welcome message
        self._add_message("Hello! I'm Nexus, your AI assistant. How can I help you today?", is_user=False)

        # Input area
        input_frame = QFrame()
        input_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_tertiary']};
                border-radius: 8px;
                padding: 8px;
            }}
        """)
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(12, 8, 12, 8)
        input_layout.setSpacing(12)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message...")
        self.input_field.setStyleSheet(f"""
            QLineEdit {{
                background-color: {COLORS['bg_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 10px 14px;
                color: {COLORS['text_primary']};
                font-size: 14px;
            }}
            QLineEdit:focus {{
                border-color: {COLORS['accent_primary']};
            }}
        """)
        self.input_field.returnPressed.connect(self._send_message)
        input_layout.addWidget(self.input_field)

        send_btn = QPushButton("Send")
        send_btn.setObjectName("primary")
        send_btn.setFixedWidth(80)
        send_btn.clicked.connect(self._send_message)
        send_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_primary']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_secondary']};
            }}
        """)
        input_layout.addWidget(send_btn)

        layout.addWidget(input_frame)

    def _add_message(self, text: str, is_user: bool = True, model_name: str = None,
                     request_id: str = None):
        """Add a message to the chat."""
        # Generate unique message ID for AI responses
        message_id = None
        if not is_user:
            self._message_counter += 1
            message_id = f"msg-{self._message_counter}"

        # Insert before the stretch
        msg = ChatMessage(text, is_user, message_id=message_id,
                          model_name=model_name, request_id=request_id)

        # Connect feedback signal for AI messages
        if not is_user and message_id:
            msg.feedback_given.connect(
                lambda mid, positive: self._on_feedback(mid, model_name, request_id, positive)
            )

        self.chat_layout.insertWidget(self.chat_layout.count() - 1, msg)
        self.messages.append(msg)

        # Scroll to bottom
        QTimer.singleShot(100, self._scroll_to_bottom)

    def _on_feedback(self, message_id: str, model_name: str, request_id: str, is_positive: bool):
        """Handle feedback from a message."""
        self.feedback_submitted.emit(message_id, model_name or "unknown",
                                      request_id or "", is_positive)

    def _scroll_to_bottom(self):
        """Scroll chat to the bottom."""
        scrollbar = self.scroll.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _send_message(self):
        """Send the current message."""
        text = self.input_field.text().strip()
        if not text:
            return

        self._add_message(text, is_user=True)
        self.input_field.clear()
        self.message_sent.emit(text)

    def add_response(self, text: str, model_name: str = None, request_id: str = None):
        """Add an AI response to the chat.

        Args:
            text: The response text
            model_name: Name of the model that generated the response
            request_id: UUID of the request for feedback tracking
        """
        self._add_message(text, is_user=False, model_name=model_name, request_id=request_id)

    def _clear_chat(self):
        """Clear all chat messages."""
        for msg in self.messages:
            msg.deleteLater()
        self.messages.clear()

        # Re-add welcome message
        self._add_message("Chat cleared. How can I help you?", is_user=False)
        self.clear_requested.emit()

    def set_loading(self, loading: bool):
        """Show/hide loading indicator."""
        if loading:
            self._add_message("Thinking...", is_user=False)
        else:
            # Remove the "Thinking..." message if it exists
            if self.messages:
                try:
                    last_msg = self.messages[-1]
                    label = last_msg.findChild(QLabel)
                    if label and "Thinking..." in label.text():
                        last_msg.deleteLater()
                        self.messages.pop()
                except (RuntimeError, AttributeError):
                    # Widget was already deleted or invalid
                    pass

    def add_assistant_message(self, text: str, model_info: dict = None):
        """Add an AI assistant message with optional model info.

        Args:
            text: The response text
            model_info: Dict with 'model', 'request_id', etc.
        """
        # Remove loading indicator if present
        self.set_loading(False)

        model_name = None
        request_id = None
        if model_info:
            model_name = model_info.get("model")
            request_id = model_info.get("request_id")

        self._add_message(text, is_user=False, model_name=model_name, request_id=request_id)

    def add_error_message(self, error: str):
        """Add an error message to the chat."""
        # Remove loading indicator if present
        self.set_loading(False)

        error_text = f"Error: {error}"
        self._add_message(error_text, is_user=False)

    def get_conversation_history(self) -> list:
        """Get the conversation history for context.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        history = []
        for msg in self.messages:
            try:
                # Find the QLabel with the message text
                label = msg.findChild(QLabel)
                if not label:
                    continue

                text = label.text()
                # Skip system messages like "Thinking..." or "Chat cleared"
                if text in ("Thinking...", "Chat cleared. How can I help you?"):
                    continue
                # Skip welcome message
                if "Hello! I'm Nexus" in text:
                    continue
                # Skip error messages
                if text.startswith("Error:"):
                    continue

                # Determine role based on message type
                frame = msg.findChild(QFrame)
                is_user = frame and frame.objectName() == "userBubble"
                history.append({
                    "role": "user" if is_user else "assistant",
                    "content": text
                })
            except (RuntimeError, AttributeError):
                # Widget was deleted or invalid
                continue

        return history
