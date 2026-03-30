"""Styling and theming for Nexus GUI."""

# Color palette - Dark theme inspired by modern IDEs
COLORS = {
    "bg_primary": "#1e1e2e",      # Main background
    "bg_secondary": "#282a36",    # Card/panel background
    "bg_tertiary": "#313244",     # Hover/selected background
    "bg_input": "#45475a",        # Input field background

    "text_primary": "#cdd6f4",    # Primary text
    "text_secondary": "#a6adc8",  # Secondary text
    "text_muted": "#6c7086",      # Muted/disabled text

    "accent_primary": "#89b4fa",  # Primary accent (blue)
    "accent_secondary": "#7ba3e8", # Secondary accent (darker blue)
    "accent_success": "#a6e3a1",  # Success (green)
    "accent_warning": "#fab387",  # Warning (orange)
    "accent_error": "#f38ba8",    # Error (red)
    "accent_info": "#89dceb",     # Info (cyan)
    "accent_purple": "#cba6f7",   # Purple accent

    "border": "#45475a",          # Border color
    "border_focus": "#89b4fa",    # Focus border

    # Shorthand aliases
    "success": "#a6e3a1",         # Alias for accent_success
    "error": "#f38ba8",           # Alias for accent_error
    "warning": "#fab387",         # Alias for accent_warning
    "info": "#89dceb",            # Alias for accent_info
}

# Main stylesheet
STYLESHEET = f"""
QMainWindow {{
    background-color: {COLORS['bg_primary']};
}}

QWidget {{
    background-color: {COLORS['bg_primary']};
    color: {COLORS['text_primary']};
    font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
    font-size: 13px;
}}

/* Sidebar Navigation */
QListWidget#sidebar {{
    background-color: {COLORS['bg_secondary']};
    border: none;
    border-right: 1px solid {COLORS['border']};
    padding: 8px;
}}

QListWidget#sidebar::item {{
    background-color: transparent;
    color: {COLORS['text_secondary']};
    padding: 12px 16px;
    border-radius: 8px;
    margin: 2px 4px;
}}

QListWidget#sidebar::item:hover {{
    background-color: {COLORS['bg_tertiary']};
    color: {COLORS['text_primary']};
}}

QListWidget#sidebar::item:selected {{
    background-color: {COLORS['accent_primary']};
    color: {COLORS['bg_primary']};
}}

/* Cards */
QFrame#card {{
    background-color: {COLORS['bg_secondary']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 16px;
}}

QFrame#card:hover {{
    border-color: {COLORS['accent_primary']};
}}

/* Buttons */
QPushButton {{
    background-color: {COLORS['accent_primary']};
    color: {COLORS['bg_primary']};
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
}}

QPushButton:hover {{
    background-color: #7ba3e8;
}}

QPushButton:pressed {{
    background-color: #6b93d8;
}}

QPushButton#secondary {{
    background-color: {COLORS['bg_tertiary']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
}}

QPushButton#secondary:hover {{
    background-color: {COLORS['bg_input']};
    border-color: {COLORS['accent_primary']};
}}

QPushButton#danger {{
    background-color: {COLORS['accent_error']};
}}

QPushButton#success {{
    background-color: {COLORS['accent_success']};
    color: {COLORS['bg_primary']};
}}

/* Input Fields */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {COLORS['bg_input']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 8px 12px;
}}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {COLORS['accent_primary']};
}}

QLineEdit::placeholder {{
    color: {COLORS['text_muted']};
}}

/* ComboBox */
QComboBox {{
    background-color: {COLORS['bg_input']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 8px 12px;
}}

QComboBox:hover {{
    border-color: {COLORS['accent_primary']};
}}

QComboBox::drop-down {{
    border: none;
    width: 30px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS['bg_secondary']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    selection-background-color: {COLORS['accent_primary']};
}}

/* Labels */
QLabel {{
    color: {COLORS['text_primary']};
    background: transparent;
}}

QLabel#title {{
    font-size: 24px;
    font-weight: 700;
    color: {COLORS['text_primary']};
}}

QLabel#subtitle {{
    font-size: 14px;
    color: {COLORS['text_secondary']};
}}

QLabel#stat_value {{
    font-size: 32px;
    font-weight: 700;
    color: {COLORS['accent_primary']};
}}

QLabel#stat_label {{
    font-size: 12px;
    color: {COLORS['text_muted']};
    text-transform: uppercase;
}}

/* Tables */
QTableWidget {{
    background-color: {COLORS['bg_secondary']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    gridline-color: {COLORS['border']};
}}

QTableWidget::item {{
    padding: 8px;
    border-bottom: 1px solid {COLORS['border']};
}}

QTableWidget::item:selected {{
    background-color: {COLORS['accent_primary']};
    color: {COLORS['bg_primary']};
}}

QHeaderView::section {{
    background-color: {COLORS['bg_tertiary']};
    color: {COLORS['text_secondary']};
    padding: 10px;
    border: none;
    border-bottom: 1px solid {COLORS['border']};
    font-weight: 600;
}}

/* ScrollBar */
QScrollBar:vertical {{
    background-color: {COLORS['bg_secondary']};
    width: 10px;
    border-radius: 5px;
}}

QScrollBar::handle:vertical {{
    background-color: {COLORS['bg_input']};
    border-radius: 5px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {COLORS['text_muted']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar:horizontal {{
    background-color: {COLORS['bg_secondary']};
    height: 10px;
    border-radius: 5px;
}}

QScrollBar::handle:horizontal {{
    background-color: {COLORS['bg_input']};
    border-radius: 5px;
    min-width: 30px;
}}

/* Tab Widget */
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

QTabBar::tab:hover {{
    color: {COLORS['text_primary']};
}}

/* Progress Bar */
QProgressBar {{
    background-color: {COLORS['bg_input']};
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {COLORS['accent_primary']};
    border-radius: 4px;
}}

/* Tooltips */
QToolTip {{
    background-color: {COLORS['bg_tertiary']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 6px 10px;
}}

/* Splitter */
QSplitter::handle {{
    background-color: {COLORS['border']};
}}

QSplitter::handle:horizontal {{
    width: 2px;
}}

QSplitter::handle:vertical {{
    height: 2px;
}}

/* GroupBox */
QGroupBox {{
    font-weight: 600;
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 12px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: {COLORS['text_secondary']};
}}
"""

# Status badge colors
STATUS_COLORS = {
    "pending": COLORS["text_muted"],
    "in_progress": COLORS["accent_primary"],
    "blocked": COLORS["accent_error"],
    "completed": COLORS["accent_success"],
    "cancelled": COLORS["text_muted"],
    "active": COLORS["accent_primary"],
    "paused": COLORS["accent_warning"],
    "abandoned": COLORS["text_muted"],
}

# Priority colors
PRIORITY_COLORS = {
    "critical": COLORS["accent_error"],
    "high": COLORS["accent_warning"],
    "medium": COLORS["accent_primary"],
    "low": COLORS["accent_info"],
    "backlog": COLORS["text_muted"],
}
