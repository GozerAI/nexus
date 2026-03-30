"""Main Nexus Intelligence GUI Application."""

import logging
import sys
import yaml
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QStackedWidget, QFrame, QMessageBox, QSizePolicy
)
from PySide6.QtCore import Qt, QSize, Signal, QTimer
from PySide6.QtGui import QIcon, QFont

from nexus.gui.styles import STYLESHEET, COLORS
from nexus.gui.async_bridge import AsyncBridge, IntelligenceController
from nexus.gui.views import (
    DashboardView, TasksView, GoalsView, MemoryView,
    KnowledgeView, DecisionsView, SettingsView, ChatView,
    ModelsView, MonitoringView, DiscoveryView, RAGView
)
from nexus.gui.views.coo import COOView
from nexus.services import OllamaManager, OllamaStatus, get_ollama_manager


logger = logging.getLogger(__name__)


class NavButton(QPushButton):
    """Navigation sidebar button."""

    def __init__(self, text: str, icon_char: str = "", parent=None):
        super().__init__(parent)
        self.setText(f"  {icon_char}  {text}" if icon_char else f"  {text}")
        self.setCheckable(True)
        self.setFixedHeight(44)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_secondary']};
                border: none;
                border-radius: 8px;
                text-align: left;
                padding-left: 12px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_tertiary']};
                color: {COLORS['text_primary']};
            }}
            QPushButton:checked {{
                background-color: {COLORS['accent_primary']};
                color: white;
            }}
        """)


class OllamaStatusIndicator(QFrame):
    """Visual indicator for Ollama service status."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(8)

        # Status dot
        self.status_dot = QLabel("●")
        self.status_dot.setFixedWidth(16)
        layout.addWidget(self.status_dot)

        # Status text
        self.status_label = QLabel("Ollama")
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        layout.addWidget(self.status_label)

        layout.addStretch()

        self.set_status(OllamaStatus.UNKNOWN)

    def set_status(self, status: OllamaStatus):
        """Update the status indicator."""
        colors = {
            OllamaStatus.RUNNING: ("#22c55e", "Running"),
            OllamaStatus.STARTING: ("#f59e0b", "Starting..."),
            OllamaStatus.STOPPED: ("#6b7280", "Stopped"),
            OllamaStatus.ERROR: ("#ef4444", "Error"),
            OllamaStatus.UNKNOWN: ("#6b7280", "Unknown"),
        }

        color, text = colors.get(status, ("#6b7280", "Unknown"))
        self.status_dot.setStyleSheet(f"color: {color}; font-size: 14px;")
        self.status_label.setText(f"Ollama: {text}")


class NexusApp(QMainWindow):
    """Main Nexus Intelligence application window."""

    # Signal to notify when initialization is complete
    initialization_complete = Signal(bool, str)  # success, message
    ollama_status_changed = Signal(object)  # OllamaStatus

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nexus Intelligence")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # Track initialization state
        self._backend_initialized = False
        self._initialization_in_progress = False

        # Initialize Ollama manager (auto-starts in background)
        self.ollama_manager = get_ollama_manager(
            auto_start=True,
            health_check_interval=30.0,
            preload_models=["llama3.1:8b"],  # Preload a default model
        )
        self.ollama_manager.on_status_change(self._on_ollama_status_change)

        # Initialize async bridge
        self.async_bridge = AsyncBridge()
        self.controller = IntelligenceController(self.async_bridge)

        # Setup UI
        self.setup_ui()
        self.connect_signals()

        # Connect initialization signal
        self.initialization_complete.connect(self._on_initialization_complete)
        self.ollama_status_changed.connect(self._update_ollama_indicator)

        # Start Ollama in background first
        self.async_bridge.run_async(
            self._start_ollama(),
            on_error=lambda e: logger.warning("Ollama start error: %s", e)
        )

        # Initialize backend
        self._initialization_in_progress = True
        self.async_bridge.run_async(
            self.initialize_backend(),
            on_error=self._on_initialization_error
        )

    async def _start_ollama(self):
        """Start Ollama service in background."""
        await self.ollama_manager.start()
        await self.ollama_manager.start_health_monitoring()

    def _on_ollama_status_change(self, status: OllamaStatus):
        """Handle Ollama status change (called from async context)."""
        # Emit signal to update UI on main thread
        self.ollama_status_changed.emit(status)

    def _update_ollama_indicator(self, status: OllamaStatus):
        """Update Ollama status indicator on main thread."""
        if hasattr(self, 'ollama_indicator'):
            self.ollama_indicator.set_status(status)

    def _on_initialization_complete(self, success: bool, message: str):
        """Handle initialization completion on the main thread."""
        self._initialization_in_progress = False
        self._backend_initialized = success

        if success:
            # Enable views that depend on the backend
            self._enable_backend_dependent_views()
        else:
            QMessageBox.critical(
                self,
                "Initialization Error",
                f"Failed to initialize Nexus Intelligence:\n{message}"
            )

    def _on_initialization_error(self, error: str):
        """Handle initialization error."""
        self._initialization_in_progress = False
        self._backend_initialized = False
        QMessageBox.critical(
            self,
            "Initialization Error",
            f"Failed to initialize Nexus Intelligence:\n{error}"
        )

    def _enable_backend_dependent_views(self):
        """Enable views that require the backend to be initialized."""
        # COO view can now safely access the controller
        if hasattr(self, 'coo_view'):
            self.coo_view._refresh_data()

    async def initialize_backend(self):
        """Initialize the Nexus Intelligence backend."""
        try:
            await self.controller.initialize()
            # Load initial data
            await self.refresh_all_data()
            # Emit success signal
            self.initialization_complete.emit(True, "Initialized successfully")
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            logger.warning("Backend initialization error: %s", error_msg)
            self.initialization_complete.emit(False, str(e))

    async def refresh_all_data(self):
        """Refresh all views with current data."""
        try:
            # Dashboard stats
            stats = await self.controller.get_dashboard_stats()
            self.dashboard_view.update_stats(
                stats.get("active_tasks", 0),
                stats.get("active_goals", 0),
                stats.get("total_memories", 0),
                stats.get("total_decisions", 0)
            )

            focus = await self.controller.get_focus_context()
            self.dashboard_view.update_focus(focus)

            # Tasks - update both tasks view AND dashboard
            tasks = await self.controller.get_tasks()
            self.tasks_view.update_tasks(tasks)
            # Filter to active tasks for dashboard display
            active_tasks = [t for t in tasks if t.status.value not in ("completed", "cancelled")]
            self.dashboard_view.update_tasks(active_tasks)

            # Goals - update both goals view AND dashboard
            goals = await self.controller.get_goals()
            self.goals_view.update_goals(goals)
            # Filter to active goals for dashboard display
            active_goals = [g for g in goals if g.status.value not in ("completed", "abandoned")]
            self.dashboard_view.update_goals(active_goals)

            # Recent decisions
            decisions = await self.controller.get_decisions()
            self.decisions_view.update_decisions(decisions)

            # Load settings
            config = self.load_config()
            self.settings_view.load_settings(config)

        except Exception as e:
            logger.warning("Error refreshing data: %s", e)

    def setup_ui(self):
        """Setup the main UI layout."""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        # Main horizontal layout
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        sidebar = self.create_sidebar()
        main_layout.addWidget(sidebar)

        # Content area
        content_frame = QFrame()
        content_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_primary']};
            }}
        """)
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # Stacked widget for views
        self.stack = QStackedWidget()

        # Create views
        self.dashboard_view = DashboardView()
        self.chat_view = ChatView()
        self.tasks_view = TasksView()
        self.goals_view = GoalsView()
        self.memory_view = MemoryView()
        self.knowledge_view = KnowledgeView()
        self.decisions_view = DecisionsView()
        self.models_view = ModelsView()
        self.monitoring_view = MonitoringView()
        self.discovery_view = DiscoveryView()
        self.rag_view = RAGView()
        self.coo_view = COOView(self.controller, self.async_bridge)
        self.settings_view = SettingsView()

        # Add views to stack (order must match navigation indices)
        self.stack.addWidget(self.dashboard_view)   # 0
        self.stack.addWidget(self.chat_view)        # 1
        self.stack.addWidget(self.tasks_view)       # 2
        self.stack.addWidget(self.goals_view)       # 3
        self.stack.addWidget(self.memory_view)      # 4
        self.stack.addWidget(self.knowledge_view)   # 5
        self.stack.addWidget(self.decisions_view)   # 6
        self.stack.addWidget(self.models_view)      # 7
        self.stack.addWidget(self.monitoring_view)  # 8
        self.stack.addWidget(self.discovery_view)   # 9
        self.stack.addWidget(self.rag_view)         # 10
        self.stack.addWidget(self.coo_view)         # 11
        self.stack.addWidget(self.settings_view)    # 12

        content_layout.addWidget(self.stack)
        main_layout.addWidget(content_frame, 1)

    def create_sidebar(self) -> QFrame:
        """Create the navigation sidebar."""
        sidebar = QFrame()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_secondary']};
                border-right: 1px solid {COLORS['border']};
            }}
        """)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 16, 12, 16)
        layout.setSpacing(4)

        # Logo/Title
        title = QLabel("Nexus")
        title.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['accent_primary']};
                font-size: 20px;
                font-weight: 700;
                padding: 8px 12px;
            }}
        """)
        layout.addWidget(title)

        subtitle = QLabel("Intelligence Platform")
        subtitle.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['text_muted']};
                font-size: 11px;
                padding: 0 12px 16px 12px;
            }}
        """)
        layout.addWidget(subtitle)

        # Navigation buttons
        self.nav_buttons = []

        nav_items = [
            ("Dashboard", "📊", 0),
            ("Chat", "💬", 1),
            ("Tasks", "✓", 2),
            ("Goals", "🎯", 3),
            ("Memory", "🧠", 4),
            ("Knowledge", "📚", 5),
            ("Decisions", "⚖", 6),
            ("Models", "🤖", 7),
            ("Monitoring", "📈", 8),
            ("Discovery", "🔍", 9),
            ("RAG", "📄", 10),
            ("COO", "👔", 11),
        ]

        for text, icon, index in nav_items:
            btn = NavButton(text, icon)
            btn.clicked.connect(lambda checked, i=index: self.switch_view(i))
            self.nav_buttons.append(btn)
            layout.addWidget(btn)

        layout.addStretch()

        # Ollama status indicator
        self.ollama_indicator = OllamaStatusIndicator()
        self.ollama_indicator.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_tertiary']};
                border-radius: 6px;
                margin: 4px 0;
            }}
        """)
        layout.addWidget(self.ollama_indicator)

        # Settings at bottom
        settings_btn = NavButton("Settings", "⚙")
        settings_btn.clicked.connect(lambda: self.switch_view(12))
        self.nav_buttons.append(settings_btn)
        layout.addWidget(settings_btn)

        # Version
        version = QLabel("v1.0.0")
        version.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['text_muted']};
                font-size: 10px;
                padding: 8px 12px;
            }}
        """)
        layout.addWidget(version)

        # Set initial selection
        self.nav_buttons[0].setChecked(True)

        return sidebar

    def switch_view(self, index: int):
        """Switch to the specified view."""
        self.stack.setCurrentIndex(index)

        # Update button states
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)

    def connect_signals(self):
        """Connect view signals to handlers."""
        # Dashboard
        self.dashboard_view.refresh_requested.connect(
            lambda: self.async_bridge.run_async(self.refresh_all_data())
        )
        # Navigate to Tasks tab when a task is selected on dashboard
        self.dashboard_view.task_selected.connect(
            lambda task_id: self.switch_view(2)  # Tasks tab is index 2
        )
        # Navigate to Goals tab when a goal is selected on dashboard
        self.dashboard_view.goal_selected.connect(
            lambda goal_id: self.switch_view(3)  # Goals tab is index 3
        )
        # Stat card navigation - clicking stats takes you to the relevant tab
        self.dashboard_view.navigate_to_tasks.connect(lambda: self.switch_view(2))
        self.dashboard_view.navigate_to_goals.connect(lambda: self.switch_view(3))
        self.dashboard_view.navigate_to_memory.connect(lambda: self.switch_view(4))
        self.dashboard_view.navigate_to_decisions.connect(lambda: self.switch_view(6))

        # Chat - model selection is automatic
        self.chat_view.message_sent.connect(
            lambda msg: self.async_bridge.run_async(self.send_chat_message(msg))
        )
        # Connect feedback signal
        self.chat_view.feedback_submitted.connect(
            lambda mid, model, req_id, positive: self.async_bridge.run_async(
                self.record_chat_feedback(mid, model, req_id, positive)
            )
        )

        # Tasks
        self.tasks_view.refresh_requested.connect(
            lambda: self.async_bridge.run_async(self.refresh_tasks())
        )
        self.tasks_view.create_task.connect(
            lambda data: self.async_bridge.run_async(self.create_task(data))
        )
        self.tasks_view.update_task.connect(
            lambda tid, data: self.async_bridge.run_async(self.update_task(tid, data))
        )

        # Goals
        self.goals_view.refresh_requested.connect(
            lambda: self.async_bridge.run_async(self.refresh_goals())
        )
        self.goals_view.create_goal.connect(
            lambda data: self.async_bridge.run_async(self.create_goal(data))
        )

        # Memory
        self.memory_view.search_requested.connect(
            lambda q, l: self.async_bridge.run_async(self.search_memory(q, l))
        )
        self.memory_view.load_topics.connect(
            lambda: self.async_bridge.run_async(self.load_topics())
        )
        self.memory_view.topic_history_requested.connect(
            lambda q: self.async_bridge.run_async(self.load_topic_history(q))
        )

        # Knowledge
        self.knowledge_view.search_entities.connect(
            lambda q, t: self.async_bridge.run_async(self.search_entities(q, t))
        )
        self.knowledge_view.search_facts.connect(
            lambda q, t: self.async_bridge.run_async(self.search_facts(q, t))
        )

        # Decisions
        self.decisions_view.refresh_requested.connect(
            lambda: self.async_bridge.run_async(self.refresh_decisions())
        )
        self.decisions_view.record_decision.connect(
            lambda data: self.async_bridge.run_async(self.record_decision(data))
        )
        self.decisions_view.search_decisions.connect(
            lambda q: self.async_bridge.run_async(self.search_decisions(q))
        )

        # Settings
        self.settings_view.save_settings.connect(self.save_settings)
        self.settings_view.reset_settings.connect(self.reset_settings)
        self.settings_view.test_connection.connect(
            lambda: self.async_bridge.run_async(self.test_ollama_connection())
        )

        # Models view
        self.models_view.refresh_requested.connect(
            lambda: self.async_bridge.run_async(self.refresh_models())
        )
        self.models_view.strategy_changed.connect(
            lambda s: self.async_bridge.run_async(self.change_ensemble_strategy(s))
        )
        self.models_view.model_quarantine_requested.connect(
            lambda m: self.async_bridge.run_async(self.quarantine_model(m))
        )
        self.models_view.model_release_requested.connect(
            lambda m: self.async_bridge.run_async(self.release_model(m))
        )

        # Monitoring view
        self.monitoring_view.refresh_requested.connect(
            lambda: self.async_bridge.run_async(self.refresh_monitoring())
        )
        self.monitoring_view.alert_dismissed.connect(
            lambda a: self.async_bridge.run_async(self.dismiss_alert(a))
        )
        self.monitoring_view.time_range_changed.connect(
            lambda r: self.async_bridge.run_async(self.refresh_monitoring_range(r))
        )

        # Discovery view
        self.discovery_view.search_requested.connect(
            lambda q, s: self.async_bridge.run_async(self.search_resources(q, s))
        )
        self.discovery_view.refresh_requested.connect(
            lambda: self.async_bridge.run_async(self.refresh_discovery())
        )
        self.discovery_view.scan_requested.connect(
            lambda s: self.async_bridge.run_async(self.scan_source(s))
        )
        self.discovery_view.resource_action.connect(
            lambda r, a: self.async_bridge.run_async(self.resource_action(r, a))
        )

        # RAG view
        self.rag_view.upload_requested.connect(
            lambda f: self.async_bridge.run_async(self.upload_documents(f))
        )
        self.rag_view.delete_document.connect(
            lambda d: self.async_bridge.run_async(self.delete_document(d))
        )
        self.rag_view.reindex_document.connect(
            lambda d: self.async_bridge.run_async(self.reindex_document(d))
        )
        self.rag_view.test_query.connect(
            lambda q, o: self.async_bridge.run_async(self.test_rag_query(q, o))
        )

    # Async handlers
    async def send_chat_message(self, message: str):
        """Send a chat message and display the response with intelligent model selection."""
        try:
            # Show loading indicator
            self.chat_view.set_loading(True)

            history = self.chat_view.get_conversation_history()
            result = await self.controller.chat(message, history)

            # Extract content and model info from result
            if isinstance(result, dict):
                content = result.get("content", str(result))
                model_info = result.get("model_info", None)
                # Include request_id in model_info if available
                if model_info is None:
                    model_info = {}
                if "request_id" in result and "request_id" not in model_info:
                    model_info["request_id"] = result.get("request_id")
            else:
                content = str(result)
                model_info = None

            self.chat_view.add_assistant_message(content, model_info)
        except Exception as e:
            self.chat_view.add_error_message(str(e))

    async def record_chat_feedback(self, message_id: str, model_name: str,
                                   request_id: str, is_positive: bool):
        """Record user feedback for a chat response."""
        try:
            await self.controller.record_feedback(
                message_id=message_id,
                model_name=model_name,
                request_id=request_id,
                is_positive=is_positive
            )
        except Exception as e:
            logger.warning("Error recording feedback: %s", e)

    async def refresh_tasks(self):
        tasks = await self.controller.get_tasks()
        self.tasks_view.update_tasks(tasks)

    async def create_task(self, data: dict):
        await self.controller.create_task(
            title=data.get("title", ""),
            description=data.get("description"),
            priority=data.get("priority", "medium"),
            project_path=data.get("project_path"),
            tags=data.get("tags")
        )
        await self.refresh_tasks()
        await self.refresh_all_data()

    async def update_task(self, task_id: str, data: dict):
        await self.controller.update_task(task_id, data)
        await self.refresh_tasks()

    async def refresh_goals(self):
        goals = await self.controller.get_goals()
        self.goals_view.update_goals(goals)

    async def create_goal(self, data: dict):
        await self.controller.create_goal(
            title=data.get("title", ""),
            description=data.get("description"),
            project_path=data.get("project_path")
        )
        await self.refresh_goals()
        await self.refresh_all_data()

    async def search_memory(self, query: str, limit: int):
        results = await self.controller.search_memory(query, limit)
        self.memory_view.show_results(results, query)

    async def load_topics(self):
        topics = await self.controller.get_topic_history()
        self.memory_view.update_topic_history(topics)

    async def load_topic_history(self, topic: str):
        mentions = await self.controller.get_topic_history(topic)
        self.memory_view.show_topic_history(mentions, topic)

    async def search_entities(self, query: str, entity_type: str = None):
        entities = await self.controller.search_entities(query, entity_type)
        self.knowledge_view.update_entities(entities)

    async def search_facts(self, query: str, topic: str = None):
        facts = await self.controller.search_facts(query, topic)
        self.knowledge_view.update_facts(facts)

    async def refresh_decisions(self):
        decisions = await self.controller.get_decisions()
        self.decisions_view.update_decisions(decisions)

    async def record_decision(self, data: dict):
        await self.controller.record_decision(
            question=data.get("question", ""),
            decision=data.get("decision", ""),
            reasoning=data.get("reasoning", ""),
            alternatives=data.get("alternatives"),
            project_path=data.get("project_path")
        )
        await self.refresh_decisions()

    async def search_decisions(self, query: str):
        decisions = await self.controller.search_decisions(query)
        self.decisions_view.update_decisions(decisions)

    async def test_ollama_connection(self):
        success, message = await self.controller.test_ollama()
        self.settings_view.show_connection_result(success, message)

    # Models view handlers
    async def refresh_models(self):
        """Refresh models view with current ensemble state."""
        try:
            data = await self.controller.get_ensemble_status()
            self.models_view.update_data(data)
        except Exception as e:
            logger.warning("Error refreshing models: %s", e)

    async def change_ensemble_strategy(self, strategy: str):
        """Change the ensemble selection strategy."""
        try:
            await self.controller.set_ensemble_strategy(strategy)
            await self.refresh_models()
        except Exception as e:
            logger.warning("Error changing strategy: %s", e)

    async def quarantine_model(self, model_id: str):
        """Quarantine a model from the ensemble."""
        try:
            await self.controller.quarantine_model(model_id)
            await self.refresh_models()
        except Exception as e:
            logger.warning("Error quarantining model: %s", e)

    async def release_model(self, model_id: str):
        """Release a quarantined model back to the ensemble."""
        try:
            await self.controller.release_model(model_id)
            await self.refresh_models()
        except Exception as e:
            logger.warning("Error releasing model: %s", e)

    # Monitoring view handlers
    async def refresh_monitoring(self):
        """Refresh monitoring view with current metrics."""
        try:
            data = await self.controller.get_monitoring_data()
            self.monitoring_view.update_data(data)
        except Exception as e:
            logger.warning("Error refreshing monitoring: %s", e)

    async def dismiss_alert(self, alert_id: str):
        """Dismiss a monitoring alert."""
        try:
            await self.controller.dismiss_alert(alert_id)
            await self.refresh_monitoring()
        except Exception as e:
            logger.warning("Error dismissing alert: %s", e)

    async def refresh_monitoring_range(self, time_range: str):
        """Refresh monitoring with a specific time range."""
        try:
            data = await self.controller.get_monitoring_data(time_range=time_range)
            self.monitoring_view.update_data(data)
        except Exception as e:
            logger.warning("Error refreshing monitoring: %s", e)

    # Discovery view handlers
    async def search_resources(self, query: str, source: str):
        """Search for resources across discovery sources."""
        try:
            results = await self.controller.search_resources(query, source)
            self.discovery_view.update_results(results)
        except Exception as e:
            logger.warning("Error searching resources: %s", e)

    async def refresh_discovery(self):
        """Refresh discovery view with current resources."""
        try:
            data = await self.controller.get_discovery_data()
            self.discovery_view.update_data(data)
        except Exception as e:
            logger.warning("Error refreshing discovery: %s", e)

    async def scan_source(self, source: str):
        """Scan a specific source for new resources."""
        try:
            await self.controller.scan_source(source)
            await self.refresh_discovery()
        except Exception as e:
            logger.warning("Error scanning source: %s", e)

    async def resource_action(self, resource_id: str, action: str):
        """Perform an action on a discovered resource."""
        try:
            await self.controller.resource_action(resource_id, action)
            await self.refresh_discovery()
        except Exception as e:
            logger.warning("Error performing resource action: %s", e)

    # RAG view handlers
    async def upload_documents(self, file_paths: list):
        """Upload documents to the RAG system."""
        try:
            await self.controller.upload_documents(file_paths)
            data = await self.controller.get_rag_data()
            self.rag_view.update_data(data)
        except Exception as e:
            logger.warning("Error uploading documents: %s", e)

    async def delete_document(self, doc_id: str):
        """Delete a document from the RAG system."""
        try:
            await self.controller.delete_document(doc_id)
            data = await self.controller.get_rag_data()
            self.rag_view.update_data(data)
        except Exception as e:
            logger.warning("Error deleting document: %s", e)

    async def reindex_document(self, doc_id: str):
        """Reindex a document in the RAG system."""
        try:
            await self.controller.reindex_document(doc_id)
            data = await self.controller.get_rag_data()
            self.rag_view.update_data(data)
        except Exception as e:
            logger.warning("Error reindexing document: %s", e)

    async def test_rag_query(self, query: str, options: dict):
        """Test a RAG query and display results."""
        try:
            results = await self.controller.test_rag_query(query, options)
            self.rag_view.show_query_results(results)
        except Exception as e:
            logger.warning("Error testing RAG query: %s", e)
            self.rag_view.show_query_results({"error": str(e)})

    def load_config(self) -> dict:
        """Load configuration from YAML file."""
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "nexus_intelligence.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def save_settings(self, settings: dict):
        """Save settings to config file."""
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "nexus_intelligence.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            yaml.dump(settings, f, default_flow_style=False)

        QMessageBox.information(
            self,
            "Settings Saved",
            "Settings have been saved successfully.\nSome changes may require a restart."
        )

    def reset_settings(self):
        """Reset settings to defaults."""
        default_config = {
            "storage": {
                "chroma_path": "data/chroma",
                "sqlite_path": "data/sqlite/nexus.db",
                "backup_path": "data/backups",
                "auto_backup": True,
                "backup_interval_hours": 24
            },
            "embedding": {
                "ollama_url": "http://localhost:11434",
                "primary_model": "nomic-embed-text",
                "fallback_model": "mxbai-embed-large",
                "batch_size": 32,
                "timeout_seconds": 30.0,
                "cache_enabled": True
            },
            "memory": {
                "max_chunk_size": 1500,
                "chunk_overlap": 200,
                "min_chunk_size": 100,
                "include_context_messages": 2
            },
            "truth": {
                "high_confidence_threshold": 0.7,
                "medium_confidence_threshold": 0.4,
                "strict_mode": False
            },
            "logging": {
                "level": "INFO",
                "file": "logs/nexus.log",
                "max_size_mb": 10,
                "backup_count": 5
            }
        }
        self.settings_view.load_settings(default_config)

    def closeEvent(self, event):
        """Handle application close."""
        self.async_bridge.run_async(self.shutdown())
        self.async_bridge.stop()
        event.accept()

    async def shutdown(self):
        """Shutdown the backend and services."""
        try:
            # Stop Ollama health monitoring (but leave Ollama running)
            await self.ollama_manager.stop_health_monitoring()
            # Note: We don't stop Ollama itself - it stays running in background
            # for other applications or future GUI sessions

            await self.controller.shutdown()
        except Exception as e:
            logger.warning("Error during shutdown: %s", e)


def run():
    """Run the Nexus Intelligence GUI application."""
    import os

    # Fix potential Qt rendering issues that can cause segmentation faults
    # These settings use software rendering which is more stable across different systems
    if 'QT_QUICK_BACKEND' not in os.environ:
        os.environ['QT_QUICK_BACKEND'] = 'software'
    if 'QSG_RENDER_LOOP' not in os.environ:
        os.environ['QSG_RENDER_LOOP'] = 'basic'

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)

    # Set application font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = NexusApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    run()
