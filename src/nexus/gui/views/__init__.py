"""View components for Nexus GUI."""

from nexus.gui.views.dashboard import DashboardView
from nexus.gui.views.tasks import TasksView
from nexus.gui.views.goals import GoalsView
from nexus.gui.views.memory import MemoryView
from nexus.gui.views.knowledge import KnowledgeView
from nexus.gui.views.decisions import DecisionsView
from nexus.gui.views.settings import SettingsView
from nexus.gui.views.chat import ChatView
from nexus.gui.views.models import ModelsView
from nexus.gui.views.monitoring import MonitoringView
from nexus.gui.views.discovery import DiscoveryView
from nexus.gui.views.rag import RAGView

__all__ = [
    "DashboardView", "TasksView", "GoalsView",
    "MemoryView", "KnowledgeView", "DecisionsView", "SettingsView", "ChatView",
    "ModelsView", "MonitoringView", "DiscoveryView", "RAGView"
]
