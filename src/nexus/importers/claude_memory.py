"""Migrate from Claude Memory MCP to Nexus Intelligence."""

import asyncio
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any


class ClaudeMemoryMigrator:
    """Migrate data from Claude Memory MCP."""

    def __init__(self, source_path: str, nexus_intel):
        self.source_path = Path(source_path)
        self.intel = nexus_intel
        self.stats = {"imported": 0, "skipped": 0, "errors": 0}

    async def migrate_all(self):
        """Run full migration."""
        print("Starting migration from Claude Memory MCP...")

        await self._migrate_conversations()
        await self._migrate_tasks()
        await self._migrate_decisions()
        await self._migrate_preferences()

        print(f"\nMigration complete!")
        print(f"  Imported: {self.stats['imported']}")
        print(f"  Skipped: {self.stats['skipped']}")
        print(f"  Errors: {self.stats['errors']}")

    async def _migrate_conversations(self):
        """Migrate conversation history."""
        conv_file = self.source_path / "conversations.json"
        if not conv_file.exists():
            print("No conversations.json found, skipping...")
            return

        with open(conv_file) as f:
            conversations = json.load(f)

        print(f"Migrating {len(conversations)} conversations...")

        from nexus.intelligence.models import Conversation, Message, MessageRole

        for conv_data in conversations:
            try:
                messages = []
                for msg in conv_data.get("messages", []):
                    messages.append(Message(
                        role=MessageRole(msg.get("role", "user")),
                        content=msg.get("content", ""),
                        timestamp=datetime.fromisoformat(msg["timestamp"]) if msg.get("timestamp") else None
                    ))

                conv = Conversation(
                    id=conv_data.get("id", str(hash(conv_data.get("title", "")))),
                    title=conv_data.get("title"),
                    messages=messages,
                    project_path=conv_data.get("project_path"),
                    created_at=datetime.fromisoformat(conv_data["created_at"]) if conv_data.get("created_at") else datetime.now(timezone.utc)
                )

                await self.intel.memory.index_conversation(conv)
                self.stats["imported"] += 1

            except Exception as e:
                print(f"  Error importing conversation: {e}")
                self.stats["errors"] += 1

    async def _migrate_tasks(self):
        """Migrate tasks."""
        tasks_file = self.source_path / "tasks.json"
        if not tasks_file.exists():
            print("No tasks.json found, skipping...")
            return

        with open(tasks_file) as f:
            tasks = json.load(f)

        print(f"Migrating {len(tasks)} tasks...")

        from nexus.intelligence.tasks import Task, TaskStatus, TaskPriority

        for task_data in tasks:
            try:
                task = Task(
                    id=task_data.get("id", Task.generate_id()),
                    title=task_data.get("title", "Untitled"),
                    description=task_data.get("description"),
                    status=TaskStatus(task_data.get("status", "pending")),
                    priority=TaskPriority(task_data.get("priority", "medium")),
                    project_path=task_data.get("project_path"),
                    tags=task_data.get("tags", [])
                )

                await self.intel.tasks.create_task(task)
                self.stats["imported"] += 1

            except Exception as e:
                print(f"  Error importing task: {e}")
                self.stats["errors"] += 1

    async def _migrate_decisions(self):
        """Migrate decisions."""
        decisions_file = self.source_path / "decisions.json"
        if not decisions_file.exists():
            print("No decisions.json found, skipping...")
            return

        with open(decisions_file) as f:
            decisions = json.load(f)

        print(f"Migrating {len(decisions)} decisions...")

        from nexus.intelligence.decisions import Decision

        for dec_data in decisions:
            try:
                dec = Decision(
                    id=dec_data.get("id", Decision.generate_id()),
                    question=dec_data.get("question", ""),
                    decision=dec_data.get("decision", ""),
                    reasoning=dec_data.get("reasoning", ""),
                    alternatives=dec_data.get("alternatives", []),
                    project_path=dec_data.get("project_path")
                )

                await self.intel.decisions.record_decision(dec)
                self.stats["imported"] += 1

            except Exception as e:
                print(f"  Error importing decision: {e}")
                self.stats["errors"] += 1

    async def _migrate_preferences(self):
        """Migrate user preferences."""
        prefs_file = self.source_path / "preferences.json"
        if not prefs_file.exists():
            print("No preferences.json found, skipping...")
            return

        with open(prefs_file) as f:
            preferences = json.load(f)

        print(f"Migrating {len(preferences)} preferences...")

        from nexus.intelligence.preferences import PreferenceCategory

        for pref_data in preferences:
            try:
                await self.intel.preferences.observe_preference(
                    category=PreferenceCategory(pref_data.get("category", "workflow")),
                    key=pref_data.get("key", "unknown"),
                    value=pref_data.get("value", "")
                )
                self.stats["imported"] += 1

            except Exception as e:
                print(f"  Error importing preference: {e}")
                self.stats["errors"] += 1
