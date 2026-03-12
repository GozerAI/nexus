"""MCP Server for Nexus Intelligence Platform."""

import asyncio
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: callable


class NexusMCPServer:
    """MCP Server exposing Nexus Intelligence features."""

    def __init__(self, intelligence):
        self.intel = intelligence
        self.server = Server("nexus-intelligence")
        self._register_tools()
        self._register_resources()

    def _register_tools(self):
        """Register all MCP tools."""

        # =====================================================================
        # Memory Tools
        # =====================================================================

        @self.server.tool()
        async def memory_search(query: str, n_results: int = 5, project_filter: str = None) -> str:
            """Search memories using semantic search."""
            results = await self.intel.memory.search(query, n_results, project_filter)
            return json.dumps([{
                "text": r.snippet, "relevance": r.relevance,
                "conversation_id": r.conversation_id
            } for r in results])

        @self.server.tool()
        async def memory_recall(topic: str, context_type: str = "any") -> str:
            """Recall specific context about a topic."""
            result = await self.intel.memory.recall(topic, context_type)
            return json.dumps(result)

        @self.server.tool()
        async def memory_stats() -> str:
            """Get memory statistics."""
            stats = await self.intel.memory.get_stats()
            return json.dumps(stats)

        @self.server.tool()
        async def topic_history(topic: str, limit: int = 10) -> str:
            """Get chronological history of a topic."""
            results = await self.intel.memory.topic_history(topic, limit)
            return json.dumps([{
                "text": r.chunk_text, "timestamp": r.timestamp.isoformat()
            } for r in results])

        # =====================================================================
        # Task Tools
        # =====================================================================

        @self.server.tool()
        async def create_task(title: str, description: str = None,
                             priority: str = "medium", project_path: str = None,
                             tags: List[str] = None) -> str:
            """Create a new task."""
            from nexus.intelligence.tasks import Task, TaskPriority
            task = Task(
                id=Task.generate_id(), title=title, description=description,
                priority=TaskPriority(priority), project_path=project_path,
                tags=tags or []
            )
            task_id = await self.intel.tasks.create_task(task)
            return json.dumps({"task_id": task_id, "title": title})

        @self.server.tool()
        async def update_task(task_id: str, status: str = None,
                             priority: str = None, add_note: str = None) -> str:
            """Update a task."""
            from nexus.intelligence.tasks import TaskStatus, TaskPriority
            updates = {}
            if status:
                updates["status"] = TaskStatus(status)
            if priority:
                updates["priority"] = TaskPriority(priority)
            if updates:
                await self.intel.tasks.update_task(task_id, updates)
            if add_note:
                await self.intel.tasks.add_note(task_id, add_note)
            return json.dumps({"updated": True, "task_id": task_id})

        @self.server.tool()
        async def list_tasks(project_path: str = None, status: str = None,
                            priority: str = None, include_completed: bool = False) -> str:
            """List tasks with filters."""
            from nexus.intelligence.tasks import TaskStatus, TaskPriority
            tasks = await self.intel.tasks.list_tasks(
                project_path=project_path,
                status=TaskStatus(status) if status else None,
                priority=TaskPriority(priority) if priority else None,
                include_completed=include_completed
            )
            return json.dumps([{
                "id": t.id, "title": t.title,
                "status": t.status.value, "priority": t.priority.value
            } for t in tasks])

        @self.server.tool()
        async def start_task(task_id: str) -> str:
            """Start working on a task."""
            task = await self.intel.tasks.start_task(task_id)
            return json.dumps({"started": True, "task_id": task_id, "title": task.title})

        @self.server.tool()
        async def complete_task(task_id: str, note: str = None) -> str:
            """Mark task as completed."""
            task = await self.intel.tasks.complete_task(task_id, note)
            return json.dumps({"completed": True, "task_id": task_id, "title": task.title})

        @self.server.tool()
        async def block_task(task_id: str, blocker_type: str, description: str) -> str:
            """Block a task with a reason."""
            from nexus.intelligence.tasks import BlockerType
            blocker = await self.intel.tasks.block_task(
                task_id, BlockerType(blocker_type), description
            )
            return json.dumps({"blocked": True, "blocker_id": blocker.id})

        @self.server.tool()
        async def resolve_blocker(blocker_id: str, resolution: str,
                                 resume_task: bool = True) -> str:
            """Resolve a blocker."""
            blocker = await self.intel.tasks.resolve_blocker(
                blocker_id, resolution, resume_task
            )
            return json.dumps({"resolved": True, "blocker_id": blocker_id})

        # =====================================================================
        # Goal Tools
        # =====================================================================

        @self.server.tool()
        async def set_project_goal(project_path: str, title: str,
                                  description: str = None) -> str:
            """Set a goal for a project."""
            from nexus.intelligence.goals import Goal
            goal = Goal(
                id=Goal.generate_id(), title=title,
                description=description, project_path=project_path
            )
            goal_id = await self.intel.goals.create_goal(goal)
            return json.dumps({"goal_id": goal_id, "title": title})

        @self.server.tool()
        async def get_project_goals(project_path: str,
                                   include_completed: bool = False) -> str:
            """Get goals for a project."""
            goals = await self.intel.goals.list_goals(
                project_path, include_completed=include_completed
            )
            return json.dumps([{
                "id": g.id, "title": g.title, "progress": g.progress
            } for g in goals])

        @self.server.tool()
        async def add_milestone(goal_id: str, title: str,
                               description: str = None) -> str:
            """Add milestone to a goal."""
            from nexus.intelligence.goals import Milestone
            milestone = Milestone(
                id=Milestone.generate_id(), goal_id=goal_id,
                title=title, description=description
            )
            ms_id = await self.intel.goals.add_milestone(goal_id, milestone)
            return json.dumps({"milestone_id": ms_id, "title": title})

        @self.server.tool()
        async def complete_milestone(milestone_id: str) -> str:
            """Complete a milestone."""
            milestone = await self.intel.goals.complete_milestone(milestone_id)
            return json.dumps({"completed": True, "milestone_id": milestone_id})

        # =====================================================================
        # Decision Tools
        # =====================================================================

        @self.server.tool()
        async def record_decision(question: str, decision: str, reasoning: str,
                                 alternatives: List[str] = None,
                                 project_path: str = None, task_id: str = None) -> str:
            """Record a decision with reasoning."""
            from nexus.intelligence.decisions import Decision
            dec = Decision(
                id=Decision.generate_id(), question=question,
                decision=decision, reasoning=reasoning,
                alternatives=alternatives or [],
                project_path=project_path, task_id=task_id
            )
            dec_id = await self.intel.decisions.record_decision(dec)
            return json.dumps({"decision_id": dec_id})

        @self.server.tool()
        async def list_decisions(project_path: str = None,
                                search_term: str = None, limit: int = 20) -> str:
            """List recorded decisions."""
            decisions = await self.intel.decisions.list_decisions(
                project_path, search_term=search_term, limit=limit
            )
            return json.dumps([{
                "id": d.id, "question": d.question, "decision": d.decision
            } for d in decisions])

        # =====================================================================
        # Correction Tools
        # =====================================================================

        @self.server.tool()
        async def record_correction(original: str, corrected: str,
                                   reason: str, topic: str = None) -> str:
            """Record a correction to prior information."""
            from nexus.intelligence.corrections import Correction
            corr = Correction(
                id=Correction.generate_id(), original=original,
                corrected=corrected, reason=reason, topic=topic
            )
            corr_id = await self.intel.corrections.record_correction(corr)
            return json.dumps({"correction_id": corr_id})

        @self.server.tool()
        async def knowledge_corrections(search_term: str = None,
                                       limit: int = 20) -> str:
            """Get history of corrections."""
            corrections = await self.intel.corrections.get_corrections(
                search_term=search_term, limit=limit
            )
            return json.dumps([{
                "original": c.original, "corrected": c.corrected, "reason": c.reason
            } for c in corrections])

        # =====================================================================
        # Truth Verification Tools (Nexus-specific)
        # =====================================================================

        @self.server.tool()
        async def verify_claim(claim: str, topic: str = None,
                              strict: bool = False) -> str:
            """Verify a claim against historical knowledge."""
            result = await self.intel.truth.verify_claim(claim, topic, strict)
            return json.dumps({
                "confidence": result.confidence.value,
                "score": result.confidence_score,
                "supporting_count": len(result.supporting_evidence),
                "contradictions_count": len(result.contradictions),
                "recommendation": result.recommendation
            })

        @self.server.tool()
        async def detect_contradictions(topic: str = None, limit: int = 20) -> str:
            """Scan for contradicting statements."""
            contradictions = await self.intel.truth.detect_contradictions(topic, limit)
            return json.dumps([{
                "a": c.statement_a, "b": c.statement_b,
                "type": c.contradiction_type
            } for c in contradictions])

        @self.server.tool()
        async def check_before_respond(proposed_response: str,
                                       topic: str = None) -> str:
            """Pre-flight check for response consistency."""
            result = await self.intel.truth.check_before_respond(
                proposed_response, topic
            )
            return json.dumps(result)

        # =====================================================================
        # Preference Tools
        # =====================================================================

        @self.server.tool()
        async def user_preferences(category: str = None) -> str:
            """Get learned user preferences."""
            from nexus.intelligence.preferences import PreferenceCategory
            prefs = await self.intel.preferences.get_preferences(
                PreferenceCategory(category) if category else None
            )
            return json.dumps([{
                "key": p.key, "value": p.value, "confidence": p.confidence
            } for p in prefs])

        @self.server.tool()
        async def preference_recall(topic: str) -> str:
            """Recall a specific preference."""
            pref = await self.intel.preferences.recall_preference(topic)
            if pref:
                return json.dumps({
                    "key": pref.key, "value": pref.value, "confidence": pref.confidence
                })
            return json.dumps({"found": False})

        @self.server.tool()
        async def preference_summary() -> str:
            """Get preference summary."""
            summary = await self.intel.preferences.get_preference_summary()
            return json.dumps(summary)

        # =====================================================================
        # Continuity Tools
        # =====================================================================

        @self.server.tool()
        async def get_current_focus(project_path: str = None) -> str:
            """Get current focus context - THE KEY TOOL for continuity."""
            focus = await self.intel.continuity.get_current_focus(project_path)
            return json.dumps({
                "active_tasks": [{"id": t.id, "title": t.title} for t in focus.active_tasks],
                "blocked_tasks": [{"id": t.id, "title": t.title} for t in focus.blocked_tasks],
                "blockers": focus.blockers_summary,
                "suggested_action": focus.suggested_action,
                "last_session_notes": focus.last_session_notes
            })

        @self.server.tool()
        async def suggest_next_action(project_path: str = None) -> str:
            """Get AI-suggested next action."""
            focus = await self.intel.continuity.get_current_focus(project_path)
            return json.dumps({"suggestion": focus.suggested_action})

        @self.server.tool()
        async def generate_handoff(project_path: str = None) -> str:
            """Generate handoff summary for session transition."""
            handoff = await self.intel.continuity.generate_handoff(project_path)
            return json.dumps({
                "summary": handoff.summary,
                "in_progress": handoff.in_progress_tasks,
                "blockers": handoff.blockers,
                "next_steps": handoff.suggested_next_steps
            })

        @self.server.tool()
        async def start_work_session(project_path: str = None,
                                    focus_task_id: str = None) -> str:
            """Start a new work session."""
            session = await self.intel.continuity.start_session(
                project_path, focus_task_id
            )
            return json.dumps({"session_id": session.id})

        @self.server.tool()
        async def end_work_session(summary: str = None, handoff_notes: str = None,
                                  insights: List[str] = None,
                                  open_questions: List[str] = None) -> str:
            """End current work session with summary."""
            session = await self.intel.continuity.end_session(
                summary, handoff_notes, insights, open_questions
            )
            return json.dumps({"session_id": session.id, "ended": True})

        @self.server.tool()
        async def get_continuation_prompt(project_path: str = None,
                                         token_budget: int = 1500) -> str:
            """Get context prompt for continuing work."""
            prompt = await self.intel.continuity.get_continuation_prompt(
                project_path, token_budget
            )
            return prompt

        # =====================================================================
        # Knowledge Graph Tools
        # =====================================================================

        @self.server.tool()
        async def entity_search(query: str = None, entity_type: str = None,
                               limit: int = 20) -> str:
            """Search for entities."""
            from nexus.intelligence.knowledge import EntityType
            entities = await self.intel.knowledge.search_entities(
                query or "",
                EntityType(entity_type) if entity_type else None,
                limit
            )
            return json.dumps([{
                "id": e.id, "name": e.name, "type": e.entity_type.value
            } for e in entities])

        @self.server.tool()
        async def entity_context(entity_name: str) -> str:
            """Get detailed context for an entity."""
            entity = await self.intel.knowledge.find_entity(entity_name)
            if not entity:
                return json.dumps({"found": False})

            relations = await self.intel.knowledge.get_relations(entity.id)
            return json.dumps({
                "entity": {"id": entity.id, "name": entity.name, "type": entity.entity_type.value},
                "relations": [{"type": r[0].relation_type.value, "target": r[1].name} for r in relations]
            })

        @self.server.tool()
        async def add_fact(statement: str, topic: str = None,
                         confidence: float = 0.5) -> str:
            """Add a fact to knowledge base."""
            from nexus.intelligence.knowledge import Fact
            fact = Fact(
                id=Fact.generate_id(), statement=statement,
                topic=topic, confidence=confidence
            )
            fact_id = await self.intel.knowledge.add_fact(fact)
            return json.dumps({"fact_id": fact_id})

    def _register_resources(self):
        """Register MCP resources."""
        pass  # Resources for exposing knowledge items

    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream)
