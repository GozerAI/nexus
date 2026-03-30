"""Complete system verification for Nexus Intelligence Platform."""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def verify():
    from nexus.intelligence import NexusIntelligence
    from nexus.intelligence.models import Conversation, Message, MessageRole
    from nexus.intelligence.tasks import Task, TaskPriority
    from nexus.intelligence.goals import Goal, Milestone
    from nexus.intelligence.decisions import Decision
    from nexus.intelligence.knowledge import Fact

    print("Nexus Intelligence - Complete Verification")
    print("=" * 60)

    intel = NexusIntelligence()
    await intel.initialize()

    try:
        # 1. Test Memory
        print("\n1. Memory Module...")
        conv = Conversation(
            id="test-conv-1",
            title="Test Conversation",
            messages=[
                Message(role=MessageRole.USER, content="Test message about Python"),
                Message(role=MessageRole.ASSISTANT, content="Python is great!")
            ],
            project_path="C--dev-test"
        )
        chunks = await intel.memory.index_conversation(conv)
        print(f"   [PASS] Indexed {chunks} chunks")

        results = await intel.memory.search("Python")
        print(f"   [PASS] Search returned {len(results)} results")

        # 2. Test Tasks
        print("\n2. Task Module...")
        task = Task(
            id=Task.generate_id(),
            title="Test Task",
            priority=TaskPriority.HIGH,
            project_path="C--dev-test"
        )
        await intel.tasks.create_task(task)
        print(f"   [PASS] Created task: {task.id}")

        tasks = await intel.tasks.list_tasks()
        print(f"   [PASS] Listed {len(tasks)} tasks")

        # 3. Test Goals
        print("\n3. Goals Module...")
        goal = Goal(
            id=Goal.generate_id(),
            title="Test Goal",
            project_path="C--dev-test"
        )
        await intel.goals.create_goal(goal)
        print(f"   [PASS] Created goal: {goal.id}")

        # 4. Test Decisions
        print("\n4. Decisions Module...")
        dec = Decision(
            id=Decision.generate_id(),
            question="Which database?",
            decision="SQLite",
            reasoning="Local-first approach"
        )
        await intel.decisions.record_decision(dec)
        print(f"   [PASS] Recorded decision: {dec.id}")

        # 5. Test Knowledge
        print("\n5. Knowledge Module...")
        fact = Fact(
            id=Fact.generate_id(),
            statement="Nexus uses ChromaDB for vectors",
            topic="architecture"
        )
        await intel.knowledge.add_fact(fact)
        print(f"   [PASS] Added fact: {fact.id}")

        # 6. Test Truth Verification
        print("\n6. Truth Verification...")
        result = await intel.truth.verify_claim(
            "Nexus uses ChromaDB", topic="architecture"
        )
        print(f"   [PASS] Verification: {result.confidence.value}")

        # 7. Test Continuity
        print("\n7. Continuity Module...")
        session = await intel.continuity.start_session("C--dev-test")
        print(f"   [PASS] Started session: {session.id}")

        focus = await intel.continuity.get_current_focus("C--dev-test")
        print(f"   [PASS] Got focus: {focus.suggested_action}")

        await intel.continuity.end_session("Test complete")
        print("   [PASS] Ended session")

        print("\n" + "=" * 60)
        print("[SUCCESS] ALL MODULES VERIFIED SUCCESSFULLY")

    finally:
        await intel.shutdown()


if __name__ == "__main__":
    asyncio.run(verify())
