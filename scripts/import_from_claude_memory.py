"""
Import goals, tasks, and decisions from claude-memory database into Nexus.

This script extracts entities from the claude-memory MCP database and
imports them into the Nexus intelligence database.
"""

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
import json


def get_claude_memory_db():
    """Get path to claude-memory database."""
    home = Path.home()
    db_path = home / ".claude-memory" / "chroma" / "memory.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Claude memory database not found at {db_path}")
    return db_path


def get_nexus_db():
    """Get path to Nexus database."""
    db_path = Path("J:/dev/nexus/data/sqlite/nexus.db")
    if not db_path.exists():
        raise FileNotFoundError(f"Nexus database not found at {db_path}")
    return db_path


def extract_entities_from_claude_memory():
    """Extract task and decision entities from claude-memory."""
    db_path = get_claude_memory_db()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Get all tasks
    tasks = conn.execute("""
        SELECT DISTINCT name, context, entity_type, timestamp, confidence
        FROM entities
        WHERE entity_type = 'task'
        AND confidence >= 0.6
        ORDER BY timestamp DESC
    """).fetchall()

    # Get all decisions
    decisions = conn.execute("""
        SELECT DISTINCT name, context, entity_type, timestamp, confidence
        FROM entities
        WHERE entity_type = 'decision'
        AND confidence >= 0.6
        ORDER BY timestamp DESC
    """).fetchall()

    # Get summaries with topics and decisions
    summaries = conn.execute("""
        SELECT session_id, project_path, title, summary, key_topics, decisions_made, start_time
        FROM summaries
        WHERE title IS NOT NULL
    """).fetchall()

    conn.close()

    return {
        'tasks': [dict(t) for t in tasks],
        'decisions': [dict(d) for d in decisions],
        'summaries': [dict(s) for s in summaries]
    }


def deduplicate_tasks(tasks):
    """Remove duplicate tasks based on name similarity."""
    seen = set()
    unique_tasks = []

    for task in tasks:
        # Normalize the task name for comparison
        normalized = task['name'].lower().strip()
        # Skip very short tasks (likely noise)
        if len(normalized) < 10:
            continue
        # Skip if too similar to already seen
        if normalized not in seen:
            seen.add(normalized)
            unique_tasks.append(task)

    return unique_tasks


def import_to_nexus(data):
    """Import extracted data into Nexus database."""
    db_path = get_nexus_db()
    conn = sqlite3.connect(str(db_path))

    # Track counts
    counts = {'tasks': 0, 'decisions': 0, 'goals': 0}

    # Import tasks
    tasks = deduplicate_tasks(data['tasks'])
    print(f"Processing {len(tasks)} deduplicated tasks...")

    for task in tasks:
        try:
            task_id = str(uuid.uuid4())
            # Clean up task title
            title = task['name'][:200] if task['name'] else "Unknown Task"
            description = task['context'][:1000] if task['context'] else ""
            timestamp = task['timestamp'] or datetime.now().isoformat()

            conn.execute("""
                INSERT OR IGNORE INTO tasks (id, title, description, status, priority, project_path, created_at, updated_at)
                VALUES (?, ?, ?, 'pending', 'medium', 'claude-memory-import', ?, ?)
            """, (task_id, title, description, timestamp, timestamp))
            counts['tasks'] += 1
        except Exception as e:
            print(f"Error importing task: {e}")

    # Import decisions
    print(f"Processing {len(data['decisions'])} decisions...")
    for decision in data['decisions']:
        try:
            decision_id = str(uuid.uuid4())
            question = f"Decision about: {decision['name'][:200]}"
            decision_text = decision['name'][:500] if decision['name'] else "Unknown Decision"
            reasoning = decision['context'][:1000] if decision['context'] else ""
            timestamp = decision['timestamp'] or datetime.now().isoformat()

            conn.execute("""
                INSERT OR IGNORE INTO decisions (id, question, decision, reasoning, project_path, created_at)
                VALUES (?, ?, ?, ?, 'claude-memory-import', ?)
            """, (decision_id, question, decision_text, reasoning, timestamp))
            counts['decisions'] += 1
        except Exception as e:
            print(f"Error importing decision: {e}")

    # Extract goals from summaries
    print(f"Processing {len(data['summaries'])} summaries for goals...")
    for summary in data['summaries']:
        try:
            if summary['title']:
                goal_id = str(uuid.uuid4())
                title = summary['title'][:200]
                description = summary['summary'][:1000] if summary['summary'] else ""
                project = summary['project_path'] or 'unknown'
                timestamp = summary['start_time'] or datetime.now().isoformat()

                # Try to parse key_topics as JSON
                topics = []
                if summary['key_topics']:
                    try:
                        topics = json.loads(summary['key_topics'])
                    except:
                        pass

                if topics:
                    description += f"\n\nKey topics: {', '.join(topics[:5])}"

                conn.execute("""
                    INSERT OR IGNORE INTO goals (id, title, description, project_path, status, created_at)
                    VALUES (?, ?, ?, ?, 'active', ?)
                """, (goal_id, title, description, project, timestamp))
                counts['goals'] += 1
        except Exception as e:
            print(f"Error importing goal from summary: {e}")

    conn.commit()
    conn.close()

    return counts


def main():
    print("=" * 60)
    print("Importing data from claude-memory to Nexus")
    print("=" * 60)

    print("\n1. Extracting entities from claude-memory database...")
    try:
        data = extract_entities_from_claude_memory()
        print(f"   Found: {len(data['tasks'])} tasks, {len(data['decisions'])} decisions, {len(data['summaries'])} summaries")
    except FileNotFoundError as e:
        print(f"   ERROR: {e}")
        return

    print("\n2. Importing to Nexus database...")
    try:
        counts = import_to_nexus(data)
        print(f"   Imported: {counts['tasks']} tasks, {counts['decisions']} decisions, {counts['goals']} goals")
    except FileNotFoundError as e:
        print(f"   ERROR: {e}")
        return

    print("\n" + "=" * 60)
    print("Import complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
