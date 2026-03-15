"""Regression tests for stateful GUI discovery and RAG actions."""

import asyncio

from nexus.gui.async_bridge import IntelligenceController


def test_scan_source_and_resource_action_update_discovery_state():
    async def run_actions():
        controller = IntelligenceController()

        await controller.scan_source("ollama")
        await controller.resource_action("qwen3:30b", "install")
        data = await controller.get_discovery_data()

        source = data["sources"]["ollama"]
        resource = next(item for item in source["resources"] if item["id"] == "qwen3:30b")

        assert source["status"] == "rescanned"
        assert "last_scanned_at" in source
        assert resource["status"] == "installed"
        assert resource["last_action"] == "install"

    asyncio.run(run_actions())


def test_rag_document_actions_update_visible_rag_state():
    async def run_actions():
        controller = IntelligenceController()

        await controller.upload_documents(["C:/tmp/Architecture_Notes.md"])
        data = await controller.get_rag_data()
        uploaded = next(doc for doc in data["documents"] if doc["id"] == "Architecture_Notes")

        assert uploaded["status"] == "indexed"
        assert data["stats"]["total_documents"] >= 1

        await controller.reindex_document("Architecture_Notes")
        data = await controller.get_rag_data()
        reindexed = next(doc for doc in data["documents"] if doc["id"] == "Architecture_Notes")
        assert reindexed["status"] == "indexed"
        assert "indexed_at" in reindexed

        await controller.delete_document("Architecture_Notes")
        data = await controller.get_rag_data()
        assert all(doc["id"] != "Architecture_Notes" for doc in data["documents"])

    asyncio.run(run_actions())
