"""Tests for Shopforge integration in Nexus — tool profile + service handler."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nexus.discovery.tool_profiles import (
    get_tool_profile,
    get_tools_by_category,
    find_tools_for_task,
    get_tools_for_executive,
)


# ── Tool Profile Tests ─────────────────────────────────────────────────────────

class TestShopforgeToolProfile:
    def test_profile_registered(self):
        p = get_tool_profile("shopforge")
        assert p is not None
        assert p.display_name == "Shopforge Commerce Platform"

    def test_profile_category(self):
        p = get_tool_profile("shopforge")
        assert p.category == "business"

    def test_profile_availability(self):
        p = get_tool_profile("shopforge")
        assert p.availability == "internal"

    def test_profile_auth_type(self):
        p = get_tool_profile("shopforge")
        assert p.auth_type == "service_token"

    def test_profile_no_free_tier(self):
        p = get_tool_profile("shopforge")
        assert p.has_free_tier is False

    def test_in_business_category(self):
        tools = get_tools_by_category("business")
        names = [t.name for t in tools]
        assert "shopforge" in names

    def test_recommended_executives(self):
        p = get_tool_profile("shopforge")
        assert "CRO" in p.recommended_executives
        assert "CFO" in p.recommended_executives
        assert "CMO" in p.recommended_executives
        assert "CoS" in p.recommended_executives

    def test_findable_for_pricing_task(self):
        tools = find_tools_for_task("pricing optimization")
        names = [t.name for t in tools]
        assert "shopforge" in names

    def test_findable_for_inventory_task(self):
        tools = find_tools_for_task("inventory management")
        names = [t.name for t in tools]
        assert "shopforge" in names

    def test_findable_for_revenue_tracking(self):
        tools = find_tools_for_task("revenue tracking")
        names = [t.name for t in tools]
        assert "shopforge" in names

    def test_tools_for_cro(self):
        tools = get_tools_for_executive("CRO")
        names = [t.name for t in tools]
        assert "shopforge" in names

    def test_tools_for_cfo(self):
        tools = get_tools_for_executive("CFO")
        names = [t.name for t in tools]
        assert "shopforge" in names

    def test_best_for_fields(self):
        p = get_tool_profile("shopforge")
        assert len(p.best_for) >= 5
        assert "pricing optimization" in p.best_for

    def test_strengths_populated(self):
        p = get_tool_profile("shopforge")
        assert len(p.strengths) >= 5

    def test_to_dict(self):
        p = get_tool_profile("shopforge")
        d = p.to_dict()
        assert d["name"] == "shopforge"
        assert d["pricing_model"] == "internal"
        assert d["reliability"] == "high"


# ── Service Handler Dispatch Tests ──────────────────────────────────────────

class TestShopforgeServiceHandlerDispatch:
    """Test that the service handler has Shopforge endpoints registered."""

    def _make_handler(self):
        from nexus.coo.service_handler import NexusServiceHandler
        platform = MagicMock()
        redis = MagicMock()
        return NexusServiceHandler(platform, redis)

    def test_shopforge_list_storefronts_registered(self):
        h = self._make_handler()
        assert h._get_handler("shopforge_list_storefronts") is not None

    def test_shopforge_get_products_registered(self):
        h = self._make_handler()
        assert h._get_handler("shopforge_get_products") is not None

    def test_shopforge_optimize_pricing_registered(self):
        h = self._make_handler()
        assert h._get_handler("shopforge_optimize_pricing") is not None

    def test_shopforge_apply_price_registered(self):
        h = self._make_handler()
        assert h._get_handler("shopforge_apply_price") is not None

    def test_shopforge_get_analytics_registered(self):
        h = self._make_handler()
        assert h._get_handler("shopforge_get_analytics") is not None

    def test_shopforge_get_margins_registered(self):
        h = self._make_handler()
        assert h._get_handler("shopforge_get_margins") is not None

    def test_shopforge_provision_storefront_registered(self):
        h = self._make_handler()
        assert h._get_handler("shopforge_provision_storefront") is not None

    def test_shopforge_run_analysis_registered(self):
        h = self._make_handler()
        assert h._get_handler("shopforge_run_analysis") is not None

    def test_shopforge_executive_report_registered(self):
        h = self._make_handler()
        assert h._get_handler("shopforge_executive_report") is not None

    def test_shopforge_revenue_summary_registered(self):
        h = self._make_handler()
        assert h._get_handler("shopforge_revenue_summary") is not None

    def test_unknown_service_not_registered(self):
        h = self._make_handler()
        assert h._get_handler("shopforge_nonexistent") is None
