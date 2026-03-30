"""Tests for tool profiles system."""

from nexus.discovery.tool_profiles import (
    get_tool_profile,
    get_all_tool_profiles,
    get_tools_by_category,
    get_tools_by_availability,
    find_tools_for_task,
    get_tools_for_executive,
)


class TestToolProfileLookup:
    def test_get_known_tool(self):
        p = get_tool_profile("stripe")
        assert p is not None
        assert p.display_name == "Stripe"
        assert p.category == "business"

    def test_get_unknown_tool(self):
        assert get_tool_profile("nonexistent_tool") is None

    def test_case_insensitive(self):
        assert get_tool_profile("Slack") is not None
        assert get_tool_profile("GITHUB") is not None

    def test_all_profiles_have_required_fields(self):
        for name, profile in get_all_tool_profiles().items():
            assert profile.name, f"Missing name for {name}"
            assert profile.display_name, f"Missing display_name for {name}"
            assert profile.category, f"Missing category for {name}"
            assert profile.availability in ("internal", "integration", "marketplace"), \
                f"Invalid availability for {name}: {profile.availability}"
            assert len(profile.strengths) > 0, f"No strengths for {name}"
            assert len(profile.best_for) > 0, f"No best_for for {name}"


class TestToolCategories:
    def test_communication_tools(self):
        tools = get_tools_by_category("communication")
        names = [t.name for t in tools]
        assert "slack" in names
        assert "email" in names

    def test_business_tools(self):
        tools = get_tools_by_category("business")
        names = [t.name for t in tools]
        assert "stripe" in names

    def test_devops_tools(self):
        tools = get_tools_by_category("devops")
        names = [t.name for t in tools]
        assert "github" in names
        assert "docker" in names

    def test_unknown_category_empty(self):
        assert len(get_tools_by_category("underwater")) == 0


class TestToolAvailability:
    def test_internal_tools(self):
        tools = get_tools_by_availability("internal")
        assert len(tools) > 0
        assert all(t.availability == "internal" for t in tools)

    def test_marketplace_tools(self):
        tools = get_tools_by_availability("marketplace")
        assert len(tools) > 0
        assert all(t.availability == "marketplace" for t in tools)


class TestFindToolsForTask:
    def test_find_payment_tools(self):
        tools = find_tools_for_task("payment processing")
        names = [t.name for t in tools]
        assert "stripe" in names

    def test_find_communication_tools(self):
        tools = find_tools_for_task("team notifications")
        names = [t.name for t in tools]
        assert "slack" in names

    def test_find_code_management(self):
        tools = find_tools_for_task("code review")
        names = [t.name for t in tools]
        assert "github" in names

    def test_filter_by_category(self):
        tools = find_tools_for_task("tracking", category="monitoring")
        assert all(t.category == "monitoring" for t in tools)

    def test_filter_by_executive(self):
        tools = find_tools_for_task("data analysis", executive="CFO")
        assert all("CFO" in t.recommended_executives for t in tools)

    def test_filter_free_only(self):
        tools = find_tools_for_task("monitoring", require_free=True)
        assert all(t.has_free_tier for t in tools)

    def test_no_match(self):
        tools = find_tools_for_task("quantum teleportation")
        assert len(tools) == 0


class TestToolsForExecutive:
    def test_cto_tools(self):
        tools = get_tools_for_executive("CTO")
        names = [t.name for t in tools]
        assert "github" in names
        assert "docker" in names
        assert "slack" in names

    def test_cfo_tools(self):
        tools = get_tools_for_executive("CFO")
        names = [t.name for t in tools]
        assert "stripe" in names
        assert "sql_query" in names

    def test_unknown_exec_empty(self):
        tools = get_tools_for_executive("CXYZ")
        assert len(tools) == 0


class TestToolProfileMetadata:
    def test_has_auth_type(self):
        p = get_tool_profile("stripe")
        assert p.auth_type == "api_key"

    def test_has_pricing_model(self):
        p = get_tool_profile("stripe")
        assert p.pricing_model == "pay_per_use"

    def test_has_reliability(self):
        p = get_tool_profile("github")
        assert p.reliability == "high"

    def test_has_maturity(self):
        p = get_tool_profile("perplexity")
        assert p.maturity == "growing"

    def test_to_dict(self):
        p = get_tool_profile("slack")
        d = p.to_dict()
        assert "name" in d
        assert "strengths" in d
        assert "best_for" in d
        assert "recommended_executives" in d
        assert "reliability" in d


class TestProfileCount:
    def test_minimum_profiles(self):
        all_tools = get_all_tool_profiles()
        assert len(all_tools) >= 40  # 22 original + 23 open-source additions
