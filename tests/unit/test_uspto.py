"""Tests for USPTO integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from nexus.discovery.uspto_integration import USPTOIntegration
from nexus.discovery.resource_discovery import (
    DiscoveredResource,
    ResourceDiscovery,
    ResourceSource,
    ResourceType,
)


@pytest.fixture
def mock_discovery():
    rd = MagicMock(spec=ResourceDiscovery)
    rd.register_source = MagicMock()
    rd.register_resource = MagicMock(return_value=True)
    return rd


@pytest.fixture
def integration(mock_discovery):
    return USPTOIntegration(
        resource_discovery=mock_discovery,
        user_agent="TestAgent/1.0 test@example.com",
        tracked_assignees=["Google LLC"],  # Single assignee for faster tests
    )


class TestInit:
    def test_registers_source(self, mock_discovery, integration):
        mock_discovery.register_source.assert_called_once_with(
            ResourceSource.USPTO, integration
        )

    def test_default_user_agent(self, mock_discovery):
        i = USPTOIntegration(mock_discovery)
        assert "Nexus" in i.user_agent

    def test_env_user_agent(self, mock_discovery, monkeypatch):
        monkeypatch.setenv("USPTO_USER_AGENT", "EnvAgent/1.0 env@test.com")
        i = USPTOIntegration(mock_discovery, user_agent=None)
        assert i.user_agent == "EnvAgent/1.0 env@test.com"

    def test_default_tracked_assignees(self, mock_discovery):
        i = USPTOIntegration(mock_discovery)
        assert "Google LLC" in i.tracked_assignees
        assert "Microsoft Corporation" in i.tracked_assignees
        assert "NVIDIA Corporation" in i.tracked_assignees

    def test_custom_tracked_assignees(self, integration):
        assert integration.tracked_assignees == ["Google LLC"]

    def test_headers(self, integration):
        h = integration._headers
        assert "User-Agent" in h
        assert "TestAgent" in h["User-Agent"]
        assert h["Accept"] == "application/json"
        assert h["Content-Type"] == "application/json"


class TestBuildQueryCriteria:
    def test_keyword_search(self, integration):
        criteria = integration._build_query_criteria(keyword="machine learning")
        assert "_or" in criteria
        assert len(criteria["_or"]) == 2

    def test_inventor_search(self, integration):
        criteria = integration._build_query_criteria(inventor="Smith")
        assert "_text_any" in criteria
        assert criteria["_text_any"]["inventor_last_name"] == "Smith"

    def test_assignee_search(self, integration):
        criteria = integration._build_query_criteria(assignee="Google LLC")
        assert "_contains" in criteria
        assert criteria["_contains"]["assignee_organization"] == "Google LLC"

    def test_classification_search(self, integration):
        criteria = integration._build_query_criteria(cpc_classification="G06N")
        assert "_begins" in criteria
        assert criteria["_begins"]["cpc_group_id"] == "G06N"

    def test_date_range(self, integration):
        criteria = integration._build_query_criteria(
            keyword="AI", date_from="2024-01-01", date_to="2024-12-31"
        )
        assert "_and" in criteria
        assert len(criteria["_and"]) == 3

    def test_combined_criteria(self, integration):
        criteria = integration._build_query_criteria(
            keyword="neural network",
            assignee="Google LLC",
            cpc_classification="G06N",
        )
        assert "_and" in criteria
        assert len(criteria["_and"]) == 3

    def test_no_criteria_returns_none(self, integration):
        criteria = integration._build_query_criteria()
        assert criteria is None

    def test_single_criterion_no_and_wrapper(self, integration):
        criteria = integration._build_query_criteria(assignee="Apple Inc.")
        # Single criterion should not be wrapped in _and
        assert "_and" not in criteria
        assert "_contains" in criteria

    def test_date_from_only(self, integration):
        criteria = integration._build_query_criteria(
            keyword="AI", date_from="2024-01-01"
        )
        assert "_and" in criteria
        assert len(criteria["_and"]) == 2

    def test_date_to_only(self, integration):
        criteria = integration._build_query_criteria(
            keyword="AI", date_to="2024-12-31"
        )
        assert "_and" in criteria
        assert len(criteria["_and"]) == 2


class TestParsePatentResults:
    def test_parses_full_patent(self, integration):
        data = {
            "patents": [
                {
                    "patent_number": "11234567",
                    "patent_title": "Neural Network Optimization Method",
                    "patent_abstract": "A method for optimizing neural networks.",
                    "patent_date": "2024-01-15",
                    "patent_type": "utility",
                    "patent_num_claims": 20,
                    "patent_kind": "B2",
                    "assignees": [
                        {"assignee_organization": "Google LLC", "assignee_type": "2"}
                    ],
                    "inventors": [
                        {"inventor_first_name": "John", "inventor_last_name": "Smith"},
                        {"inventor_first_name": "Jane", "inventor_last_name": "Doe"},
                    ],
                    "cpcs": [
                        {"cpc_group_id": "G06N3/08", "cpc_group_title": "Learning methods"}
                    ],
                }
            ]
        }
        results = integration._parse_patent_results(data)
        assert len(results) == 1
        p = results[0]
        assert p["patent_number"] == "11234567"
        assert p["patent_title"] == "Neural Network Optimization Method"
        assert p["assignee_organization"] == "Google LLC"
        assert "John Smith" in p["inventors"]
        assert "Jane Doe" in p["inventors"]
        assert p["cpc_group_id"] == "G06N3/08"
        assert p["patent_type"] == "utility"
        assert p["patent_num_claims"] == 20

    def test_parses_multiple_patents(self, integration):
        data = {
            "patents": [
                {
                    "patent_number": "11111111",
                    "patent_title": "Patent A",
                    "patent_abstract": "Abstract A",
                    "patent_date": "2024-01-01",
                    "patent_type": "utility",
                    "assignees": [],
                    "inventors": [],
                    "cpcs": [],
                },
                {
                    "patent_number": "22222222",
                    "patent_title": "Patent B",
                    "patent_abstract": "Abstract B",
                    "patent_date": "2024-02-01",
                    "patent_type": "design",
                    "assignees": [],
                    "inventors": [],
                    "cpcs": [],
                },
            ]
        }
        results = integration._parse_patent_results(data)
        assert len(results) == 2
        assert results[0]["patent_number"] == "11111111"
        assert results[1]["patent_number"] == "22222222"

    def test_empty_response(self, integration):
        assert integration._parse_patent_results({}) == []
        assert integration._parse_patent_results({"patents": []}) == []
        assert integration._parse_patent_results({"patents": None}) == []

    def test_missing_nested_fields(self, integration):
        data = {
            "patents": [
                {
                    "patent_number": "33333333",
                    "patent_title": "Minimal Patent",
                    # no assignees, inventors, cpcs
                }
            ]
        }
        results = integration._parse_patent_results(data)
        assert len(results) == 1
        assert results[0]["assignee_organization"] == ""
        assert results[0]["inventors"] == ""
        assert results[0]["cpc_group_id"] == ""

    def test_empty_assignee_list(self, integration):
        data = {
            "patents": [
                {
                    "patent_number": "44444444",
                    "patent_title": "Test",
                    "assignees": [],
                    "inventors": [],
                    "cpcs": [],
                }
            ]
        }
        results = integration._parse_patent_results(data)
        assert results[0]["assignee_organization"] == ""

    def test_none_assignee_organization(self, integration):
        data = {
            "patents": [
                {
                    "patent_number": "55555555",
                    "patent_title": "Test",
                    "assignees": [{"assignee_organization": None}],
                    "inventors": [],
                    "cpcs": [],
                }
            ]
        }
        results = integration._parse_patent_results(data)
        assert results[0]["assignee_organization"] == ""


class TestPatentToResource:
    def test_basic_conversion(self, integration):
        patent = {
            "patent_number": "11234567",
            "patent_title": "Neural Network Optimization Method",
            "patent_abstract": "A method for optimizing neural networks.",
            "patent_date": "2024-01-15",
            "patent_type": "utility",
            "patent_num_claims": 20,
            "assignee_organization": "Google LLC",
            "inventors": "John Smith, Jane Doe",
            "cpc_group_id": "G06N3/08",
        }
        resource = integration._patent_to_resource(patent)

        assert isinstance(resource, DiscoveredResource)
        assert resource.source == ResourceSource.USPTO
        assert resource.resource_type == ResourceType.DATASET
        assert resource.id == "uspto:11234567"
        assert "Neural Network Optimization Method" in resource.name
        assert "11234567" in resource.name
        assert resource.quality_score == 0.85
        assert "uspto" in resource.tags
        assert "patent" in resource.tags
        assert "11234567" in resource.tags
        assert "G06N3/08" in resource.tags
        assert "utility" in resource.tags
        assert "Google LLC" in resource.description
        assert resource.url == "https://patents.google.com/patent/US11234567"
        assert resource.raw_metadata["patent_number"] == "11234567"
        assert resource.raw_metadata["assignee_organization"] == "Google LLC"
        assert resource.raw_metadata["inventors"] == "John Smith, Jane Doe"
        assert resource.raw_metadata["num_claims"] == 20

    def test_missing_patent_number(self, integration):
        patent = {
            "patent_number": "",
            "patent_title": "Some Invention",
            "patent_abstract": "",
            "patent_date": "",
            "patent_type": "",
            "assignee_organization": "",
            "inventors": "",
            "cpc_group_id": "",
        }
        resource = integration._patent_to_resource(patent)
        assert resource.id == "uspto:Some Invention"
        assert resource.url == ""

    def test_capabilities_and_use_cases(self, integration):
        patent = {
            "patent_number": "99999999",
            "patent_title": "Test Patent",
            "patent_abstract": "",
            "patent_date": "",
            "patent_type": "",
            "assignee_organization": "",
            "inventors": "",
            "cpc_group_id": "",
        }
        resource = integration._patent_to_resource(patent)
        assert "patent_data" in resource.capabilities
        assert "intellectual_property" in resource.capabilities
        assert "research" in resource.capabilities
        assert "patent_analysis" in resource.use_cases
        assert "competitive_intelligence" in resource.use_cases
        assert resource.is_available is True

    def test_description_with_assignee_and_date(self, integration):
        patent = {
            "patent_number": "12345678",
            "patent_title": "Test",
            "patent_abstract": "An abstract.",
            "patent_date": "2024-06-01",
            "patent_type": "",
            "assignee_organization": "Apple Inc.",
            "inventors": "",
            "cpc_group_id": "",
        }
        resource = integration._patent_to_resource(patent)
        assert "Apple Inc." in resource.description
        assert "2024-06-01" in resource.description

    def test_long_abstract_truncated(self, integration):
        patent = {
            "patent_number": "88888888",
            "patent_title": "Test",
            "patent_abstract": "A" * 1000,
            "patent_date": "",
            "patent_type": "",
            "assignee_organization": "",
            "inventors": "",
            "cpc_group_id": "",
        }
        resource = integration._patent_to_resource(patent)
        # Abstract is truncated to 500 chars in description
        assert len(resource.description) <= 600  # 500 abstract + some prefix


class TestDiscover:
    @pytest.mark.asyncio
    async def test_discovers_patents(self, integration, mock_discovery):
        patents_data = {
            "patents": [
                {
                    "patent_number": "11111111",
                    "patent_title": "AI Method 1",
                    "patent_abstract": "Abstract 1",
                    "patent_date": "2024-01-01",
                    "patent_type": "utility",
                    "assignees": [{"assignee_organization": "Google LLC"}],
                    "inventors": [{"inventor_first_name": "A", "inventor_last_name": "B"}],
                    "cpcs": [{"cpc_group_id": "G06N3/08"}],
                },
                {
                    "patent_number": "22222222",
                    "patent_title": "AI Method 2",
                    "patent_abstract": "Abstract 2",
                    "patent_date": "2024-02-01",
                    "patent_type": "utility",
                    "assignees": [{"assignee_organization": "Google LLC"}],
                    "inventors": [],
                    "cpcs": [],
                },
            ]
        }

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=patents_data)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("aiohttp.ClientSession") as mock_cs:
            mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)

            count = await integration.discover()

        assert count == 2
        assert mock_discovery.register_resource.call_count == 2

    @pytest.mark.asyncio
    async def test_discover_api_error(self, integration, mock_discovery):
        mock_resp = MagicMock()
        mock_resp.status = 500

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("aiohttp.ClientSession") as mock_cs:
            mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)

            count = await integration.discover()

        assert count == 0
        mock_discovery.register_resource.assert_not_called()

    @pytest.mark.asyncio
    async def test_discover_network_exception(self, integration, mock_discovery):
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(side_effect=Exception("Connection refused")),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("aiohttp.ClientSession") as mock_cs:
            mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)

            count = await integration.discover()

        assert count == 0

    @pytest.mark.asyncio
    async def test_discover_skips_duplicate(self, integration, mock_discovery):
        """When register_resource returns False (duplicate), count should not increase."""
        mock_discovery.register_resource = MagicMock(return_value=False)

        patents_data = {
            "patents": [
                {
                    "patent_number": "11111111",
                    "patent_title": "Already Known",
                    "patent_abstract": "",
                    "patent_date": "2024-01-01",
                    "patent_type": "utility",
                    "assignees": [],
                    "inventors": [],
                    "cpcs": [],
                }
            ]
        }

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=patents_data)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("aiohttp.ClientSession") as mock_cs:
            mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)

            count = await integration.discover()

        assert count == 0


class TestSearchPatents:
    @pytest.mark.asyncio
    async def test_search_by_keyword(self, integration):
        patents_data = {
            "patents": [
                {
                    "patent_number": "11111111",
                    "patent_title": "Machine Learning Patent",
                    "patent_abstract": "About ML.",
                    "patent_date": "2024-01-01",
                    "patent_type": "utility",
                    "assignees": [],
                    "inventors": [],
                    "cpcs": [],
                }
            ]
        }

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=patents_data)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("aiohttp.ClientSession") as mock_cs:
            mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)

            results = await integration.search_patents(keyword="machine learning")

        assert len(results) == 1
        assert results[0]["patent_title"] == "Machine Learning Patent"

    @pytest.mark.asyncio
    async def test_search_no_criteria_returns_empty(self, integration):
        results = await integration.search_patents()
        assert results == []

    @pytest.mark.asyncio
    async def test_search_api_error(self, integration):
        mock_resp = MagicMock()
        mock_resp.status = 400

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("aiohttp.ClientSession") as mock_cs:
            mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)

            results = await integration.search_patents(keyword="test")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_exception(self, integration):
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(side_effect=Exception("timeout")),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("aiohttp.ClientSession") as mock_cs:
            mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)

            results = await integration.search_patents(assignee="Google LLC")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_max_results_capped(self, integration):
        """max_results is capped at 100 in the request."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"patents": []})

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("aiohttp.ClientSession") as mock_cs:
            mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)

            await integration.search_patents(keyword="test", max_results=200)

            # Verify the request body has per_page capped at 100
            call_args = mock_session.post.call_args
            body = call_args.kwargs.get("json") or call_args[1].get("json")
            assert body["o"]["per_page"] == 100


class TestGetPatent:
    @pytest.mark.asyncio
    async def test_returns_patent(self, integration):
        patents_data = {
            "patents": [
                {
                    "patent_number": "11234567",
                    "patent_title": "Specific Patent",
                    "patent_abstract": "Details here.",
                    "patent_date": "2024-03-15",
                    "patent_type": "utility",
                    "patent_num_claims": 15,
                    "assignees": [{"assignee_organization": "NVIDIA Corporation"}],
                    "inventors": [
                        {"inventor_first_name": "Alice", "inventor_last_name": "Chen"}
                    ],
                    "cpcs": [{"cpc_group_id": "G06N3/04"}],
                }
            ]
        }

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=patents_data)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("aiohttp.ClientSession") as mock_cs:
            mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await integration.get_patent("11234567")

        assert result is not None
        assert result["patent_number"] == "11234567"
        assert result["patent_title"] == "Specific Patent"
        assert result["assignee_organization"] == "NVIDIA Corporation"
        assert "Alice Chen" in result["inventors"]

    @pytest.mark.asyncio
    async def test_returns_none_on_not_found(self, integration):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"patents": []})

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("aiohttp.ClientSession") as mock_cs:
            mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await integration.get_patent("00000000")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_api_error(self, integration):
        mock_resp = MagicMock()
        mock_resp.status = 404

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("aiohttp.ClientSession") as mock_cs:
            mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await integration.get_patent("99999999")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self, integration):
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(side_effect=Exception("network error")),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("aiohttp.ClientSession") as mock_cs:
            mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await integration.get_patent("11234567")

        assert result is None


class TestSearchByClassification:
    @pytest.mark.asyncio
    async def test_delegates_to_search_patents(self, integration):
        with patch.object(integration, "search_patents", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [{"patent_number": "12345678"}]

            results = await integration.search_by_classification("G06N", max_results=10)

        mock_search.assert_called_once_with(
            cpc_classification="G06N",
            max_results=10,
        )
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_default_max_results(self, integration):
        with patch.object(integration, "search_patents", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = []

            await integration.search_by_classification("G06F")

        mock_search.assert_called_once_with(
            cpc_classification="G06F",
            max_results=25,
        )


class TestResourceSourceEnum:
    def test_uspto_enum_exists(self):
        assert ResourceSource.USPTO == "uspto"
        assert ResourceSource.USPTO.value == "uspto"
