"""Tests for SEC EDGAR integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from nexus.discovery.sec_edgar_integration import SECEdgarIntegration
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
    return SECEdgarIntegration(
        resource_discovery=mock_discovery,
        user_agent="TestAgent/1.0 test@example.com",
        form_types=["10-K", "10-Q"],
        tracked_ciks=["0000320193"],  # Apple only for faster tests
    )


class TestInit:
    def test_registers_source(self, mock_discovery, integration):
        mock_discovery.register_source.assert_called_once_with(
            ResourceSource.SEC_EDGAR, integration
        )

    def test_default_user_agent(self, mock_discovery):
        i = SECEdgarIntegration(mock_discovery)
        assert "Nexus" in i.user_agent

    def test_env_user_agent(self, mock_discovery, monkeypatch):
        monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "EnvAgent/1.0 env@test.com")
        i = SECEdgarIntegration(mock_discovery, user_agent=None)
        assert i.user_agent == "EnvAgent/1.0 env@test.com"

    def test_default_form_types(self, mock_discovery):
        i = SECEdgarIntegration(mock_discovery)
        assert "10-K" in i.form_types
        assert "10-Q" in i.form_types
        assert "8-K" in i.form_types

    def test_headers(self, integration):
        h = integration._headers
        assert "User-Agent" in h
        assert "TestAgent" in h["User-Agent"]


class TestParseSubmissions:
    def test_parses_parallel_arrays(self, integration):
        data = {
            "name": "Apple Inc.",
            "cik": "320193",
            "filings": {
                "recent": {
                    "form": ["10-K", "10-Q", "8-K", "10-K"],
                    "filingDate": ["2024-11-01", "2024-08-02", "2024-07-15", "2023-11-03"],
                    "accessionNumber": ["0000320193-24-000001", "0000320193-24-000002", "0000320193-24-000003", "0000320193-23-000001"],
                    "primaryDocument": ["aapl-10k.htm", "aapl-10q.htm", "aapl-8k.htm", "aapl-10k.htm"],
                    "reportDate": ["2024-09-28", "2024-06-29", "", "2023-09-30"],
                },
            },
        }
        # integration.form_types = ["10-K", "10-Q"] — no 8-K
        results = integration._parse_submissions(data, form_type=None, max_results=10)
        assert len(results) == 3  # 10-K, 10-Q, 10-K (8-K filtered out)
        assert results[0]["entity_name"] == "Apple Inc."
        assert results[0]["form_type"] == "10-K"

    def test_form_type_filter(self, integration):
        data = {
            "name": "Apple Inc.",
            "cik": "320193",
            "filings": {
                "recent": {
                    "form": ["10-K", "10-Q", "10-K"],
                    "filingDate": ["2024-11-01", "2024-08-02", "2023-11-03"],
                    "accessionNumber": ["acc1", "acc2", "acc3"],
                    "primaryDocument": ["d1", "d2", "d3"],
                    "reportDate": ["r1", "r2", "r3"],
                },
            },
        }
        results = integration._parse_submissions(data, form_type="10-K", max_results=10)
        assert len(results) == 2
        assert all(r["form_type"] == "10-K" for r in results)

    def test_max_results(self, integration):
        data = {
            "name": "Test",
            "cik": "1",
            "filings": {
                "recent": {
                    "form": ["10-K"] * 50,
                    "filingDate": ["2024-01-01"] * 50,
                    "accessionNumber": [f"acc-{i}" for i in range(50)],
                    "primaryDocument": ["doc"] * 50,
                    "reportDate": [""] * 50,
                },
            },
        }
        results = integration._parse_submissions(data, form_type=None, max_results=5)
        assert len(results) == 5

    def test_empty_filings(self, integration):
        data = {
            "name": "Empty Corp",
            "cik": "999",
            "filings": {"recent": {"form": [], "filingDate": []}},
        }
        results = integration._parse_submissions(data, form_type=None, max_results=10)
        assert results == []


class TestFilingToResource:
    def test_basic_conversion(self, integration):
        filing = {
            "cik": "320193",
            "entity_name": "Apple Inc.",
            "form_type": "10-K",
            "filing_date": "2024-11-01",
            "accession_number": "0000320193-24-000001",
            "primary_document": "aapl-10k.htm",
            "report_date": "2024-09-28",
        }
        resource = integration._filing_to_resource(filing)

        assert isinstance(resource, DiscoveredResource)
        assert resource.source == ResourceSource.SEC_EDGAR
        assert resource.resource_type == ResourceType.DATASET
        assert "sec_edgar:" in resource.id
        assert "Apple Inc." in resource.name
        assert "10-K" in resource.name
        assert resource.quality_score == 0.85
        assert "sec" in resource.tags
        assert resource.raw_metadata["form_type"] == "10-K"

    def test_missing_accession(self, integration):
        filing = {
            "cik": "320193",
            "entity_name": "Apple Inc.",
            "form_type": "10-K",
            "filing_date": "2024-11-01",
            "accession_number": "",
            "primary_document": "",
            "report_date": "",
        }
        resource = integration._filing_to_resource(filing)
        assert "sec_edgar:" in resource.id
        assert "10-K" in resource.id  # fallback ID format


class TestParseSearchResults:
    def test_parses_hits(self, integration):
        data = {
            "hits": {
                "hits": [
                    {
                        "_id": "0000320193-24-000001",
                        "_source": {
                            "entity_name": ["Apple Inc."],
                            "form_type": "10-K",
                            "file_date": "2024-11-01",
                            "period_of_report": "2024-09-28",
                            "cik": "320193",
                        },
                    },
                    {
                        "_id": "0000789019-24-000002",
                        "_source": {
                            "entity_name": "Microsoft Corporation",
                            "form_type": "10-Q",
                            "file_date": "2024-10-25",
                            "period_of_report": "2024-09-30",
                            "cik": "789019",
                        },
                    },
                ]
            }
        }
        results = integration._parse_search_results(data)
        assert len(results) == 2
        assert results[0]["entity_name"] == "Apple Inc."  # list unwrapped
        assert results[1]["entity_name"] == "Microsoft Corporation"  # string preserved

    def test_empty_hits(self, integration):
        assert integration._parse_search_results({"hits": {"hits": []}}) == []
        assert integration._parse_search_results({}) == []


class TestDiscover:
    @pytest.mark.asyncio
    async def test_registers_filings(self, integration, mock_discovery):
        submissions_data = {
            "name": "Apple Inc.",
            "cik": "320193",
            "filings": {
                "recent": {
                    "form": ["10-K", "10-Q"],
                    "filingDate": ["2024-11-01", "2024-08-02"],
                    "accessionNumber": ["acc1", "acc2"],
                    "primaryDocument": ["d1", "d2"],
                    "reportDate": ["r1", "r2"],
                },
            },
        }

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=submissions_data)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
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
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("aiohttp.ClientSession") as mock_cs:
            mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)

            count = await integration.discover()

        assert count == 0
        mock_discovery.register_resource.assert_not_called()


class TestGetCompanyFacts:
    @pytest.mark.asyncio
    async def test_returns_facts(self, integration):
        facts_data = {
            "cik": 320193,
            "entityName": "Apple Inc.",
            "facts": {"us-gaap": {"Revenue": {"units": {"USD": []}}}},
        }
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=facts_data)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("aiohttp.ClientSession") as mock_cs:
            mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await integration.get_company_facts("320193")

        assert result["entityName"] == "Apple Inc."

    @pytest.mark.asyncio
    async def test_returns_none_on_error(self, integration):
        mock_resp = MagicMock()
        mock_resp.status = 404

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("aiohttp.ClientSession") as mock_cs:
            mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await integration.get_company_facts("999999")

        assert result is None
