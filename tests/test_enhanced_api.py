"""Integration tests for enhanced API."""

import os
import sys
import pytest
import json
import tempfile
import uuid
from pathlib import Path

# Set test database path BEFORE importing nexus modules
os.environ['DATABASE_PATH'] = os.path.join(
    tempfile.gettempdir(),
    f'test_nexus_{uuid.uuid4().hex}.db',
)
os.environ['TESTING'] = 'True'

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from nexus.core.enhanced_api import app, api_key_manager, cache_manager, cost_tracker, metrics


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Set up test database at session start and clean up at session end."""
    test_db_path = os.environ.get('DATABASE_PATH')
    from nexus.core.enhanced_api import runtime

    # Remove test database if it exists
    runtime.reset_for_testing()
    if test_db_path and os.path.exists(test_db_path):
        os.remove(test_db_path)

    yield

    # Clean up test database after all tests
    runtime.reset_for_testing()
    if test_db_path and os.path.exists(test_db_path):
        os.remove(test_db_path)


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def api_key():
    """Create a test API key."""
    # Create a test user (or reuse existing)
    try:
        user = api_key_manager.create_user(
            username="test_user",
            email="test@example.com",
            role="user"
        )
    except ValueError:
        # User already exists from previous test class; look up by username
        from nexus.core.database import get_db
        from nexus.core.database.repositories import UserRepository
        from nexus.core.auth.models import User, UserRole
        db = get_db()
        with db.get_session() as session:
            user_repo = UserRepository(session)
            user_model = user_repo.get_by_username("test_user")
            user = User(
                user_id=user_model.user_id,
                username=user_model.username,
                email=user_model.email,
                role=UserRole(user_model.role),
                created_at=user_model.created_at,
            )

    # Generate API key
    raw_key, key_obj = api_key_manager.generate_key(
        user_id=user.user_id,
        name="Test Key",
        rate_limit=1000
    )

    yield raw_key

    # Cleanup handled by autouse cleanup fixture


@pytest.fixture
def admin_api_key():
    """Create an admin API key."""
    # Create admin user (or reuse existing)
    try:
        user = api_key_manager.create_user(
            username="admin_user",
            email="admin@example.com",
            role="admin"
        )
    except ValueError:
        from nexus.core.database import get_db
        from nexus.core.database.repositories import UserRepository
        from nexus.core.auth.models import User, UserRole
        db = get_db()
        with db.get_session() as session:
            user_repo = UserRepository(session)
            user_model = user_repo.get_by_username("admin_user")
            user = User(
                user_id=user_model.user_id,
                username=user_model.username,
                email=user_model.email,
                role=UserRole(user_model.role),
                created_at=user_model.created_at,
            )

    # Generate API key
    raw_key, key_obj = api_key_manager.generate_key(
        user_id=user.user_id,
        name="Admin Key",
        rate_limit=1000
    )

    yield raw_key


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up after each test."""
    yield
    # Clear cache
    cache_manager.clear_all()
    # Reset cost tracker alerts
    cost_tracker.reset_alerts()

    # Clear database tables for clean state between tests
    from nexus.core.database.connection import get_db
    from sqlalchemy import text
    db_connection = get_db()
    session = db_connection.get_session()
    try:
        # Clear all users and API keys from database
        session.execute(text("DELETE FROM api_keys"))
        session.execute(text("DELETE FROM users"))
        session.execute(text("DELETE FROM cost_entries"))
        session.execute(text("DELETE FROM usage_entries"))
        session.commit()
    except Exception as e:
        # If tables don't exist or error occurs, just continue
        session.rollback()
    finally:
        session.close()


class TestHealthAndStatus:
    """Tests for health and status endpoints."""

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get('/health')

        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert data['service'] == 'Nexus Enhanced API'

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get('/metrics')

        assert response.status_code == 200
        assert b'thenexus' in response.data

    def test_status_endpoint(self, client):
        """Test status endpoint."""
        response = client.get('/status')

        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'operational'
        assert 'models_available' in data
        assert 'cache' in data
        assert 'budget' in data


class TestAuthentication:
    """Tests for authentication endpoints."""

    def test_register_user(self, client):
        """Test user registration."""
        import secrets
        uname = f"newuser_{secrets.token_hex(4)}"
        response = client.post('/auth/register', json={
            'username': uname,
            'email': f'{uname}@example.com',
            'role': 'user'
        })

        assert response.status_code == 201
        data = response.get_json()
        assert data['username'] == uname
        assert 'api_key' in data
        assert data['api_key'].startswith('sk_')
        assert data['role'] == 'user'
        assert 'message' in data

    def test_register_user_missing_fields(self, client):
        """Test registration with missing fields."""
        response = client.post('/auth/register', json={
            'username': 'newuser'
        })

        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_register_user_default_role(self, client):
        """Test registration with default role."""
        import secrets
        uname = f"defaultrole_{secrets.token_hex(4)}"
        response = client.post('/auth/register', json={
            'username': uname,
            'email': f'{uname}@example.com'
        })

        assert response.status_code == 201
        data = response.get_json()
        assert data['role'] == 'user'

    def test_list_keys_authenticated(self, client, api_key):
        """Test listing API keys with authentication."""
        response = client.get('/auth/keys', headers={
            'X-API-Key': api_key
        })

        assert response.status_code == 200
        data = response.get_json()
        assert 'keys' in data
        assert len(data['keys']) > 0

    def test_list_keys_unauthenticated(self, client):
        """Test listing keys without authentication."""
        response = client.get('/auth/keys')

        assert response.status_code == 401


class TestThinkEndpoint:
    """Tests for /think endpoint."""

    def test_think_with_valid_input(self, client, api_key):
        """Test /think with valid input."""
        response = client.post('/think',
            headers={'X-API-Key': api_key},
            json={'input': 'Test input'}
        )

        assert response.status_code == 200
        data = response.get_json()
        assert 'response' in data
        assert data['status'] == 'success'
        assert 'processing_time_ms' in data

    def test_think_without_auth(self, client):
        """Test /think without authentication."""
        response = client.post('/think', json={'input': 'Test'})

        assert response.status_code == 401

    def test_think_missing_input(self, client, api_key):
        """Test /think with missing input."""
        response = client.post('/think',
            headers={'X-API-Key': api_key},
            json={}
        )

        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_think_empty_input(self, client, api_key):
        """Test /think with empty input."""
        response = client.post('/think',
            headers={'X-API-Key': api_key},
            json={'input': ''}
        )

        assert response.status_code == 400

    def test_think_input_too_long(self, client, api_key):
        """Test /think with input exceeding max length."""
        long_input = 'a' * 10001
        response = client.post('/think',
            headers={'X-API-Key': api_key},
            json={'input': long_input}
        )

        assert response.status_code == 400

    def test_think_invalid_api_key(self, client):
        """Test /think with invalid API key."""
        response = client.post('/think',
            headers={'X-API-Key': 'invalid_key'},
            json={'input': 'Test'}
        )

        assert response.status_code == 401


class TestEnsembleEndpoint:
    """Tests for /ensemble endpoint."""

    def test_ensemble_basic(self, client, api_key):
        """Test basic ensemble inference."""
        response = client.post('/ensemble',
            headers={'X-API-Key': api_key},
            json={'input': 'Hello world'}
        )

        assert response.status_code == 200
        data = response.get_json()
        assert 'response' in data
        assert 'model' in data
        assert 'provider' in data
        assert 'score' in data
        assert 'confidence' in data
        assert 'strategy_used' in data
        assert 'models_queried' in data
        assert 'total_cost_usd' in data
        assert 'avg_latency_ms' in data
        assert 'metadata' in data
        assert 'cached' in data
        assert data['cached'] is False
        assert data['status'] == 'success'
        assert data['strategy_used'] == 'simple_best'  # default strategy

    def test_ensemble_caching(self, client, api_key):
        """Test ensemble response caching."""
        input_data = {'input': 'Cached test prompt'}

        # First request - should not be cached
        response1 = client.post('/ensemble',
            headers={'X-API-Key': api_key},
            json=input_data
        )

        assert response1.status_code == 200
        data1 = response1.get_json()
        assert data1['cached'] is False

        # Second request - should be cached
        response2 = client.post('/ensemble',
            headers={'X-API-Key': api_key},
            json=input_data
        )

        assert response2.status_code == 200
        data2 = response2.get_json()
        assert data2['cached'] is True

        # Response should be the same
        assert data1['response'] == data2['response']

    def test_ensemble_cache_disabled(self, client, api_key):
        """Test ensemble with caching disabled."""
        input_data = {'input': 'Non-cached prompt', 'cache': False}

        # First request
        response1 = client.post('/ensemble',
            headers={'X-API-Key': api_key},
            json=input_data
        )

        assert response1.status_code == 200
        data1 = response1.get_json()
        assert data1['cached'] is False

        # Second request should also not be cached
        response2 = client.post('/ensemble',
            headers={'X-API-Key': api_key},
            json=input_data
        )

        assert response2.status_code == 200
        data2 = response2.get_json()
        assert data2['cached'] is False

    def test_ensemble_cost_tracking(self, client, api_key):
        """Test that ensemble tracks costs."""
        initial_summary = cost_tracker.get_summary()
        initial_requests = initial_summary.total_requests

        response = client.post('/ensemble',
            headers={'X-API-Key': api_key},
            json={'input': 'Cost tracking test'}
        )

        assert response.status_code == 200
        data = response.get_json()
        assert 'total_cost_usd' in data
        assert data['total_cost_usd'] >= 0

        # Should have recorded a cost entry
        updated_summary = cost_tracker.get_summary()
        assert updated_summary.total_requests == initial_requests + 1

    def test_ensemble_without_auth(self, client):
        """Test ensemble without authentication."""
        response = client.post('/ensemble',
            json={'input': 'Test'}
        )

        assert response.status_code == 401

    def test_ensemble_missing_input(self, client, api_key):
        """Test ensemble with missing input."""
        response = client.post('/ensemble',
            headers={'X-API-Key': api_key},
            json={}
        )

        assert response.status_code == 400


class TestCostEndpoints:
    """Tests for cost tracking endpoints."""

    def test_cost_summary(self, client, api_key):
        """Test getting cost summary."""
        # First make a request to generate costs
        client.post('/ensemble',
            headers={'X-API-Key': api_key},
            json={'input': 'Test for costs'}
        )

        # Get cost summary
        response = client.get('/costs/summary',
            headers={'X-API-Key': api_key}
        )

        assert response.status_code == 200
        data = response.get_json()
        assert 'total_cost' in data
        assert 'total_requests' in data
        assert 'total_tokens' in data
        assert 'cost_by_model' in data

    def test_cost_summary_without_auth(self, client):
        """Test cost summary without authentication."""
        response = client.get('/costs/summary')

        assert response.status_code == 401

    def test_budget_status_admin(self, client, admin_api_key):
        """Test budget status with admin key."""
        response = client.get('/costs/budget',
            headers={'X-API-Key': admin_api_key}
        )

        assert response.status_code == 200
        data = response.get_json()
        assert 'budget_limit' in data
        assert 'current_spend' in data
        assert 'remaining' in data

    def test_budget_status_non_admin(self, client, api_key):
        """Test budget status with non-admin key."""
        response = client.get('/costs/budget',
            headers={'X-API-Key': api_key}
        )

        # Should be forbidden for non-admin users
        assert response.status_code == 403


class TestErrorHandling:
    """Tests for error handling."""

    def test_404_error(self, client):
        """Test 404 error handler."""
        response = client.get('/nonexistent')

        assert response.status_code == 404
        data = response.get_json()
        assert 'error' in data

    def test_method_not_allowed(self, client):
        """Test method not allowed."""
        response = client.get('/think')

        assert response.status_code == 405


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_enforcement(self, client):
        """Test rate limiting is enforced."""
        import secrets
        uname = f"limited_{secrets.token_hex(4)}"
        # Create user with very low rate limit
        user = api_key_manager.create_user(
            username=uname,
            email=f"{uname}@example.com",
            role="user"
        )

        raw_key, key_obj = api_key_manager.generate_key(
            user_id=user.user_id,
            name="Limited Key",
            rate_limit=2  # Only 2 requests allowed
        )

        # First request - should succeed
        response1 = client.post('/think',
            headers={'X-API-Key': raw_key},
            json={'input': 'Request 1'}
        )
        assert response1.status_code == 200

        # Second request - should succeed
        response2 = client.post('/think',
            headers={'X-API-Key': raw_key},
            json={'input': 'Request 2'}
        )
        assert response2.status_code == 200

        # Third request - should be rate limited
        response3 = client.post('/think',
            headers={'X-API-Key': raw_key},
            json={'input': 'Request 3'}
        )
        assert response3.status_code == 429


class TestFullWorkflow:
    """Tests for complete user workflows."""

    def test_complete_user_flow(self, client):
        """Test complete flow: register -> authenticate -> use API."""
        import secrets
        uname = f"workflow_{secrets.token_hex(4)}"
        # 1. Register a new user
        reg_response = client.post('/auth/register', json={
            'username': uname,
            'email': f'{uname}@example.com'
        })

        assert reg_response.status_code == 201
        reg_data = reg_response.get_json()
        api_key = reg_data['api_key']

        # 2. Use the API with the key
        think_response = client.post('/think',
            headers={'X-API-Key': api_key},
            json={'input': 'Hello'}
        )

        assert think_response.status_code == 200

        # 3. Check cost summary
        cost_response = client.get('/costs/summary',
            headers={'X-API-Key': api_key}
        )

        assert cost_response.status_code == 200

        # 4. List API keys
        keys_response = client.get('/auth/keys',
            headers={'X-API-Key': api_key}
        )

        assert keys_response.status_code == 200
        keys_data = keys_response.get_json()
        assert len(keys_data['keys']) == 1

    def test_ensemble_with_cost_and_cache(self, client, api_key):
        """Test ensemble with caching and cost tracking."""
        # First request
        response1 = client.post('/ensemble',
            headers={'X-API-Key': api_key},
            json={'input': 'Integration test'}
        )

        assert response1.status_code == 200
        data1 = response1.get_json()
        initial_cost = data1['total_cost_usd']
        assert initial_cost >= 0

        # Second request (cached) - should have same response but cached=True
        response2 = client.post('/ensemble',
            headers={'X-API-Key': api_key},
            json={'input': 'Integration test'}
        )

        assert response2.status_code == 200
        data2 = response2.get_json()
        assert data2['cached'] is True
        assert data2['response'] == data1['response']

        # Check cost summary
        cost_response = client.get('/costs/summary',
            headers={'X-API-Key': api_key}
        )

        cost_data = cost_response.get_json()
        # At least one cost entry should be recorded (cached request doesn't add cost)
        assert cost_data['total_requests'] >= 1


class TestStrategySelection:
    """Tests for ensemble strategy selection."""

    def test_simple_best_strategy(self, client, api_key):
        """Test simple_best strategy (default)."""
        response = client.post('/ensemble',
            headers={'X-API-Key': api_key},
            json={'input': 'Test prompt', 'strategy': 'simple_best'}
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data['strategy_used'] == 'simple_best'
        assert 'score' in data
        assert 'confidence' in data

    def test_weighted_voting_strategy(self, client, api_key):
        """Test weighted_voting strategy."""
        response = client.post('/ensemble',
            headers={'X-API-Key': api_key},
            json={'input': 'Test prompt', 'strategy': 'weighted_voting'}
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data['strategy_used'] == 'weighted_voting'

    def test_cascading_strategy(self, client, api_key):
        """Test cascading strategy (cost-optimized)."""
        response = client.post('/ensemble',
            headers={'X-API-Key': api_key},
            json={'input': 'Simple question', 'strategy': 'cascading'}
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data['strategy_used'] == 'cascading'
        assert data['models_queried'] >= 1  # May early-stop

    def test_dynamic_weight_strategy(self, client, api_key):
        """Test dynamic_weight strategy."""
        response = client.post('/ensemble',
            headers={'X-API-Key': api_key},
            json={'input': 'Test prompt', 'strategy': 'dynamic_weight'}
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data['strategy_used'] == 'dynamic_weight'

    def test_majority_voting_strategy(self, client, api_key):
        """Test majority_voting strategy."""
        response = client.post('/ensemble',
            headers={'X-API-Key': api_key},
            json={'input': 'Test prompt', 'strategy': 'majority_voting'}
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data['strategy_used'] == 'majority_voting'

    def test_cost_optimized_strategy(self, client, api_key):
        """Test cost_optimized strategy."""
        response = client.post('/ensemble',
            headers={'X-API-Key': api_key},
            json={'input': 'Test prompt', 'strategy': 'cost_optimized'}
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data['strategy_used'] == 'cost_optimized'
        assert 'total_cost_usd' in data

    def test_invalid_strategy(self, client, api_key):
        """Test invalid strategy returns error."""
        response = client.post('/ensemble',
            headers={'X-API-Key': api_key},
            json={'input': 'Test prompt', 'strategy': 'invalid_strategy'}
        )

        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert 'valid_strategies' in data

    def test_strategy_specific_caching(self, client, api_key):
        """Test that different strategies have separate cache entries."""
        prompt_data = {'input': 'Cache test prompt'}

        # Request with simple_best
        response1 = client.post('/ensemble',
            headers={'X-API-Key': api_key},
            json={**prompt_data, 'strategy': 'simple_best'}
        )
        assert response1.status_code == 200
        data1 = response1.get_json()
        assert data1['cached'] is False

        # Same prompt with weighted_voting - should not be cached
        response2 = client.post('/ensemble',
            headers={'X-API-Key': api_key},
            json={**prompt_data, 'strategy': 'weighted_voting'}
        )
        assert response2.status_code == 200
        data2 = response2.get_json()
        assert data2['cached'] is False  # Different strategy = different cache

        # Same prompt with simple_best again - should be cached
        response3 = client.post('/ensemble',
            headers={'X-API-Key': api_key},
            json={**prompt_data, 'strategy': 'simple_best'}
        )
        assert response3.status_code == 200
        data3 = response3.get_json()
        assert data3['cached'] is True

    def test_strategy_metadata(self, client, api_key):
        """Test that strategy results include metadata."""
        response = client.post('/ensemble',
            headers={'X-API-Key': api_key},
            json={'input': 'Test prompt', 'strategy': 'simple_best'}
        )

        assert response.status_code == 200
        data = response.get_json()
        assert 'metadata' in data
        assert isinstance(data['metadata'], dict)


class TestStrategiesEndpoint:
    """Tests for /strategies endpoint."""

    def test_list_strategies(self, client):
        """Test listing available strategies."""
        response = client.get('/strategies')

        assert response.status_code == 200
        data = response.get_json()
        assert 'strategies' in data
        assert 'default' in data
        assert data['default'] == 'simple_best'

        strategies = data['strategies']
        expected_strategies = [
            'simple_best', 'weighted_voting', 'cascading',
            'dynamic_weight', 'majority_voting', 'cost_optimized'
        ]

        for strategy in expected_strategies:
            assert strategy in strategies
            assert 'name' in strategies[strategy]
            assert 'description' in strategies[strategy]
            assert 'use_case' in strategies[strategy]

    def test_strategies_endpoint_no_auth_required(self, client):
        """Test that /strategies endpoint doesn't require authentication."""
        response = client.get('/strategies')
        assert response.status_code == 200


class TestAnalyticsEndpoint:
    """Tests for /analytics/usage endpoint."""

    def test_usage_analytics_admin(self, client, admin_api_key):
        """Test usage analytics with admin key."""
        # Make some requests first
        client.post('/ensemble',
            headers={'X-API-Key': admin_api_key},
            json={'input': 'Test 1'}
        )
        client.post('/ensemble',
            headers={'X-API-Key': admin_api_key},
            json={'input': 'Test 2'}
        )

        # Get analytics
        response = client.get('/analytics/usage',
            headers={'X-API-Key': admin_api_key}
        )

        assert response.status_code == 200
        data = response.get_json()
        assert 'summary' in data
        assert 'hourly_stats' in data
        assert 'top_users' in data

        summary = data['summary']
        assert 'total_requests' in summary
        assert summary['total_requests'] >= 2

    def test_usage_analytics_non_admin(self, client, api_key):
        """Test usage analytics with non-admin key (should be forbidden)."""
        response = client.get('/analytics/usage',
            headers={'X-API-Key': api_key}
        )

        assert response.status_code == 403

    def test_usage_analytics_no_auth(self, client):
        """Test usage analytics without authentication."""
        response = client.get('/analytics/usage')

        assert response.status_code == 401

    def test_usage_analytics_custom_hours(self, client, admin_api_key):
        """Test usage analytics with custom time range."""
        response = client.get('/analytics/usage?hours=48',
            headers={'X-API-Key': admin_api_key}
        )

        assert response.status_code == 200
        data = response.get_json()
        assert 'hourly_stats' in data
