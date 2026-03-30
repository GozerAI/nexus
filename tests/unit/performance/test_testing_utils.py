"""Tests for testing utility modules."""

import pytest

from nexus.core.testing.sqlite_memory import InMemorySQLiteFactory, InMemorySession
from nexus.core.testing.fixture_factory import FixtureFactory, LazyFixture


# ── InMemorySQLiteFactory ────────────────────────────────────

class TestInMemorySQLiteFactory:
    def test_create_session(self):
        """Test creating an in-memory session without models."""
        factory = InMemorySQLiteFactory()
        session = factory.create()
        assert session is not None
        # Execute a simple query to verify it works
        result = session.execute("SELECT 1")
        assert result is not None
        session.close()

    def test_create_with_sqlalchemy_base(self):
        """Test with actual SQLAlchemy models."""
        from sqlalchemy import Column, Integer, String
        from sqlalchemy.orm import declarative_base

        Base = declarative_base()

        class TestModel(Base):
            __tablename__ = "test_items"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))

        factory = InMemorySQLiteFactory(base=Base)
        session = factory.create()

        # Insert and query
        from sqlalchemy import text
        session.execute(text("INSERT INTO test_items (id, name) VALUES (1, 'hello')"))
        session.commit()
        result = session.execute(text("SELECT name FROM test_items WHERE id=1"))
        row = result.fetchone()
        assert row[0] == "hello"
        session.close()

    def test_seed_function(self):
        from sqlalchemy import Column, Integer, String, text
        from sqlalchemy.orm import declarative_base

        Base = declarative_base()

        class Item(Base):
            __tablename__ = "items"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))

        def seed(session):
            session.execute(text("INSERT INTO items (id, name) VALUES (1, 'seed')"))

        factory = InMemorySQLiteFactory(base=Base)
        factory.add_seed(seed)
        session = factory.create()

        result = session.execute(text("SELECT name FROM items WHERE id=1"))
        row = result.fetchone()
        assert row[0] == "seed"
        session.close()

    def test_session_context_manager(self):
        factory = InMemorySQLiteFactory()
        with factory.create() as session:
            result = session.execute("SELECT 42")
            assert result is not None

    def test_create_session_factory(self):
        factory = InMemorySQLiteFactory()
        SessionClass = factory.create_session_factory()
        s1 = SessionClass()
        s2 = SessionClass()
        assert s1 is not s2
        s1.close()
        s2.close()

    def test_foreign_keys_enabled(self):
        from sqlalchemy import text
        factory = InMemorySQLiteFactory(enable_fk=True)
        session = factory.create()
        result = session.execute(text("PRAGMA foreign_keys"))
        row = result.fetchone()
        assert row[0] == 1
        session.close()


# ── FixtureFactory ───────────────────────────────────────────

class TestFixtureFactory:
    def test_register_and_get(self):
        factory = FixtureFactory()
        factory.register("greeting", lambda: "hello")
        assert factory.get("greeting") == "hello"

    def test_lazy_initialization(self):
        call_count = 0

        def create():
            nonlocal call_count
            call_count += 1
            return "value"

        factory = FixtureFactory()
        factory.register("lazy", create)
        assert call_count == 0
        factory.get("lazy")
        assert call_count == 1
        factory.get("lazy")  # Should use cached value
        assert call_count == 1

    def test_missing_fixture_raises(self):
        factory = FixtureFactory()
        with pytest.raises(KeyError, match="not registered"):
            factory.get("nonexistent")

    def test_builder(self):
        factory = FixtureFactory()
        factory.register_builder("user", lambda **kw: {
            "name": kw.get("name", "Default"),
            "email": kw.get("email", "default@test.com"),
        })
        user = factory.build("user", name="Alice")
        assert user["name"] == "Alice"
        assert user["email"] == "default@test.com"

    def test_builder_missing_raises(self):
        factory = FixtureFactory()
        with pytest.raises(KeyError):
            factory.build("nonexistent")

    def test_build_many(self):
        factory = FixtureFactory()
        factory.register_builder("item", lambda **kw: {"id": kw.get("id", 0)})
        items = factory.build_many("item", 5)
        assert len(items) == 5

    def test_sequence(self):
        factory = FixtureFactory()
        assert factory.sequence("users") == 1
        assert factory.sequence("users") == 2
        assert factory.sequence("users") == 3
        assert factory.sequence("other") == 1

    def test_unique_helpers(self):
        factory = FixtureFactory()
        email1 = factory.unique_email()
        email2 = factory.unique_email()
        assert email1 != email2
        assert "@test.example.com" in email1

        name1 = factory.unique_name("User")
        name2 = factory.unique_name("User")
        assert name1 != name2
        assert name1.startswith("User")

        id1 = factory.unique_id()
        id2 = factory.unique_id()
        assert id1 != id2

    def test_override(self):
        factory = FixtureFactory()
        factory.register("config", lambda: {"debug": False})
        factory.override("config", {"debug": True})
        assert factory.get("config") == {"debug": True}

    def test_reset(self):
        factory = FixtureFactory()
        factory.register("val", lambda: "original")
        factory.get("val")  # Initialize
        factory.reset()
        # After reset, fixture should re-initialize
        assert factory.get("val") == "original"

    def test_reset_fixture(self):
        call_count = 0

        def create():
            nonlocal call_count
            call_count += 1
            return call_count

        factory = FixtureFactory()
        factory.register("counter", create)
        assert factory.get("counter") == 1
        factory.reset_fixture("counter")
        assert factory.get("counter") == 2

    def test_dependencies(self):
        factory = FixtureFactory()
        factory.register("base", lambda: {"base": True})
        factory.register(
            "derived",
            lambda: {**factory.get("base"), "derived": True},
            dependencies=["base"],
        )
        result = factory.get("derived")
        assert result["base"] is True
        assert result["derived"] is True

    def test_cleanup_callback(self):
        cleaned = []
        factory = FixtureFactory()
        factory.add_cleanup(lambda: cleaned.append(True))
        factory.reset()
        assert len(cleaned) == 1

    def test_registered_names(self):
        factory = FixtureFactory()
        factory.register("a", lambda: 1)
        factory.register("b", lambda: 2)
        assert set(factory.registered_names) == {"a", "b"}

    def test_builder_names(self):
        factory = FixtureFactory()
        factory.register_builder("x", lambda **kw: kw)
        assert factory.builder_names == ["x"]

    def test_stats(self):
        factory = FixtureFactory()
        factory.register("a", lambda: 1)
        factory.get("a")
        stats = factory.get_stats()
        assert stats["fixtures_registered"] == 1
        assert stats["fixtures_initialized"] == 1


class TestLazyFixture:
    def test_value_cached(self):
        count = 0

        def create():
            nonlocal count
            count += 1
            return "val"

        fixture = LazyFixture(create, name="test")
        assert not fixture.is_initialized
        assert fixture.value == "val"
        assert fixture.is_initialized
        _ = fixture.value  # Second access
        assert count == 1

    def test_override(self):
        fixture = LazyFixture(lambda: "original", name="test")
        fixture.override("overridden")
        assert fixture.value == "overridden"

    def test_reset(self):
        fixture = LazyFixture(lambda: "val", name="test")
        fixture.value  # Initialize
        fixture.reset()
        assert not fixture.is_initialized

    def test_name_and_deps(self):
        fixture = LazyFixture(lambda: None, name="f", dependencies=["a", "b"])
        assert fixture.name == "f"
        assert fixture.dependencies == ["a", "b"]
