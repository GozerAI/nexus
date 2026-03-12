"""Tests for core_engine module."""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from nexus.core.core_engine import (
    CognitiveCore,
    SymbolicReasoner,
    HolographicMemory,
    ConceptualMapper,
)


class TestSymbolicReasoner:
    """Tests for SymbolicReasoner class."""

    def test_init(self):
        """Test SymbolicReasoner initialization."""
        reasoner = SymbolicReasoner()
        assert reasoner is not None

    def test_process(self):
        """Test process method."""
        reasoner = SymbolicReasoner()
        result = reasoner.process("test_concept")
        assert result == "Analyzed test_concept"
        assert isinstance(result, str)


class TestHolographicMemory:
    """Tests for HolographicMemory class."""

    def test_init(self):
        """Test HolographicMemory initialization."""
        memory = HolographicMemory()
        assert memory is not None
        assert memory.memory == {}

    def test_store_and_retrieve(self):
        """Test storing and retrieving from memory."""
        memory = HolographicMemory()
        memory.store("input1", "output1")
        
        result = memory.retrieve("input1")
        assert result == "output1"

    def test_retrieve_nonexistent(self):
        """Test retrieving nonexistent key."""
        memory = HolographicMemory()
        result = memory.retrieve("nonexistent")
        assert result is None

    def test_size(self):
        """Test size method."""
        memory = HolographicMemory()
        assert memory.size() == 0
        
        memory.store("input1", "output1")
        assert memory.size() == 1
        
        memory.store("input2", "output2")
        assert memory.size() == 2


class TestConceptualMapper:
    """Tests for ConceptualMapper class."""

    def test_init(self):
        """Test ConceptualMapper initialization."""
        mapper = ConceptualMapper()
        assert mapper is not None

    def test_convert(self):
        """Test convert method."""
        mapper = ConceptualMapper()
        result = mapper.convert("test_input")
        assert result == "Concept(test_input)"
        assert isinstance(result, str)

    def test_convert_empty_string(self):
        """Test convert with empty string."""
        mapper = ConceptualMapper()
        result = mapper.convert("")
        assert result == "Concept()"


class TestCognitiveCore:
    """Tests for CognitiveCore class."""

    def test_init(self):
        """Test CognitiveCore initialization."""
        core = CognitiveCore()
        assert core is not None
        assert isinstance(core.reasoning_engine, SymbolicReasoner)
        assert isinstance(core.memory_bank, HolographicMemory)
        assert isinstance(core.translator, ConceptualMapper)

    def test_think(self):
        """Test think method."""
        core = CognitiveCore()
        result = core.think("test_query")
        assert result == "Analyzed Concept(test_query)"
        assert isinstance(result, str)

    def test_think_empty_input(self):
        """Test think with empty input."""
        core = CognitiveCore()
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            core.think("")

    def test_think_none_input(self):
        """Test think with None input."""
        core = CognitiveCore()
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            core.think(None)

    def test_think_stores_in_memory(self):
        """Test that think stores results in memory."""
        core = CognitiveCore()
        input_data = "test_query"
        result = core.think(input_data)
        
        # Check that the result was stored
        stored = core.memory_bank.retrieve(input_data)
        assert stored == result
        assert core.memory_bank.size() == 1
