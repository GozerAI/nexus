"""Tests for memory optimization modules."""

import math
import os
import tempfile
import pytest

from nexus.core.memory_opt.mmap_embeddings import MMapEmbeddingStore, MMapConfig
from nexus.core.memory_opt.tensor_prealloc import (
    TensorPreAllocator, TensorPool, TensorSpec, PooledTensor,
)
from nexus.core.memory_opt.sparse_vectors import SparseVector, SparseVectorStore
from nexus.core.memory_opt.memory_profiler import (
    MemoryProfiler, MemorySnapshot, ComponentTracker,
)
from nexus.core.memory_opt.dimension_reduction import (
    DimensionReducer, ReductionConfig, ReductionMethod,
)
from nexus.core.memory_opt.auto_unloader import (
    AutoModelUnloader, MemoryPressureMonitor, PressureLevel, LoadedModel,
)


# ── MMapEmbeddingStore ───────────────────────────────────────

class TestMMapEmbeddingStore:
    @pytest.fixture
    def emb_path(self, tmp_path):
        """Return a path for a new embedding file (not pre-created)."""
        return str(tmp_path / "test.emb")

    def test_create_and_append(self, emb_path):
        config = MMapConfig(file_path=emb_path, dimension=4)
        store = MMapEmbeddingStore(config)
        store.open()
        idx = store.append([1.0, 2.0, 3.0, 4.0])
        assert idx == 0
        assert store.num_vectors == 1
        store.close()

    def test_append_and_get(self, emb_path):
        config = MMapConfig(file_path=emb_path, dimension=3)
        store = MMapEmbeddingStore(config)
        store.open()
        store.append([1.0, 2.0, 3.0])
        store.append([4.0, 5.0, 6.0])
        v = store.get(1)
        assert v == pytest.approx([4.0, 5.0, 6.0])
        store.close()

    def test_batch_operations(self, emb_path):
        config = MMapConfig(file_path=emb_path, dimension=2)
        store = MMapEmbeddingStore(config)
        store.open()
        indices = store.append_batch([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert indices == [0, 1, 2]
        batch = store.get_batch([0, 2])
        assert len(batch) == 2
        assert batch[0] == pytest.approx([1.0, 2.0])
        assert batch[1] == pytest.approx([5.0, 6.0])
        store.close()

    def test_dimension_mismatch(self, emb_path):
        config = MMapConfig(file_path=emb_path, dimension=3)
        store = MMapEmbeddingStore(config)
        store.open()
        with pytest.raises(ValueError, match="dimension"):
            store.append([1.0, 2.0])
        store.close()

    def test_index_out_of_range(self, emb_path):
        config = MMapConfig(file_path=emb_path, dimension=2)
        store = MMapEmbeddingStore(config)
        store.open()
        store.append([1.0, 2.0])
        with pytest.raises(IndexError):
            store.get(5)
        store.close()

    def test_reopen_existing(self, emb_path):
        config = MMapConfig(file_path=emb_path, dimension=2)
        store = MMapEmbeddingStore(config)
        store.open()
        store.append([1.0, 2.0])
        store.close()

        store2 = MMapEmbeddingStore(config)
        store2.open()
        assert store2.num_vectors == 1
        assert store2.get(0) == pytest.approx([1.0, 2.0])
        store2.close()

    def test_stats(self, emb_path):
        config = MMapConfig(file_path=emb_path, dimension=4)
        store = MMapEmbeddingStore(config)
        store.open()
        store.append([1.0, 2.0, 3.0, 4.0])
        stats = store.get_stats()
        assert stats["num_vectors"] == 1
        assert stats["dimension"] == 4
        store.close()


# ── TensorPreAllocator ──────────────────────────────────────

class TestTensorPreAllocator:
    def test_create_pool(self):
        alloc = TensorPreAllocator(max_total_memory_mb=100)
        pool = alloc.create_pool("test", shape=(10,), pool_size=5)
        assert pool.available_count == 5

    def test_acquire_release(self):
        alloc = TensorPreAllocator()
        alloc.create_pool("buf", shape=(100,), pool_size=3)
        tensor = alloc.acquire("buf")
        assert tensor is not None
        assert tensor.spec.name == "buf"
        alloc.release(tensor)

    def test_pool_exhaustion(self):
        alloc = TensorPreAllocator()
        alloc.create_pool("small", shape=(10,), pool_size=1)
        t1 = alloc.acquire("small")
        t2 = alloc.acquire("small")
        assert t1 is not None
        assert t2 is None  # Pool exhausted

    def test_memory_limit(self):
        alloc = TensorPreAllocator(max_total_memory_mb=1)
        with pytest.raises(MemoryError):
            alloc.create_pool("huge", shape=(1000000,), pool_size=1000)

    def test_zero_fill(self):
        alloc = TensorPreAllocator()
        alloc.create_pool("z", shape=(5,), pool_size=1)
        tensor = alloc.acquire("z", zero_fill=True)
        assert all(v == 0 for v in tensor.data)

    def test_stats(self):
        alloc = TensorPreAllocator()
        alloc.create_pool("p", shape=(10,), pool_size=2)
        stats = alloc.get_stats()
        assert "p" in stats["pools"]
        assert stats["total_allocated_mb"] > 0


class TestTensorPool:
    def test_basic(self):
        spec = TensorSpec(name="test", shape=(4,))
        pool = TensorPool(spec, pool_size=3)
        assert pool.available_count == 3
        assert pool.in_use_count == 0

    def test_acquire_and_release(self):
        spec = TensorSpec(name="test", shape=(4,))
        pool = TensorPool(spec, pool_size=2)
        t = pool.acquire()
        assert pool.available_count == 1
        assert pool.in_use_count == 1
        pool.release(t)
        assert pool.available_count == 2

    def test_stats(self):
        spec = TensorSpec(name="test", shape=(4,))
        pool = TensorPool(spec, pool_size=2)
        pool.acquire()
        stats = pool.get_stats()
        assert stats["acquires"] == 1
        assert stats["in_use"] == 1


class TestTensorSpec:
    def test_num_elements(self):
        spec = TensorSpec(name="t", shape=(3, 4, 5))
        assert spec.num_elements == 60

    def test_size_bytes_float32(self):
        spec = TensorSpec(name="t", shape=(100,), dtype="float32")
        assert spec.size_bytes == 400


# ── SparseVector ─────────────────────────────────────────────

class TestSparseVector:
    def test_from_dense(self):
        dense = [0.0, 1.0, 0.0, 2.0, 0.0]
        sv = SparseVector.from_dense(dense)
        assert sv.nnz == 2
        assert sv.dimension == 5

    def test_to_dense(self):
        sv = SparseVector(indices=[1, 3], values=[1.0, 2.0], dimension=5)
        dense = sv.to_dense()
        assert dense == [0.0, 1.0, 0.0, 2.0, 0.0]

    def test_dot_product(self):
        a = SparseVector(indices=[0, 2], values=[1.0, 3.0], dimension=4)
        b = SparseVector(indices=[0, 2], values=[2.0, 4.0], dimension=4)
        assert a.dot(b) == pytest.approx(14.0)

    def test_cosine_similarity(self):
        a = SparseVector(indices=[0], values=[1.0], dimension=3)
        b = SparseVector(indices=[0], values=[2.0], dimension=3)
        assert a.cosine_similarity(b) == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        a = SparseVector(indices=[0], values=[1.0], dimension=3)
        b = SparseVector(indices=[1], values=[1.0], dimension=3)
        assert a.cosine_similarity(b) == pytest.approx(0.0)

    def test_add(self):
        a = SparseVector(indices=[0, 2], values=[1.0, 3.0], dimension=4)
        b = SparseVector(indices=[1, 2], values=[2.0, 1.0], dimension=4)
        c = a.add(b)
        assert c.to_dense() == pytest.approx([1.0, 2.0, 4.0, 0.0])

    def test_scale(self):
        a = SparseVector(indices=[0, 1], values=[2.0, 3.0], dimension=3)
        b = a.scale(2.0)
        assert b.to_dense() == pytest.approx([4.0, 6.0, 0.0])

    def test_top_k(self):
        sv = SparseVector(indices=[0, 1, 2], values=[1.0, -5.0, 3.0], dimension=4)
        top = sv.top_k(2)
        assert top[0] == (1, -5.0)
        assert top[1] == (2, 3.0)

    def test_getitem(self):
        sv = SparseVector(indices=[1, 3], values=[10.0, 20.0], dimension=5)
        assert sv[1] == 10.0
        assert sv[0] == 0.0

    def test_sparsity(self):
        sv = SparseVector(indices=[0], values=[1.0], dimension=100)
        assert sv.sparsity == 0.99

    def test_compression_ratio(self):
        sv = SparseVector(indices=[0], values=[1.0], dimension=1000)
        assert sv.compression_ratio > 10

    def test_from_dict(self):
        sv = SparseVector.from_dict({2: 1.0, 5: 2.0}, dimension=10)
        assert sv.nnz == 2
        assert sv[5] == 2.0

    def test_repr(self):
        sv = SparseVector(indices=[0], values=[1.0], dimension=100)
        r = repr(sv)
        assert "dim=100" in r
        assert "nnz=1" in r


class TestSparseVectorStore:
    def test_put_get(self):
        store = SparseVectorStore()
        sv = SparseVector(indices=[0], values=[1.0], dimension=10)
        store.put("key1", sv)
        result = store.get("key1")
        assert result is not None
        assert result.nnz == 1

    def test_search(self):
        store = SparseVectorStore()
        store.put("a", SparseVector(indices=[0], values=[1.0], dimension=5))
        store.put("b", SparseVector(indices=[1], values=[1.0], dimension=5))
        query = SparseVector(indices=[0], values=[1.0], dimension=5)
        results = store.search(query, top_k=2)
        assert results[0][0] == "a"
        assert results[0][1] == pytest.approx(1.0)

    def test_stats(self):
        store = SparseVectorStore()
        store.put("k", SparseVector(indices=[0], values=[1.0], dimension=100))
        stats = store.get_stats()
        assert stats["stored_vectors"] == 1
        assert stats["compression_ratio"] > 1


# ── MemoryProfiler ───────────────────────────────────────────

class TestMemoryProfiler:
    def test_take_snapshot(self):
        profiler = MemoryProfiler()
        snap = profiler.take_snapshot(label="test")
        assert snap.label == "test"
        assert snap.gc_objects > 0

    def test_get_snapshots(self):
        profiler = MemoryProfiler()
        profiler.take_snapshot()
        profiler.take_snapshot()
        snaps = profiler.get_snapshots()
        assert len(snaps) == 2

    def test_component_tracker(self):
        profiler = MemoryProfiler()
        with profiler.track_component("test") as tracker:
            _ = [0] * 1000
        # Delta may be 0 on some systems
        assert tracker.delta_bytes >= 0

    def test_record_component_usage(self):
        profiler = MemoryProfiler()
        profiler.record_component_usage("cache", 1024 * 1024)
        usage = profiler.get_component_usage()
        assert usage["cache"] == 1024 * 1024

    def test_stats(self):
        profiler = MemoryProfiler()
        stats = profiler.get_stats()
        assert "current_rss_mb" in stats
        assert "gc_objects" in stats

    def test_snapshot_rss_mb(self):
        snap = MemorySnapshot(rss_bytes=10 * 1024 * 1024)
        assert snap.rss_mb == 10.0


# ── DimensionReducer ─────────────────────────────────────────

class TestDimensionReducer:
    def test_truncation(self):
        config = ReductionConfig(
            input_dim=10, output_dim=3, method=ReductionMethod.TRUNCATION
        )
        reducer = DimensionReducer(config)
        vec = list(range(10))
        result = reducer.reduce([float(x) for x in vec])
        assert len(result) == 3

    def test_random_projection(self):
        config = ReductionConfig(
            input_dim=100, output_dim=10, method=ReductionMethod.RANDOM_PROJECTION
        )
        reducer = DimensionReducer(config)
        vec = [float(i) for i in range(100)]
        result = reducer.reduce(vec)
        assert len(result) == 10

    def test_gaussian_rp(self):
        config = ReductionConfig(
            input_dim=50, output_dim=5, method=ReductionMethod.GAUSSIAN_RP
        )
        reducer = DimensionReducer(config)
        vec = [1.0] * 50
        result = reducer.reduce(vec)
        assert len(result) == 5

    def test_dimension_mismatch(self):
        config = ReductionConfig(input_dim=10, output_dim=3)
        reducer = DimensionReducer(config)
        with pytest.raises(ValueError, match="dimension"):
            reducer.reduce([1.0, 2.0])

    def test_batch_reduce(self):
        config = ReductionConfig(
            input_dim=10, output_dim=3, method=ReductionMethod.TRUNCATION
        )
        reducer = DimensionReducer(config)
        vecs = [[float(i)] * 10 for i in range(5)]
        results = reducer.reduce_batch(vecs)
        assert len(results) == 5
        assert all(len(r) == 3 for r in results)

    def test_measure_distortion(self):
        config = ReductionConfig(
            input_dim=20, output_dim=10, method=ReductionMethod.RANDOM_PROJECTION,
            normalize_output=False,
        )
        reducer = DimensionReducer(config)
        vecs = [[float(i + j) for j in range(20)] for i in range(10)]
        distortion = reducer.measure_distortion(vecs, sample_pairs=20)
        assert 0.0 <= distortion <= 2.0

    def test_recommended_dim(self):
        dim = DimensionReducer.recommended_output_dim(n_vectors=1000, epsilon=0.1)
        assert dim > 0
        # JL bound: O(log(n)/eps^2) - for n=1000, eps=0.1 this is ~16k
        dim_loose = DimensionReducer.recommended_output_dim(n_vectors=100, epsilon=0.5)
        assert dim_loose < dim  # More tolerance -> fewer dimensions needed

    def test_stats(self):
        config = ReductionConfig(input_dim=10, output_dim=5)
        reducer = DimensionReducer(config)
        reducer.reduce([1.0] * 10)
        stats = reducer.get_stats()
        assert stats["vectors_reduced"] == 1
        assert stats["compression_ratio"] == 0.5


# ── AutoModelUnloader ────────────────────────────────────────

class TestAutoModelUnloader:
    def test_register_model(self):
        unloader = AutoModelUnloader()
        model = unloader.register_model("gpt-4", memory_bytes=4 * 1024**3)
        assert model.model_name == "gpt-4"

    def test_mark_used(self):
        unloader = AutoModelUnloader()
        unloader.register_model("m", memory_bytes=1000)
        import time
        t1 = time.time()
        unloader.mark_used("m")
        loaded = unloader.get_loaded_models()
        assert loaded["m"]["use_count"] == 1

    def test_unregister(self):
        unloader = AutoModelUnloader()
        unloader.register_model("m", memory_bytes=1000)
        unloader.unregister_model("m")
        assert "m" not in unloader.get_loaded_models()

    def test_stats(self):
        unloader = AutoModelUnloader()
        unloader.register_model("m", memory_bytes=1024)
        stats = unloader.get_stats()
        assert stats["loaded_models"] == 1

    def test_events_empty(self):
        unloader = AutoModelUnloader()
        assert unloader.get_events() == []


class TestMemoryPressureMonitor:
    def test_get_usage(self):
        monitor = MemoryPressureMonitor()
        usage = monitor.get_memory_usage()
        assert 0.0 <= usage <= 1.0

    def test_pressure_level(self):
        monitor = MemoryPressureMonitor()
        level = monitor.get_pressure_level()
        assert isinstance(level, PressureLevel)


class TestLoadedModel:
    def test_memory_mb(self):
        m = LoadedModel(model_name="m", memory_bytes=10 * 1024 * 1024)
        assert m.memory_mb == 10.0

    def test_idle_seconds(self):
        import time
        m = LoadedModel(model_name="m", last_used_at=time.time() - 60)
        assert m.idle_seconds >= 59
