# Nexus Metrics Reference

**Workstream:** Create real-time dashboard
**Goal:** goal_0008 / Milestone: goal_0008_m3
**Obstacle resolved:** Missing context: metrics
**Source:** `src/nexus/core/monitoring/metrics.py`, `src/nexus/gui/views/monitoring.py`

---

## Prometheus Metrics (MetricsCollector)

All metrics are exposed via `prometheus_client` and use the `thenexus_` namespace prefix.

### Request Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `thenexus_requests_total` | Counter | `endpoint`, `method`, `status` | Total HTTP requests |
| `thenexus_request_duration_seconds` | Histogram | `endpoint`, `method` | Request latency in seconds |

### Model Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `thenexus_model_requests_total` | Counter | `model_name`, `provider` | Total requests per model |
| `thenexus_model_latency_milliseconds` | Histogram | `model_name`, `provider` | Model response latency (ms) |
| `thenexus_model_tokens_total` | Counter | `model_name`, `provider` | Total tokens consumed |
| `thenexus_model_errors_total` | Counter | `model_name`, `provider`, `error_type` | Total model-level errors |

### Cost Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `thenexus_cost_usd_total` | Counter | `model_name`, `provider` | Cumulative cost in USD |
| `thenexus_monthly_budget_usd` | Gauge | — | Configured monthly budget cap |
| `thenexus_budget_remaining_usd` | Gauge | — | Remaining budget this month |

### Cache Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `thenexus_cache_hits_total` | Counter | — | Total cache hits |
| `thenexus_cache_misses_total` | Counter | — | Total cache misses |
| `thenexus_cache_size_bytes` | Gauge | — | Current cache size |

### Ensemble Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `thenexus_ensemble_models_total` | Gauge | — | Number of active ensemble models |
| `thenexus_ensemble_score` | Histogram | `model_name` | Per-model ensemble response scores |

### System Info

| Metric | Type | Description |
|--------|------|-------------|
| `thenexus` (Info) | Info | Version, Python version, available models |

---

## GUI Monitoring View

The existing `MonitoringView` (`src/nexus/gui/views/monitoring.py`) renders:

- **MetricCard** widgets — display a single metric value with a trend indicator (↑/↓)
- **AlertItem** widgets — severity-tagged alerts (low / medium / high) with dismiss support
- Tab-grouped panels covering drift detection, performance, cost, and system health
- `QTimer`-driven refresh loop for live updates

### Dashboard Panels (existing)

1. **Drift Detection** — model behaviour drift alerts
2. **Performance Metrics** — request latency and throughput
3. **Cost Tracking** — spend vs budget gauges
4. **System Health** — error rates and cache efficiency

---

## Derived / Computed Metrics for Dashboard Design

These are not raw Prometheus series but are useful KPIs to surface on a real-time dashboard:

| KPI | Formula |
|-----|---------|
| Cache hit rate | `cache_hits / (cache_hits + cache_misses)` |
| Error rate | `model_errors / model_requests` |
| Budget utilisation % | `(budget - remaining) / budget × 100` |
| Avg model latency | derived from `model_latency_milliseconds` histogram |
| Cost per token | `cost_usd_total / model_tokens_total` (per model/provider) |

---

## Integration Notes

- `MetricsCollector` accepts an optional `CollectorRegistry`; pass a shared registry when wiring to a Prometheus HTTP server or Pushgateway.
- `record_model_request()` is the primary call-site to instrument — covers latency, tokens, cost, and error in a single call.
- `update_budget_metrics()` must be called periodically (e.g. on each billing poll) to keep gauge values fresh.
- Drift monitoring is handled separately by `src/nexus/providers/monitoring/drift_monitor.py`.
