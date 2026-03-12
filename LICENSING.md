# Commercial Licensing — nexus

This project is dual-licensed:

- **AGPL-3.0** — Free for open-source use with copyleft obligations
- **Commercial License** — Proprietary use without AGPL requirements

## Tiers

| Feature | Community (Free) | Pro | Enterprise |
|---------|:---:|:---:|:---:|
| Base reasoning & memory | Yes | Yes | Yes |
| RAG pipeline | Yes | Yes | Yes |
| Multi-model ensemble | — | Yes | Yes |
| Advanced reasoning chains | — | Yes | Yes |
| Discovery & intelligence | — | — | Yes |
| Strategic analysis | — | — | Yes |

For pricing and tier details, visit **https://gozerai.com/pricing** or contact sales@gozerai.com.

## How It Works

- **No license key** — Community features only. Gated features return a clear error with upgrade instructions.
- **License key set** — Only entitled features are unlocked based on your tier.
- **Server unreachable** — Fail-closed for gated features.

## Getting a License

Visit **https://gozerai.com/pricing** or contact sales@gozerai.com.

```bash
export VINZY_LICENSE_KEY="your-key-here"
export VINZY_SERVER="https://api.gozerai.com"
```

## Feature Flags

Flags follow the convention `nxs.{module}.{capability}`:

| Flag | Tier | Description |
|------|------|-------------|
| `nxs.reasoning.advanced` | Pro | Advanced reasoning chains |
| `nxs.ensemble.multi_model` | Pro | Multi-model ensemble |
| `nxs.discovery.intelligence` | Enterprise | Discovery & intelligence |
| `nxs.strategic.analysis` | Enterprise | Strategic analysis |
