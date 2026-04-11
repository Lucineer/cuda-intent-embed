# cuda-intent-embed

Intent embedding system — human text to structured intent vectors for A2A deliberation

Part of the Cocapn fleet — a Lucineer vessel component.

## What It Does

### Key Types

- `IntentVector` — core data structure
- `IntentParser` — core data structure
- `IntentCache` — core data structure

## Quick Start

```bash
# Clone
git clone https://github.com/Lucineer/cuda-intent-embed.git
cd cuda-intent-embed

# Build
cargo build

# Run tests
cargo test
```

## Usage

```rust
use cuda_intent_embed::*;

// See src/lib.rs for full API
// 11 unit tests included
```

### Available Implementations

- `Domain` — see source for methods
- `IntentVector` — see source for methods
- `IntentParser` — see source for methods
- `Default for IntentParser` — see source for methods
- `IntentCache` — see source for methods

## Testing

```bash
cargo test
```

11 unit tests covering core functionality.

## Architecture

This crate is part of the **Cocapn Fleet** — a git-native multi-agent ecosystem.

- **Category**: other
- **Language**: Rust
- **Dependencies**: See `Cargo.toml`
- **Status**: Active development

## Related Crates


## Fleet Position

```
Casey (Captain)
├── JetsonClaw1 (Lucineer realm — hardware, low-level systems, fleet infrastructure)
├── Oracle1 (SuperInstance — lighthouse, architecture, consensus)
└── Babel (SuperInstance — multilingual scout)
```

## Contributing

This is a fleet vessel component. Fork it, improve it, push a bottle to `message-in-a-bottle/for-jetsonclaw1/`.

## License

MIT

---

*Built by JetsonClaw1 — part of the Cocapn fleet*
*See [cocapn-fleet-readme](https://github.com/Lucineer/cocapn-fleet-readme) for the full fleet roadmap*
