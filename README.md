# fired 🔥

A flexible neural network regression implementation using [burn](https://github.com/burn-rs/burn).

## Overview

`fired` provides a high-level interface to `nero`, a Rust library for building and training neural network regression models with dynamic feature selection. It provides a clean API for loading data, configuring models, training, and evaluating results.

Key features:
- Dynamic feature selection and column mapping
- Configurable neural network architecture
- Real-time training visualization
- Comprehensive result analysis and plotting
- Built on burn's efficient deep learning primitives
- Polars integration for fast data processing

## Project Structure

```
fired/
├── nero/           # Core library implementation
├── log/            # Logging utilities
└── examples/       # Example applications
    ├── data.rs     # Sample regression problem
    └── data.txt    # Sample dataset
```

## Installation

Add `fired` to your `Cargo.toml`:

```toml
[dependencies]
fired = { git = "https://github.com/airstrike/fired.git" }
```

Or to use the core library directly:

```toml
[dependencies]
nero = { git = "https://github.com/airstrike/fired.git", package = "nero" }
```

## Quick Start

```rust
use nero::*;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut project = Project::new()
        .with_data_path("data.txt")
        .with_features(vec![
            "feature1".into(),
            "feature2".into(),
        ])
        .with_target("output".into())
        .with_artifact_dir("artifacts".into());

    // Load and split data
    project.load_data().await?;
    project.split_data(&[0.8, 0.2])?;

    // Train model and get predictions
    let summary = project.run().await?;

    // Generate visualizations and metrics
    summary.plot(&project.artifact_dir.unwrap())?;
    summary.print_metrics();

    Ok(())
}
```

## Example

The repository includes a sample regression problem in the `examples` directory. Run it with:

```bash
cargo run --example data
```

This trains a neural network to predict manufacturing process outcomes, demonstrating the library's real-world regression capabilities. The example includes real-time training visualization and generates detailed performance metrics and plots.

## Development

Building:
```bash
cargo build
```

Running tests:
```bash
cargo test
```

Running benchmarks:
```bash
cargo bench
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with:
- [burn](https://github.com/burn-rs/burn) - Deep learning framework
- [polars](https://github.com/pola-rs/polars) - Fast DataFrame library
- [plotters](https://github.com/plotters-rs/plotters) - Plotting library

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
