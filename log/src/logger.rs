use std::{fs::File, sync::Arc};
use tracing::Level;
use tracing_subscriber::{
    filter::{LevelFilter, Targets},
    fmt,
    prelude::*,
};

pub use tracing::{debug, error, info, trace, warn};

// From the excellent example in:
// https://docs.rs/tracing-subscriber/latest/tracing_subscriber/filter/targets/struct.Targets.html
pub fn init() -> Result<(), std::io::Error> {
    // A layer that logs events to stdout using the human-readable "pretty"
    // format.
    let stdout_log = fmt::layer().compact().without_time().with_target(true);

    // A layer that logs events to a file, using the JSON format.
    let file = File::create("debug_log.jsonl")?;
    let debug_log = fmt::layer().with_writer(Arc::new(file)).compact();

    tracing_subscriber::registry()
        // Only log INFO and above to stdout, unless the span or event
        // has the `my_crate::cool_module` target prefix.
        .with(
            stdout_log.with_filter(
                Targets::default()
                    .with_target("fired::nero", Level::DEBUG)
                    .with_target("fired", Level::DEBUG)
                    .with_default(Level::DEBUG),
            ),
        )
        // Log everything enabled by the global filter to `debug_log.json`.
        .with(debug_log)
        // Configure a global filter for the whole subscriber stack. This will
        // control what spans and events are recorded by both the `debug_log`
        // and the `stdout_log` layers, and `stdout_log` will *additionally* be
        // filtered by its per-layer filter.
        .with(
            Targets::default()
                .with_target("iced", Level::WARN)
                .with_target("iced_wgpu", Level::WARN)
                .with_target("burn_train", LevelFilter::OFF)
                .with_target("burn_fusion", LevelFilter::OFF)
                .with_target("cubecl_wgpu", LevelFilter::OFF)
                .with_target("cubecl_runtime", LevelFilter::OFF)
                .with_target("wgpu_core", LevelFilter::OFF)
                .with_target("cosmic_text", LevelFilter::OFF)
                .with_default(Level::INFO),
        )
        .init();

    Ok(())
}
