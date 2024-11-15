use nero::*;
use tokio;

fn relative(path: &str) -> std::path::PathBuf {
    format!("{}/examples/{}", env!("CARGO_MANIFEST_DIR"), path).into()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if let Err(error) = log::init() {
        eprintln!("examples/data: failed to initialize logger: {}", error);
    }

    // Create a new project with the following configuration
    let mut project = Project::new(200, 16, 64) // epochs, batch_size, hidden_size
        .with_data_path(relative("data.txt"))
        .with_features(vec![
            "Con_1".into(),
            "Con_2".into(),
            "Con_3".into(),
            "Con_4".into(),
        ])
        .with_seed(125591)
        .with_target("Rejects".into())
        .with_learning_rate(0.001)
        .with_weight_decay(0.00001)
        .with_artifact_dir(relative("artifacts").into());

    // Load data, split data and check that we're ready to run
    project.load_data().await?;
    project.split_data(&[0.8, 0.10, 0.10])?;
    project.validate()?;

    if let Some(df) = project.source_data() {
        log::info!("Original data shape: {:?}", df.shape());
    }

    // Run training/inference
    let summary = project.run().await?;

    // Save results and create visualizations
    let artifact_dir = project.artifact_dir().unwrap();
    summary.save_predictions(artifact_dir)?;

    let hyperparameters = project.hyperparameters();

    summary.plot(artifact_dir, hyperparameters)?;
    summary.print_metrics();

    Ok(())
}
