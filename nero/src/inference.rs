use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    module::Module,
    record::{NoStdTrainingRecorder, Recorder},
    tensor::backend::Backend,
};
use rgb::RGB8;
use textplots::{Chart, ColorPlot, Shape};

use crate::{
    batch::RegressionBatcher,
    dataframe::DynamicRegressionItem,
    dynamic::FlexibleDataset,
    model::{RegressionModelConfig, RegressionModelRecord},
};

pub fn infer<B: Backend>(
    test_dataset: FlexibleDataset<DynamicRegressionItem>,
    features: Vec<String>,
    target: String,
    device: B::Device,
    artifact_dir: &str,
    min_values: Vec<f32>,
    max_values: Vec<f32>,
    hidden_size: usize,
) -> Vec<(f32, f32)> {
    let record: RegressionModelRecord<B> = NoStdTrainingRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    let model = RegressionModelConfig::new(features.len(), hidden_size)
        .init(&device)
        .load_record(record);

    // Process in smaller batches
    let batch_size = 32;
    let mut all_points = Vec::new();

    // Create batches
    let items: Vec<DynamicRegressionItem> = test_dataset.iter().collect();
    for chunk in items.chunks(batch_size) {
        let batcher = RegressionBatcher::new(
            device.clone(),
            features.clone(),
            target.clone(),
            min_values.clone(),
            max_values.clone(),
        );

        let batch = batcher.batch(chunk.to_vec());
        let predicted = model.forward(batch.inputs);
        let targets = batch.targets;

        let predicted = predicted.squeeze::<1>(1).into_data();
        let expected = targets.into_data();

        let points: Vec<(f32, f32)> = predicted
            .iter::<f32>()
            .zip(expected.iter::<f32>())
            .collect();

        all_points.extend(points);
    }

    log::info!("Total predictions: {}", all_points.len());

    // Calculate some basic statistics
    let mse: f32 = all_points
        .iter()
        .map(|(pred, actual)| (pred - actual).powi(2))
        .sum::<f32>()
        / all_points.len() as f32;

    log::info!("Mean Squared Error: {:.6}", mse);

    // Plot all predictions
    Chart::new_with_y_range(120, 60, 0., 5., 0., 5.)
        .linecolorplot(
            &Shape::Points(&all_points),
            RGB8 {
                r: 255,
                g: 85,
                b: 85,
            },
        )
        .display();

    // Show some sample predictions
    log::info!("Sample predictions (Predicted vs Expected):");
    for (i, (pred, actual)) in all_points.iter().take(5).enumerate() {
        log::info!("Sample {}: {:.3} vs {:.3}", i + 1, pred, actual);
    }
    log::info!("");

    all_points
}
