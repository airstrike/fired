use burn::optim::decay::WeightDecayConfig;
use burn::optim::AdamConfig;
use burn::LearningRate;
use burn::{
    data::dataloader::batcher::Batcher,
    data::dataloader::DataLoaderBuilder,
    data::dataset::Dataset as _,
    prelude::*,
    record::{CompactRecorder, NoStdTrainingRecorder},
    tensor::backend::AutodiffBackend,
    train::{metric::LossMetric, LearnerBuilder},
};

use crate::batch::RegressionBatcher;
use crate::dataframe::DynamicRegressionItem;
use crate::dynamic::FlexibleDataset;
use crate::model::RegressionModelConfig;

pub struct TrainingConfig {
    num_epochs: usize,
    num_workers: usize,
    seed: u64,
    optimizer: AdamConfig,
    batch_size: usize,
    hidden_size: usize,
    learning_rate: LearningRate,
}

impl TrainingConfig {
    pub fn new(
        num_epochs: usize,
        num_workers: usize,
        seed: u64,
        batch_size: usize,
        hidden_size: usize,
        weight_decay: f32,
        learning_rate: LearningRate,
    ) -> Self {
        Self {
            num_epochs,
            num_workers: num_workers.max(1), // Ensure at least 1 worker
            seed,
            batch_size,
            hidden_size,
            optimizer: AdamConfig::new()
                .with_weight_decay(Some(WeightDecayConfig::new(weight_decay as f64))),
            learning_rate,
        }
    }
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn run<B: AutodiffBackend>(
    train_dataset: FlexibleDataset<DynamicRegressionItem>,
    valid_dataset: FlexibleDataset<DynamicRegressionItem>,
    features: Vec<String>,
    target: String,
    device: B::Device,
    config: TrainingConfig,
    artifact_dir: &str,
    min_values: Vec<f32>,
    max_values: Vec<f32>,
) -> Vec<(f32, f32)> {
    create_artifact_dir(artifact_dir);

    let model = RegressionModelConfig::new(features.len(), config.hidden_size).init(&device);
    B::seed(config.seed);

    log::info!("Train Dataset Size: {}", train_dataset.len());
    log::info!("Valid Dataset Size: {}", valid_dataset.len());

    let batcher_train = super::batch::RegressionBatcher::<B>::new(
        device.clone(),
        features.clone(),
        target.clone(),
        min_values.clone(),
        max_values.clone(),
    );

    let batcher_test = super::batch::RegressionBatcher::<B::InnerBackend>::new(
        device.clone(),
        features.clone(),
        target.clone(),
        min_values.clone(),
        max_values.clone(),
    );

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset.clone());

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(valid_dataset);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, config.optimizer.init(), config.learning_rate);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .clone()
        .save_file(
            format!("{artifact_dir}/model"),
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save trained model");

    // Get training predictions
    let train_predictions = {
        let items: Vec<DynamicRegressionItem> = train_dataset.iter().collect();
        let mut predictions = Vec::new();

        for chunk in items.chunks(config.batch_size) {
            let batcher = RegressionBatcher::new(
                device.clone(),
                features.clone(),
                target.clone(),
                min_values.clone(),
                max_values.clone(),
            );

            let batch = batcher.batch(chunk.to_vec());
            let predicted = model_trained.forward(batch.inputs);
            let targets = batch.targets;

            let predicted = predicted.squeeze::<1>(1).into_data();
            let expected = targets.into_data();

            predictions.extend(
                predicted
                    .iter::<f32>()
                    .zip(expected.iter::<f32>())
                    .map(|(pred, actual)| (actual, pred)),
            );
        }

        predictions
    };

    train_predictions
}

impl std::fmt::Debug for TrainingConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainingConfig")
            .field("num_epochs", &self.num_epochs)
            .field("num_workers", &self.num_workers)
            .field("seed", &self.seed)
            .field("optimizer", &"AdamConfig".to_string())
            .field("batch_size", &self.batch_size)
            .field("hidden_size", &self.hidden_size)
            .finish()
    }
}
