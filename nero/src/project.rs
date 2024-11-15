use rand::Rng;
use std::path::PathBuf;

use burn::LearningRate;
use polars::prelude::*;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::dataframe::*;
use crate::dynamic::{DatasetError, FlexibleDataset};
use crate::summary::RegressionSummary;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ProjectError {
    #[error("Data path not set")]
    MissingDataPath,
    #[error("Artifact directory not set")]
    MissingArtifactDir,
    #[error("Features not configured")]
    MissingFeatures,
    #[error("Target column not configured")]
    MissingTarget,
    #[error("Dataset error: {0}")]
    Dataset(#[from] DatasetError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Other error: {0}")]
    Other(String),
}

#[derive(Serialize, Deserialize)]
pub struct Project {
    // Data configuration
    data_path: Option<PathBuf>,
    features: Option<Vec<String>>,
    target: Option<String>,
    artifact_dir: Option<PathBuf>,

    // Training configuration
    num_epochs: usize,
    num_workers: usize,
    seed: u64,
    batch_size: usize,
    hidden_size: usize,
    weight_decay: f32,
    learning_rate: LearningRate,

    // Runtime state (not serialized)
    #[serde(skip)]
    source_data: Option<DataFrame>,
    #[serde(skip)]
    split_points: Vec<(Vec<usize>, usize, usize)>,
}

impl Project {
    pub fn new(num_epochs: usize, batch_size: usize, hidden_size: usize) -> Self {
        let seed = rand::thread_rng().gen_range(0..u64::MAX);
        Self {
            data_path: None,
            features: None,
            target: None,
            artifact_dir: None,
            num_epochs,
            num_workers: 1,
            seed,
            batch_size,
            hidden_size,
            weight_decay: 1e-5,
            learning_rate: 0.1,
            source_data: None,
            split_points: Vec::new(),
        }
    }

    pub fn with_data_path(mut self, path: PathBuf) -> Self {
        self.data_path = Some(path);
        self
    }

    pub fn with_features(mut self, features: Vec<String>) -> Self {
        self.features = Some(features);
        self
    }

    pub fn with_target(mut self, target: String) -> Self {
        self.target = Some(target);
        self
    }

    pub fn with_artifact_dir(mut self, path: PathBuf) -> Self {
        self.artifact_dir = Some(path);
        self
    }

    pub fn with_num_epochs(mut self, num_epochs: usize) -> Self {
        self.num_epochs = num_epochs;
        self
    }

    pub fn with_num_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = num_workers;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn with_hidden_size(mut self, hidden_size: usize) -> Self {
        self.hidden_size = hidden_size;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_learning_rate(mut self, learning_rate: LearningRate) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    // Helper method to get features safely
    pub fn features(&self) -> Result<&Vec<String>, ProjectError> {
        self.features.as_ref().ok_or(ProjectError::MissingFeatures)
    }

    // Helper method to get target safely
    pub fn target(&self) -> Result<&String, ProjectError> {
        self.target.as_ref().ok_or(ProjectError::MissingTarget)
    }

    pub fn data_path(&self) -> Option<&PathBuf> {
        self.data_path.as_ref()
    }

    pub fn artifact_dir(&self) -> Option<&PathBuf> {
        self.artifact_dir.as_ref()
    }

    pub fn validate(&self) -> Result<(), ProjectError> {
        if self.data_path.is_none() {
            return Err(ProjectError::MissingDataPath);
        }
        if self.artifact_dir.is_none() {
            return Err(ProjectError::MissingArtifactDir);
        }
        if self.features.is_none() {
            return Err(ProjectError::MissingFeatures);
        }
        if self.target.is_none() {
            return Err(ProjectError::MissingTarget);
        }

        // Validate that paths exist and are accessible
        if let Some(path) = &self.data_path {
            if !path.exists() {
                return Err(ProjectError::Other(format!(
                    "Data path does not exist: {:?}",
                    path
                )));
            }
        }
        if let Some(path) = &self.artifact_dir {
            // Try to create the artifact directory if it doesn't exist
            std::fs::create_dir_all(path).map_err(|e| {
                ProjectError::Other(format!("Failed to create artifact directory: {}", e))
            })?;

            // Test write permissions by creating a temporary file
            let test_file = path.join(".test_write");
            std::fs::write(&test_file, "test").map_err(|e| {
                ProjectError::Other(format!("Artifact directory is not writable: {}", e))
            })?;
            let _ = std::fs::remove_file(test_file);
        }

        Ok(())
    }

    pub async fn load_data(&mut self) -> Result<(), ProjectError> {
        self.validate()?;

        let data_path = self.data_path.as_ref().unwrap();
        self.source_data = Some(
            LazyCsvReader::new(data_path)
                .with_has_header(true)
                .with_separator(b'\t')
                .finish()
                .map_err(|e| ProjectError::Other(format!("Failed to read CSV: {}", e)))?
                .collect()
                .map_err(|e| ProjectError::Other(format!("Failed to collect DataFrame: {}", e)))?,
        );
        Ok(())
    }

    pub fn split_data(&mut self, ratios: &[f32]) -> Result<(), DatasetError> {
        let df = self.source_data.as_ref().ok_or_else(|| {
            DatasetError::Polars(PolarsError::ComputeError("No source data loaded".into()))
        })?;

        // Create a random permutation of row indices based on the seed
        let n_rows = df.height();
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let mut indices: Vec<usize> = (0..n_rows).collect();
        indices.shuffle(&mut rng);

        // Clear existing split points
        self.split_points.clear();

        // Calculate split points using cumulative ratios
        let total: f32 = ratios.iter().sum();
        let mut cumsum = 0.0;
        let mut start_idx = 0;

        for &ratio in ratios {
            let portion = ratio / total;
            cumsum += portion;
            let end_idx = (n_rows as f32 * cumsum) as usize;
            self.split_points
                .push((indices[start_idx..end_idx].to_vec(), start_idx, end_idx));
            start_idx = end_idx;
        }

        Ok(())
    }

    pub fn create_dataset(
        &self,
        idx: usize,
    ) -> Result<FlexibleDataset<DynamicRegressionItem>, ProjectError> {
        let df = self.source_data.as_ref().ok_or_else(|| {
            ProjectError::Other("No source data loaded. Call load_data() first.".into())
        })?;

        let features = self
            .features
            .as_ref()
            .ok_or(ProjectError::MissingFeatures)?;
        let target = self.target.as_ref().ok_or(ProjectError::MissingTarget)?;

        let (indices, _, _) = self.split_points.get(idx).ok_or_else(|| {
            ProjectError::Other(format!(
                "Invalid split index: {}. Call split_data() first.",
                idx
            ))
        })?;

        // Create a UInt32Chunked array for the indices
        let idx_ca = UInt32Chunked::new(
            "idx",
            indices.iter().map(|&idx| idx as u32).collect::<Vec<_>>(),
        );

        // Take rows using the chunked array
        let split_df = df.take(&idx_ca).map_err(|e| {
            ProjectError::Other(format!("Failed to take rows from DataFrame: {}", e))
        })?;

        let (dataset, _) = create_dataset(
            split_df,
            RegressionConfig {
                features: features.clone(),
                target: target.clone(),
            },
        )?;

        Ok(dataset)
    }

    pub fn source_data(&self) -> Option<&DataFrame> {
        self.source_data.as_ref()
    }

    pub fn num_features(&self) -> Result<usize, ProjectError> {
        self.features
            .as_ref()
            .map(|features| features.len())
            .ok_or(ProjectError::MissingFeatures)
    }

    fn get_training_config(&self) -> crate::training::TrainingConfig {
        crate::training::TrainingConfig::new(
            self.num_epochs,
            self.num_workers,
            self.seed,
            self.batch_size,
            self.hidden_size,
            self.weight_decay,
            self.learning_rate,
        )
    }

    pub fn hyperparameters(&self) -> String {
        format!(
            "Epochs: {}, Workers: {}, Seed: {}, Batch Size: {}, Hidden Size: {}, Learning Rate: {:.3}, Weight Decay: {:.5}",
            self.num_epochs, self.num_workers, self.seed, self.batch_size, self.hidden_size, self.learning_rate, self.weight_decay
        )
    }

    pub async fn run(&mut self) -> Result<RegressionSummary, ProjectError> {
        self.validate()?;

        use burn::backend::wgpu::{Wgpu, WgpuDevice};
        use burn::backend::Autodiff;

        // Load data if not already loaded
        if self.source_data.is_none() {
            self.load_data().await?;
        }

        // Split into train/test if not already split
        if self.split_points.is_empty() {
            self.split_data(&[0.8, 0.2])?;
        }

        type Backend = Autodiff<Wgpu>;
        let device = WgpuDevice::default();

        let train_dataset = self.create_dataset(0)?;
        let test_dataset = self.create_dataset(1)?;
        let infer_test_dataset = self.create_dataset(1)?;

        let features = self.features.as_ref().unwrap();
        let target = self.target.as_ref().unwrap();
        let artifact_dir = self.artifact_dir.as_ref().unwrap();

        // Calculate min/max values
        let (min_values, max_values) =
            calculate_min_max(self.source_data.as_ref().unwrap(), features)?;

        // Print configuration
        log::info!("Features:    {:?}", features);
        log::info!("Target:      {:?}", target);
        log::info!("Artifacts:   {:?}", artifact_dir);
        log::info!("Num Epochs:  {:?}", self.num_epochs);
        log::info!("Num Workers: {:?}", self.num_workers);
        log::info!("Seed:        {:?}", self.seed);
        log::info!("Batch Size:  {:?}", self.batch_size);
        log::info!("Hidden Size: {:?}", self.hidden_size);
        log::info!("Min Values:  {:?}", min_values);
        log::info!("Max Values:  {:?}", max_values);

        // Train
        let train_predictions = crate::training::run::<Backend>(
            train_dataset,
            test_dataset,
            features.clone(),
            target.clone(),
            device.clone(),
            self.get_training_config(),
            artifact_dir.to_str().unwrap(),
            min_values.clone(),
            max_values.clone(),
        );

        // Get test predictions
        let test_predictions = crate::inference::infer::<Wgpu>(
            infer_test_dataset,
            features.clone(),
            target.clone(),
            device,
            artifact_dir.to_str().unwrap(),
            min_values,
            max_values,
            self.hidden_size,
        );

        Ok(RegressionSummary::new(train_predictions, test_predictions))
    }
}
