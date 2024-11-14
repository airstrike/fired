use burn_dataset::{DataframeDataset, DataframeDatasetError};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DynamicRegressionItem {
    pub features: HashMap<String, f32>,
    pub target: f32,
}

pub struct RegressionConfig {
    pub features: Vec<String>,
    pub target: String,
}

#[derive(Error, Debug)]
pub enum DatasetError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Polars error: {0}")]
    Polars(#[from] PolarsError),

    #[error("Dataset error: {0}")]
    Dataset(#[from] DataframeDatasetError),
}

pub async fn load_dataset_from_path(
    path: impl Into<PathBuf>,
    config: RegressionConfig,
) -> Result<(DataframeDataset<DynamicRegressionItem>, usize), DatasetError> {
    let path = path.into();

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(path))?
        .finish()?;

    create_dataset(df, config)
}

fn create_dataset(
    df: DataFrame,
    config: RegressionConfig,
) -> Result<(DataframeDataset<DynamicRegressionItem>, usize), DatasetError> {
    // Validate columns exist
    for col in &config.features {
        if !df.schema().contains(col) {
            return Err(DatasetError::Polars(PolarsError::ColumnNotFound(
                col.to_owned().into(),
            )));
        }
    }
    if !df.schema().contains(&config.target) {
        return Err(DatasetError::Polars(PolarsError::ColumnNotFound(
            config.target.to_owned().into(),
        )));
    }

    Ok((DataframeDataset::new(df)?, config.features.len()))
}

#[derive(Clone, Debug)]
pub struct Normalizer<B: burn::tensor::backend::Backend> {
    pub min: burn::tensor::Tensor<B, 2>,
    pub max: burn::tensor::Tensor<B, 2>,
}

impl<B: burn::tensor::backend::Backend> Normalizer<B> {
    pub fn new(device: &B::Device, min: &[f32], max: &[f32]) -> Self {
        let min = burn::tensor::Tensor::<B, 1>::from_floats(min, device).unsqueeze();
        let max = burn::tensor::Tensor::<B, 1>::from_floats(max, device).unsqueeze();
        Self { min, max }
    }

    pub fn normalize(&self, input: burn::tensor::Tensor<B, 2>) -> burn::tensor::Tensor<B, 2> {
        (input - self.min.clone()) / (self.max.clone() - self.min.clone())
    }
}
