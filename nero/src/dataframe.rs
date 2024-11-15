use std::collections::HashMap;

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use polars::prelude::*;

use crate::dynamic::{DatasetError, FlexibleDataset, NumericVector};

#[derive(Clone, Debug)]
pub struct DynamicRegressionItem {
    values: HashMap<String, f32>,
}

impl DynamicRegressionItem {
    pub fn features(&self, feature_names: &[String]) -> Vec<f32> {
        feature_names
            .iter()
            .map(|name| *self.values.get(name).unwrap_or(&0.0))
            .collect()
    }

    pub fn target(&self, target_name: &str) -> f32 {
        *self.values.get(target_name).unwrap_or(&0.0)
    }
}

// Modify the From implementation to use the column names
impl From<(NumericVector, Vec<String>)> for DynamicRegressionItem {
    fn from((vec, names): (NumericVector, Vec<String>)) -> Self {
        let values = vec
            .values
            .into_iter()
            .zip(names.into_iter())
            .map(|(v, name)| (name, v))
            .collect();
        DynamicRegressionItem { values }
    }
}

pub struct RegressionConfig {
    pub features: Vec<String>,
    pub target: String,
}

pub fn create_dataset(
    df: DataFrame,
    config: RegressionConfig,
) -> Result<(FlexibleDataset<DynamicRegressionItem>, usize), DatasetError> {
    // Validate columns exist
    for col in &config.features {
        if !df.schema().contains(col) {
            return Err(DatasetError::Dataset(format!("Column not found: {}", col)));
        }
    }
    if !df.schema().contains(&config.target) {
        return Err(DatasetError::Dataset(format!(
            "Target column not found: {}",
            config.target
        )));
    }

    // Create columns vector with all needed columns
    let mut cols = Vec::new();
    cols.extend(config.features.iter().cloned());
    cols.push(config.target.clone());

    // Only keep the columns we need
    let df = df.select(&cols).map_err(|e| DatasetError::Polars(e))?;

    // Create the dataset
    Ok((FlexibleDataset::new(df, &cols)?, config.features.len()))
}

#[derive(Clone, Debug)]
pub struct Normalizer<B: Backend> {
    pub min: Tensor<B, 2>,
    pub max: Tensor<B, 2>,
}

impl<B: Backend> Normalizer<B> {
    pub fn new(device: &B::Device, min: &[f32], max: &[f32]) -> Self {
        let min = Tensor::<B, 1>::from_floats(min, device).unsqueeze();
        let max = Tensor::<B, 1>::from_floats(max, device).unsqueeze();
        Self { min, max }
    }

    pub fn normalize(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        (input - self.min.clone()) / (self.max.clone() - self.min.clone())
    }
}

pub fn calculate_min_max(
    df: &DataFrame,
    features: &[String],
) -> Result<(Vec<f32>, Vec<f32>), DatasetError> {
    let mut min_values = Vec::with_capacity(features.len());
    let mut max_values = Vec::with_capacity(features.len());

    for feature in features {
        let col = df.column(feature)?;
        let min = col.min::<f32>()?.unwrap_or(0.0);
        let max = col.max::<f32>()?.unwrap_or(1.0);
        min_values.push(min);
        max_values.push(max);
    }

    Ok((min_values, max_values))
}

// Optional: Helper trait to make working with column names easier
pub trait NamedColumns {
    fn get_value(&self, name: &str) -> Option<f32>;
    fn set_value(&mut self, name: String, value: f32);
}

impl NamedColumns for DynamicRegressionItem {
    fn get_value(&self, name: &str) -> Option<f32> {
        self.values.get(name).copied()
    }

    fn set_value(&mut self, name: String, value: f32) {
        self.values.insert(name, value);
    }
}
