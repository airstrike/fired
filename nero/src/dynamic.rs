use burn_dataset::{DataframeDatasetError, Dataset};
use polars::frame::row::Row;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DatasetError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Polars error: {0}")]
    Polars(#[from] PolarsError),

    #[error("Dataset error: {0}")]
    Dataset(String),

    #[error("DataframeDataset error: {0}")]
    DataframeDataset(#[from] DataframeDatasetError),
}
/// A simplified wrapper for numeric values that can be parsed from DataFrame columns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericVector {
    pub values: Vec<f32>,
}

impl NumericVector {
    fn from_row(row: &Row, column_indices: &[usize]) -> Result<Self, DatasetError> {
        log::trace!("Raw row data: {:?}", row.0);
        let values = column_indices
            .iter()
            .map(|&idx| {
                let val = match &row.0[idx] {
                    AnyValue::Float32(f) => Ok(*f),
                    AnyValue::Float64(f) => Ok(*f as f32),
                    AnyValue::Int32(i) => Ok(*i as f32),
                    AnyValue::Int64(i) => Ok(*i as f32),
                    AnyValue::UInt32(i) => Ok(*i as f32),
                    AnyValue::UInt64(i) => Ok(*i as f32),
                    other => Err(DatasetError::Dataset(format!(
                        "Unsupported type for numeric conversion: {other:?}"
                    ))),
                };
                log::trace!("Column {}: {:?}", idx, val);
                val
            })
            .collect::<Result<Vec<_>, _>>()?;

        log::trace!("Converted values: {:?}", values);
        Ok(NumericVector { values })
    }
}

#[derive(Clone)]
pub struct ColumnInfo {
    pub names: Vec<String>,
}

/// Dataset implementation for Polars DataFrame with dynamic field support
#[derive(Clone)]
pub struct FlexibleDataset<I> {
    df: DataFrame,
    len: usize,
    column_indices: Vec<usize>,
    column_names: Vec<String>,
    _phantom: PhantomData<I>,
}

impl<I: From<(NumericVector, Vec<String>)> + Clone + Send + Sync> FlexibleDataset<I> {
    pub fn new(df: DataFrame, columns: &[String]) -> Result<Self, DatasetError> {
        let len = df.height();
        let column_indices = columns
            .iter()
            .map(|col| {
                df.schema()
                    .try_get_full(col)
                    .map(|(idx, _, _)| idx)
                    .map_err(|err| DatasetError::Polars(err.into()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(FlexibleDataset {
            df,
            len,
            column_indices,
            column_names: columns.to_vec(),
            _phantom: PhantomData,
        })
    }
}

impl<I: From<(NumericVector, Vec<String>)> + Clone + Send + Sync> Dataset<I>
    for FlexibleDataset<I>
{
    fn get(&self, index: usize) -> Option<I> {
        self.df
            .get_row(index)
            .ok()
            .and_then(|row| NumericVector::from_row(&row, &self.column_indices).ok())
            .map(|vec| I::from((vec, self.column_names.clone())))
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }
}
