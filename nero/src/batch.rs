use burn::data::dataloader::batcher::Batcher;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::dataframe::{DynamicRegressionItem, Normalizer};

#[derive(Clone, Debug)]
pub struct RegressionBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1>,
}

#[derive(Clone, Debug)]
pub struct RegressionBatcher<B: Backend> {
    device: B::Device,
    normalizer: Normalizer<B>,
    features: Vec<String>,
    target: String,
}

impl<B: Backend> RegressionBatcher<B> {
    pub fn new(
        device: B::Device,
        features: Vec<String>,
        target: String,
        min: Vec<f32>,
        max: Vec<f32>,
    ) -> Self {
        Self {
            device: device.clone(),
            normalizer: Normalizer::new(&device, &min, &max),
            features,
            target,
        }
    }
}

impl<B: Backend> Batcher<DynamicRegressionItem, RegressionBatch<B>> for RegressionBatcher<B> {
    fn batch(&self, items: Vec<DynamicRegressionItem>) -> RegressionBatch<B> {
        let mut inputs: Vec<Tensor<B, 2>> = Vec::new();

        for item in items.iter() {
            // Extract features in the correct order using the helper method
            let input_values = item.features(&self.features);
            let input_tensor = Tensor::<B, 1>::from_floats(input_values.as_slice(), &self.device);
            inputs.push(input_tensor.unsqueeze());
        }

        let inputs = Tensor::cat(inputs, 0);
        let inputs = self.normalizer.normalize(inputs);

        let targets = items
            .iter()
            .map(|item| {
                let target_value = item.target(&self.target);
                Tensor::<B, 1>::from_floats([target_value], &self.device)
            })
            .collect();

        let targets = Tensor::cat(targets, 0);

        RegressionBatch { inputs, targets }
    }
}
