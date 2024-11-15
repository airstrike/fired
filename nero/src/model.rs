use burn::module::Module;
use burn::nn::{
    loss::{MseLoss, Reduction::Mean},
    Linear, LinearConfig, Relu,
};
use burn::prelude::*;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::train::{RegressionOutput, TrainOutput, TrainStep, ValidStep};

use crate::batch::RegressionBatch;

#[derive(Module, Debug)]
pub struct RegressionModel<B: Backend> {
    input_layer: Linear<B>,
    output_layer: Linear<B>,
    activation: Relu,
}

#[derive(Config)]
pub struct RegressionModelConfig {
    input_size: usize,
    hidden_size: usize,
}

impl RegressionModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RegressionModel<B> {
        let input_layer = LinearConfig::new(self.input_size, self.hidden_size)
            .with_bias(true)
            .init(device);
        let output_layer = LinearConfig::new(self.hidden_size, 1)
            .with_bias(true)
            .init(device);

        RegressionModel {
            input_layer,
            output_layer,
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> RegressionModel<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.input_layer.forward(input);
        let x = self.activation.forward(x);
        self.output_layer.forward(x)
    }

    pub fn forward_step(&self, item: RegressionBatch<B>) -> RegressionOutput<B> {
        let targets: Tensor<B, 2> = item.targets.unsqueeze_dim(1);
        let output: Tensor<B, 2> = self.forward(item.inputs);

        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Mean);

        RegressionOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<RegressionBatch<B>, RegressionOutput<B>> for RegressionModel<B> {
    fn step(&self, item: RegressionBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_step(item);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<RegressionBatch<B>, RegressionOutput<B>> for RegressionModel<B> {
    fn step(&self, item: RegressionBatch<B>) -> RegressionOutput<B> {
        self.forward_step(item)
    }
}
