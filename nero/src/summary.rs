use plotters::prelude::*;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct RegressionSummary {
    pub train_predictions: Vec<(f32, f32)>,
    pub test_predictions: Vec<(f32, f32)>,
    pub train_sse: f32,
    pub test_sse: f32,
}

impl RegressionSummary {
    pub fn new(train_results: Vec<(f32, f32)>, test_results: Vec<(f32, f32)>) -> Self {
        // Calculate SSE for both sets
        let train_sse = train_results
            .iter()
            .map(|(actual, pred)| (pred - actual).powi(2))
            .sum();

        let test_sse = test_results
            .iter()
            .map(|(actual, pred)| (pred - actual).powi(2))
            .sum();

        Self {
            train_predictions: train_results,
            test_predictions: test_results,
            train_sse,
            test_sse,
        }
    }

    pub fn save_predictions(&self, artifact_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        // Save training predictions
        let mut train_file = File::create(artifact_dir.join("train_predictions.csv"))?;
        writeln!(train_file, "actual,predicted")?;
        for (actual, pred) in &self.train_predictions {
            writeln!(train_file, "{},{}", actual, pred)?;
        }

        // Save test predictions
        let mut test_file = File::create(artifact_dir.join("test_predictions.csv"))?;
        writeln!(test_file, "actual,predicted")?;
        for (actual, pred) in &self.test_predictions {
            writeln!(test_file, "{},{}", actual, pred)?;
        }

        Ok(())
    }

    pub fn plot(
        &self,
        artifact_dir: &PathBuf,
        hyperparams: String,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let output_path = artifact_dir.join("predictions.svg");
        let root = SVGBackend::new(&output_path, (1024, 612)).into_drawing_area();

        root.fill(&WHITE)?;

        // First split horizontally to get title area and main area
        let (title_area, main_area) = root.split_vertically(76);

        // Then split the main area horizontally for the two plots
        let (train_area, test_area) = main_area.split_horizontally(512);

        // Draw hyperparameters text in title area
        title_area.draw(&Text::new(
            hyperparams,
            (10, 30),
            ("sans-serif", 20).into_font(),
        ))?;

        // Helper function to calculate all metrics in one pass
        fn calculate_metrics(data: &[(f32, f32)]) -> (f64, f64, f64) {
            let n = data.len() as f64;
            let mut sum_actual = 0.0;
            let mut sum_squared_actual = 0.0;
            let mut sum_squared_error = 0.0;
            let mut sum_abs_error = 0.0;

            for (actual, pred) in data {
                let a = *actual as f64;
                let p = *pred as f64;
                sum_actual += a;
                sum_squared_actual += a * a;

                let error = p - a;
                sum_squared_error += error * error;
                sum_abs_error += error.abs();
            }

            // Calculate total sum of squares (TSS)
            let tss = sum_squared_actual - (sum_actual * sum_actual) / n;

            // Calculate metrics
            let r_squared = 1.0 - (sum_squared_error / tss);
            let mse = sum_squared_error / n;
            let mae = sum_abs_error / n;

            (r_squared, mse, mae)
        }

        // Helper function to create subplot
        let create_subplot = |area: &DrawingArea<SVGBackend, _>,
                              data: &[(f32, f32)],
                              title: &str,
                              point_color: &RGBColor|
         -> Result<(), Box<dyn std::error::Error>> {
            let min_val = 1.5f32;
            let max_val = 4.0f32;

            // let r_squared = 1.0 - (rss / tss);
            let (r_squared, mse, mae) = calculate_metrics(data);

            let mut chart = ChartBuilder::on(area)
                .caption(title, ("sans-serif", 30))
                .margin(10)
                .x_label_area_size(30)
                .y_label_area_size(30)
                .build_cartesian_2d(min_val..max_val, min_val..max_val)?;

            chart
                .configure_mesh()
                .x_desc("Actual Rejects")
                .y_desc("Predicted Rejects")
                .draw()?;

            // Perfect prediction line
            chart.draw_series(LineSeries::new(
                vec![(min_val, min_val), (max_val, max_val)],
                &BLACK.mix(0.5),
            ))?;

            // Draw metrics
            chart.draw_series(std::iter::once(Text::new(
                format!("R² = {:.3}", r_squared),
                (min_val + 0.1, max_val - 0.2),
                ("sans-serif", 20).into_font(),
            )))?;

            chart.draw_series(std::iter::once(Text::new(
                format!("MSE = {:.3}", mse),
                (min_val + 0.1, max_val - 0.4),
                ("sans-serif", 20).into_font(),
            )))?;

            chart.draw_series(std::iter::once(Text::new(
                format!("MAE = {:.3}", mae),
                (min_val + 0.1, max_val - 0.6),
                ("sans-serif", 20).into_font(),
            )))?;

            // Draw data points
            chart.draw_series(
                data.iter()
                    .map(|point| Circle::new(*point, 5, point_color.mix(0.8).filled())),
            )?;

            Ok(())
        };

        // Define colors for training and testing data
        let train_color = RGBColor(65, 105, 225); // Royal Blue
        let test_color = RGBColor(220, 20, 60); // Crimson

        create_subplot(
            &train_area,
            &self.train_predictions,
            "Training Data: Actual vs Predicted",
            &train_color,
        )?;
        create_subplot(
            &test_area,
            &self.test_predictions,
            "Testing Data: Actual vs Predicted",
            &test_color,
        )?;

        Ok(())
    }

    #[rustfmt::skip]
    pub fn print_metrics(&self) {
        log::info!(
            "Regression Metrics:");
        log::info!(
            "Training Set ............ Sum of Squared Errors: {:.6}",
            self.train_sse
        );
        log::info!(
            "Test Set ................ Sum of Squared Errors: {:.6}",
            self.test_sse
        );
        log::info!(
            "Training Set ............ Mean Squared Error:    {:.6}",
            self.train_sse / self.train_predictions.len() as f32
        );
        log::info!(
            "Test Set ................ Mean Squared Error:    {:.6}",
            self.test_sse / self.test_predictions.len() as f32
        );
    }
}
