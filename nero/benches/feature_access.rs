use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::collections::HashMap;

#[derive(Clone)]
struct StaticItem {
    con_1: f32,
    con_2: f32,
    con_3: f32,
    con_4: f32,
    #[allow(dead_code)]
    rejects: f32,
}

impl StaticItem {
    fn features(&self, names: &[&str]) -> Vec<f32> {
        names
            .iter()
            .map(|&name| match name {
                "Con_1" => self.con_1,
                "Con_2" => self.con_2,
                "Con_3" => self.con_3,
                "Con_4" => self.con_4,
                _ => 0.0,
            })
            .collect()
    }
}

#[derive(Clone)]
struct DynamicItem {
    values: HashMap<String, f32>,
}

impl DynamicItem {
    fn features(&self, names: &[&str]) -> Vec<f32> {
        names
            .iter()
            .map(|&name| *self.values.get(name).unwrap_or(&0.0))
            .collect()
    }
}

fn create_items() -> (StaticItem, DynamicItem) {
    let static_item = StaticItem {
        con_1: 1.0,
        con_2: 2.0,
        con_3: 3.0,
        con_4: 4.0,
        rejects: 5.0,
    };

    let mut values = HashMap::with_capacity(5);
    values.insert("Con_1".to_string(), 1.0);
    values.insert("Con_2".to_string(), 2.0);
    values.insert("Con_3".to_string(), 3.0);
    values.insert("Con_4".to_string(), 4.0);
    values.insert("Rejects".to_string(), 5.0);

    let dynamic_item = DynamicItem { values };

    (static_item, dynamic_item)
}

fn criterion_benchmark(c: &mut Criterion) {
    let feature_names = &["Con_1", "Con_2", "Con_3", "Con_4"];
    let (static_item, dynamic_item) = create_items();

    let mut group = c.benchmark_group("feature_access");

    group.bench_function("static", |b| {
        b.iter(|| {
            black_box(static_item.features(black_box(feature_names)));
        })
    });

    group.bench_function("dynamic", |b| {
        b.iter(|| {
            black_box(dynamic_item.features(black_box(feature_names)));
        })
    });

    // Benchmark creation cost too
    group.bench_function("create_static", |b| {
        b.iter(|| {
            let item = StaticItem {
                con_1: black_box(1.0),
                con_2: black_box(2.0),
                con_3: black_box(3.0),
                con_4: black_box(4.0),
                rejects: black_box(5.0),
            };
            black_box(item)
        })
    });

    group.bench_function("create_dynamic", |b| {
        b.iter(|| {
            let mut values = HashMap::with_capacity(5);
            values.insert(black_box("Con_1".to_string()), black_box(1.0));
            values.insert(black_box("Con_2".to_string()), black_box(2.0));
            values.insert(black_box("Con_3".to_string()), black_box(3.0));
            values.insert(black_box("Con_4".to_string()), black_box(4.0));
            values.insert(black_box("Rejects".to_string()), black_box(5.0));
            let item = DynamicItem { values };
            black_box(item)
        })
    });

    // Benchmark bulk operations
    let static_items: Vec<_> = (0..1000).map(|_| static_item.clone()).collect();
    let dynamic_items: Vec<_> = (0..1000).map(|_| dynamic_item.clone()).collect();

    group.bench_function("bulk_static", |b| {
        b.iter(|| {
            black_box(
                static_items
                    .iter()
                    .map(|item| item.features(feature_names))
                    .collect::<Vec<_>>(),
            )
        })
    });

    group.bench_function("bulk_dynamic", |b| {
        b.iter(|| {
            black_box(
                dynamic_items
                    .iter()
                    .map(|item| item.features(feature_names))
                    .collect::<Vec<_>>(),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
