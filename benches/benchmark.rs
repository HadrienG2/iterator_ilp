use criterion::{BenchmarkId, Criterion, Throughput};
use hwlocality::Topology;
use iterator_ilp::{IteratorILP, TrustedLen};
use multiversion::multiversion;
use rand::random;
use std::{iter::Copied, slice};

#[multiversion(targets = "simd")]
fn criterion_benchmark(c: &mut Criterion) {
    // Probe cache capacity in unit of floats (or 32-bit numbers in general)

    let min_l1d_bytes = Topology::new()
        .unwrap()
        .cpu_cache_stats()
        .smallest_data_cache_sizes()[0];
    let min_l1d_floats = min_l1d_bytes as usize / std::mem::size_of::<f32>();
    let min_l1d_floats_pow2 = usize::BITS - min_l1d_floats.leading_zeros();

    // Benchmark floating-point sum as a demonstration of the benefits of ILP on
    // accumulation of non-associative quantities

    sum_f32_benchmark(c, min_l1d_floats_pow2, "Iterator::sum::<f32>", |floats| {
        floats.iter().sum::<f32>()
    });

    sum_f32_benchmark(c, min_l1d_floats_pow2, "Naive loop", |floats| {
        let mut acc = 0.0;
        for &float in floats {
            acc += float;
        }
        acc
    });

    sum_f32_benchmark(
        c,
        min_l1d_floats_pow2,
        "IteratorILP::sum_ilp::<1, f32>",
        |floats| floats.iter().sum_ilp::<1, f32>(),
    );
    sum_f32_benchmark(
        c,
        min_l1d_floats_pow2,
        "IteratorILP::sum_ilp::<2, f32>",
        |floats| floats.iter().sum_ilp::<2, f32>(),
    );
    sum_f32_benchmark(
        c,
        min_l1d_floats_pow2,
        "IteratorILP::sum_ilp::<4, f32>",
        |floats| floats.iter().sum_ilp::<4, f32>(),
    );
    sum_f32_benchmark(
        c,
        min_l1d_floats_pow2,
        "IteratorILP::sum_ilp::<8, f32>",
        |floats| floats.iter().sum_ilp::<8, f32>(),
    );
    sum_f32_benchmark(
        c,
        min_l1d_floats_pow2,
        "IteratorILP::sum_ilp::<16, f32>",
        |floats| floats.iter().sum_ilp::<16, f32>(),
    );
    sum_f32_benchmark(
        c,
        min_l1d_floats_pow2,
        "IteratorILP::sum_ilp::<32, f32>",
        |floats| floats.iter().sum_ilp::<32, f32>(),
    );
    sum_f32_benchmark(
        c,
        min_l1d_floats_pow2,
        "IteratorILP::sum_ilp::<64, f32>",
        |floats| floats.iter().sum_ilp::<64, f32>(),
    );

    // Benchmark integer search with a nontrivial criterion on an iterator with
    // side effects, showing that even for integers and easier tasks like
    // searching there can be a benefit to ILP if the iterator gets complicated.

    let predicate = |needle: u32| {
        move |mut item: u32| {
            item ^= item << 13;
            item ^= item >> 17;
            item ^= item << 5;
            item == needle
        }
    };

    find_u32_benchmark(
        c,
        min_l1d_floats_pow2,
        "Iterator::any",
        |mut iter, needle| iter.any(predicate(needle)),
    );

    find_u32_benchmark(
        c,
        min_l1d_floats_pow2,
        "IteratorILP::any_ilp<1>",
        |iter, needle| iter.any_ilp::<1>(predicate(needle)),
    );
    find_u32_benchmark(
        c,
        min_l1d_floats_pow2,
        "IteratorILP::any_ilp<2>",
        |iter, needle| iter.any_ilp::<2>(predicate(needle)),
    );
    find_u32_benchmark(
        c,
        min_l1d_floats_pow2,
        "IteratorILP::any_ilp<4>",
        |iter, needle| iter.any_ilp::<4>(predicate(needle)),
    );
    find_u32_benchmark(
        c,
        min_l1d_floats_pow2,
        "IteratorILP::any_ilp<8>",
        |iter, needle| iter.any_ilp::<8>(predicate(needle)),
    );
    find_u32_benchmark(
        c,
        min_l1d_floats_pow2,
        "IteratorILP::any_ilp<16>",
        |iter, needle| iter.any_ilp::<16>(predicate(needle)),
    );
    find_u32_benchmark(
        c,
        min_l1d_floats_pow2,
        "IteratorILP::any_ilp<32>",
        |iter, needle| iter.any_ilp::<32>(predicate(needle)),
    );
    find_u32_benchmark(
        c,
        min_l1d_floats_pow2,
        "IteratorILP::any_ilp<64>",
        |iter, needle| iter.any_ilp::<64>(predicate(needle)),
    );
}

fn sum_f32_benchmark(
    c: &mut Criterion,
    min_l1d_floats_pow2: u32,
    name: &str,
    mut sum: impl FnMut(&[f32]) -> f32,
) {
    let mut group = c.benchmark_group(name);
    for size_pow2 in 0..=min_l1d_floats_pow2 {
        let num_floats = 2usize.pow(size_pow2);
        let mut floats = std::iter::repeat_with(random)
            .take(num_floats)
            .collect::<Vec<_>>();

        group.throughput(Throughput::Elements(num_floats as u64));
        group.bench_function(BenchmarkId::from_parameter(num_floats), |b| {
            b.iter(|| {
                std::hint::black_box(&mut floats);
                sum(&floats[..])
            });
        });
    }
    group.finish();
}

fn find_u32_benchmark(
    c: &mut Criterion,
    min_l1d_u32_pow2: u32,
    name: &str,
    mut any: impl FnMut(SideEffects, u32) -> bool,
) {
    let mut group = c.benchmark_group(name);
    for size_pow2 in 0..=min_l1d_u32_pow2 {
        let num_ints = 2usize.pow(size_pow2);
        let needle = random();
        let mut ints = std::iter::repeat_with(random)
            .filter(|&i| i != needle)
            .take(num_ints)
            .collect::<Vec<_>>();

        group.throughput(Throughput::Elements(num_ints as u64));
        group.bench_function(BenchmarkId::from_parameter(num_ints), |b| {
            b.iter(|| {
                std::hint::black_box(&mut ints);
                let iter = SideEffects {
                    iter: ints.iter().copied(),
                    last: 0,
                };
                any(iter, needle)
            });
        });
    }
    group.finish();
}

struct SideEffects<'slice> {
    iter: Copied<slice::Iter<'slice, u32>>,
    last: u32,
}
//
impl Iterator for SideEffects<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        self.last += 1;
        self.iter.next()
    }
}
//
impl ExactSizeIterator for SideEffects<'_> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}
//
impl Drop for SideEffects<'_> {
    fn drop(&mut self) {
        std::hint::black_box(self.last);
    }
}
//
unsafe impl TrustedLen for SideEffects {}

criterion::criterion_group!(benches, criterion_benchmark);
criterion::criterion_main!(benches);
