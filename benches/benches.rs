use criterion::{criterion_group, criterion_main, Criterion};
use qfilter::*;

fn bench_new(c: &mut Criterion) {
    c.bench_function("new", |b| b.iter(|| Filter::new(1000, 0.005).unwrap()));
}

fn bench_get_ok_medium(c: &mut Criterion) {
    let mut f = Filter::new(100000, 0.01).unwrap();
    for i in 0..f.capacity() {
        f.insert_duplicated(i).unwrap();
    }
    let capacity = f.capacity();
    c.bench_function("get_ok_medium", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let mut n = 0;
            for _ in 0..100 {
                n += f.contains(i) as u64;
                i = (i + 1) % capacity;
            }
            n
        })
    });
}

fn bench_get_nok_medium(c: &mut Criterion) {
    let mut f = Filter::new(100000, 0.01).unwrap();
    for i in 0..f.capacity() {
        f.insert_duplicated(i).unwrap();
    }
    c.bench_function("get_nok_medium", |b| {
        let mut i = f.capacity();
        b.iter(|| {
            let mut n = 0;
            for _ in 0..100 {
                n += f.contains(i) as u64;
                i = i.wrapping_add(1);
            }
            n
        })
    });
}

fn bench_get_ok_medium_75(c: &mut Criterion) {
    let mut f = Filter::new(100000, 0.01).unwrap();
    for i in 0..f.capacity() * 3 / 4 {
        f.insert_duplicated(i).unwrap();
    }
    let capacity = f.capacity();
    c.bench_function("get_ok_medium_75", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let mut n = 0;
            for _ in 0..100 {
                n += f.contains(i) as u64;
                i = (i + 1) % capacity;
            }
            n
        })
    });
}

fn bench_get_nok_medium_75(c: &mut Criterion) {
    let mut f = Filter::new(100000, 0.01).unwrap();
    for i in 0..f.capacity() * 3 / 4 {
        f.insert_duplicated(i).unwrap();
    }
    c.bench_function("get_nok_medium_75", |b| {
        let mut i = f.capacity();
        b.iter(|| {
            let mut n = 0;
            for _ in 0..100 {
                n += f.contains(i) as u64;
                i = i.wrapping_add(1);
            }
            n
        })
    });
}

fn bench_grow(c: &mut Criterion) {
    c.bench_function("grow", |b| {
        b.iter(|| {
            let mut f = Filter::new(10000, 0.01).unwrap();
            for i in 0..f.capacity() {
                f.insert_duplicated(i).unwrap();
            }
            f
        })
    });
}

fn bench_grow_from_90pct(c: &mut Criterion) {
    let mut f = Filter::new(10000, 0.01).unwrap();
    for i in 0..f.capacity() / 10 * 9 {
        f.insert_duplicated(i).unwrap();
    }
    c.bench_function("grow_from_90pct", |b| {
        b.iter(|| {
            let mut f = f.clone();
            for i in f.len()..f.capacity() {
                f.insert_duplicated(i).unwrap();
            }
            f
        })
    });
}

fn bench_grow_resizeable(c: &mut Criterion) {
    c.bench_function("grow_resizeable", |b| {
        b.iter(|| {
            let mut f = Filter::new_resizeable(0, 10000, 0.01).unwrap();
            for i in 0u64.. {
                if f.insert_duplicated(i).is_err() {
                    break;
                }
            }
            f
        })
    });
}

fn bench_shrink(c: &mut Criterion) {
    let mut f = Filter::new(10000, 0.01).unwrap();
    for i in 0..f.capacity() {
        f.insert_duplicated(i).unwrap();
    }
    c.bench_function("shrink", |b| {
        b.iter(|| {
            let mut f = f.clone();
            for i in 0..f.capacity() {
                f.remove(i);
            }
            f
        })
    });
}

fn bench_shrink_10pct(c: &mut Criterion) {
    let mut f = Filter::new(10000, 0.01).unwrap();
    for i in 0..f.capacity() {
        f.insert_duplicated(i).unwrap();
    }
    c.bench_function("shrink_10pct", |b| {
        b.iter(|| {
            let mut f = f.clone();
            // Remove 10% of items (from 100% to 90% fill)
            for i in 0..f.capacity() / 10 {
                f.remove(i);
            }
            f
        })
    });
}

fn bench_fingerprints(c: &mut Criterion) {
    let mut f = Filter::new(100000, 0.01).unwrap();
    for i in 0..f.capacity() {
        f.insert_duplicated(i).unwrap();
    }
    c.bench_function("fingerprints", |b| {
        b.iter(|| {
            assert_eq!(f.fingerprints().count(), f.capacity() as usize);
        })
    });
}

fn bench_sorted_insert(c: &mut Criterion) {
    let inserter = Builder::new(10000, 0.01).unwrap();
    let fp_size = inserter.fingerprint_size();
    let cap = inserter.capacity();
    let mut fingerprints: Vec<u64> = (0..cap).map(|i| compute_fingerprint(i, fp_size)).collect();
    fingerprints.sort_unstable();

    c.bench_function("sorted_insert", |b| {
        b.iter(|| {
            let mut inserter = Builder::new(10000, 0.01).unwrap();
            for &h in &fingerprints {
                inserter.insert_fingerprint(true, h).unwrap();
            }
            inserter.into_filter()
        })
    });
}

fn bench_merge(c: &mut Criterion) {
    let mut f1 = Filter::new(10000, 0.01).unwrap();
    let mut f2 = Filter::new(10000, 0.01).unwrap();
    for i in 0..f1.capacity() / 2 {
        f1.insert_duplicated(i).unwrap();
    }
    for i in f1.capacity() / 2..f1.capacity() {
        f2.insert_duplicated(i).unwrap();
    }
    c.bench_function("merge", |b| {
        b.iter(|| {
            let mut f = f1.clone();
            f.merge(false, &f2).unwrap();
            f
        })
    });
}

criterion_group!(
    benches,
    bench_new,
    bench_get_ok_medium,
    bench_get_nok_medium,
    bench_get_ok_medium_75,
    bench_get_nok_medium_75,
    bench_grow,
    bench_grow_from_90pct,
    bench_grow_resizeable,
    bench_shrink,
    bench_shrink_10pct,
    bench_fingerprints,
    bench_sorted_insert,
    bench_merge,
);
criterion_main!(benches);
