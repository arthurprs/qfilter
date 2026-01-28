use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::hint::black_box;
use std::time::Duration;

/// Old implementation: loop-based bit clearing
#[inline]
fn select_old(x: u64, n: u64) -> Option<u64> {
    let mut v = x;
    for _ in 0..n / 8 {
        for _ in 0..8 {
            v &= v.wrapping_sub(1);
        }
    }
    for _ in 0..n % 8 {
        v &= v.wrapping_sub(1);
    }
    if v == 0 {
        None
    } else {
        Some(v.trailing_zeros() as u64)
    }
}

const L8: u64 = 0x0101_0101_0101_0101;

#[inline(always)]
fn le8(x: u64, y: u64) -> u64 {
    ((y | 0x8080_8080_8080_8080).wrapping_sub(x & !0x8080_8080_8080_8080)) & 0x8080_8080_8080_8080
}

#[inline(always)]
fn u_nz8(x: u64) -> u64 {
    ((x | 0x8080_8080_8080_8080).wrapping_sub(L8) | x) & 0x8080_8080_8080_8080
}

/// New implementation: Vigna's broadword select
#[inline]
fn select_new(x: u64, n: u64) -> Option<u64> {
    let mut s = x - ((x & 0xAAAA_AAAA_AAAA_AAAA) >> 1);
    s = (s & 0x3333_3333_3333_3333) + ((s >> 2) & 0x3333_3333_3333_3333);
    s = ((s + (s >> 4)) & 0x0F0F_0F0F_0F0F_0F0F).wrapping_mul(L8);

    let b = (le8(s, n.wrapping_mul(L8)) >> 7).wrapping_mul(L8) >> 53 & !7;
    let l = n - ((s << 8).wrapping_shr(b as u32) & 0xFF);

    s = (u_nz8((x.wrapping_shr(b as u32) & 0xFF).wrapping_mul(L8) & 0x8040_2010_0804_0201) >> 7)
        .wrapping_mul(L8);

    let result = b + ((le8(s, l.wrapping_mul(L8)) >> 7).wrapping_mul(L8) >> 56);
    if result == 72 {
        None
    } else {
        Some(result)
    }
}

fn bench_select(c: &mut Criterion) {
    let mut group = c.benchmark_group("select");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    // 50% occupancy (32 bits set)
    let pattern_50 = 0b10101010_10101010_10101010_10101010_10101010_10101010_10101010_10101010u64;
    let positions_50: &[u64] = &[0, 15, 31];

    // 75% occupancy (48 bits set)
    let pattern_75 = 0b11101110_11101110_11101110_11101110_11101110_11101110_11101110_11101110u64;
    let positions_75: &[u64] = &[0, 23, 47];

    // 95% occupancy (61 bits set)
    let pattern_95 = 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111101u64;
    let positions_95: &[u64] = &[0, 30, 60];

    for (name, pattern, positions) in [
        ("50%", pattern_50, positions_50),
        ("75%", pattern_75, positions_75),
        ("95%", pattern_95, positions_95),
    ] {
        group.bench_with_input(BenchmarkId::new("old", name), &(pattern, positions), |b, &(p, pos)| {
            b.iter(|| {
                let mut sum = 0u64;
                for &n in pos {
                    sum += black_box(select_old(black_box(p), black_box(n))).unwrap_or(0);
                }
                sum
            })
        });

        group.bench_with_input(BenchmarkId::new("new", name), &(pattern, positions), |b, &(p, pos)| {
            b.iter(|| {
                let mut sum = 0u64;
                for &n in pos {
                    sum += black_box(select_new(black_box(p), black_box(n))).unwrap_or(0);
                }
                sum
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_select);
criterion_main!(benches);
