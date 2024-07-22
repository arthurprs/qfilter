#![feature(test)]
extern crate test;

use qfilter::*;
use test::Bencher;

#[bench]
fn bench_new(b: &mut Bencher) {
    b.iter(|| Filter::new(1000, 0.005).unwrap());
}
#[bench]
fn bench_get_ok_medium(b: &mut Bencher) {
    let mut f = Filter::new(100000, 0.01).unwrap();
    for i in 0..f.capacity() {
        f.insert_duplicated(&i).unwrap();
    }
    let mut i = 0;
    b.iter(|| {
        i += 1;
        f.contains(&i)
    })
}

#[bench]
fn bench_get_nok_medium(b: &mut Bencher) {
    let mut f = Filter::new(100000, 0.01).unwrap();
    for i in 0..f.capacity() {
        f.insert_duplicated(&i).unwrap();
    }
    let mut i = f.capacity();
    b.iter(|| {
        i += 1;
        f.contains(&i)
    })
}

#[bench]
fn bench_grow(b: &mut Bencher) {
    b.iter(|| {
        let mut f = Filter::new(10000, 0.01).unwrap();
        for i in 0..f.capacity() {
            f.insert_duplicated(i).unwrap();
        }
        f
    });
}

#[bench]
fn bench_grow_from_90pct(b: &mut Bencher) {
    let mut f = Filter::new(10000, 0.01).unwrap();
    for i in 0..f.capacity() / 10 * 9 {
        f.insert_duplicated(i).unwrap();
    }
    b.iter(|| {
        let mut f = f.clone();
        for i in f.len()..f.capacity() {
            f.insert_duplicated(i).unwrap();
        }
        f
    });
}

#[bench]
fn bench_grow_resizeable(b: &mut Bencher) {
    b.iter(|| {
        let mut f = Filter::new_resizeable(0, 10000, 0.01).unwrap();
        for i in 0u64.. {
            if f.insert_duplicated(i).is_err() {
                break;
            }
        }
        assert_eq!(f.len(), 10000u64.next_power_of_two() * 19 / 20);
        f
    });
}

#[bench]
fn bench_shrink(b: &mut Bencher) {
    let mut f = Filter::new(10000, 0.01).unwrap();
    for i in 0..f.capacity() {
        let _ = f.insert(i);
    }
    b.iter(|| {
        let mut f = f.clone();
        for i in 0..f.capacity() {
            f.remove(i);
        }
        f
    });
}
