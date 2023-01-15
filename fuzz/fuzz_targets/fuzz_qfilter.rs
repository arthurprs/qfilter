#![no_main]
use libfuzzer_sys::fuzz_target;

const FUZZ_REMOVES: bool = true;
const CHECK_EVERY: usize = 8;

fuzz_target!(|data: Vec<i16>| {
    if data.len() < 2 {
        return;
    }
    let cap = (data[0] as u64).min(data.len() as u64 / 2);
    let fp = (0.1f64).powi(data[1].leading_ones() as i32).min(0.1);
    let ops = data
        .into_iter()
        .map(|i| {
            if i < 0 && FUZZ_REMOVES {
                (false, i.checked_neg().unwrap_or(0) as u16)
            } else {
                (true, i as u16)
            }
        })
        .collect::<Vec<(bool, u16)>>();
    // The "Model", tracks the count for each item
    let mut counts = [0u64; (u16::MAX as usize) + 1];
    let mut f = qfilter::Filter::new(cap, fp);
    for i in 0..ops.len() {
        // print_sample(&counts);
        // dbg!(ops[i]);

        let (add, item) = ops[i];
        if add {
            if f.insert_duplicated(item).is_err() {
                continue;
            }
            counts[item as usize] += 1;
        } else {
            if counts[item as usize] != 0 && f.remove(item) {
                counts[item as usize] -= 1;
            } else {
                continue;
            }
        }
        if i % CHECK_EVERY == 0 {
            for &(_add, e) in &ops[..=i] {
                let min = counts[e as usize];
                // Since we can only check for >= due to collisions skip min = 0
                if min != 0 {
                    let est = f.count(e);
                    assert!(est >= min, "{}: est {} min {}", e, est, min);
                }
            }
        }
    }
    for &(_add, e) in &ops {
        let min = counts[e as usize];
        let est = f.count(e);
        assert!(est >= min, "{}: est {} min {}", e, est, min);
    }
});

#[allow(dead_code)]
fn print_sample(counts: &[u64]) {
    print!("[");
    for (i, c) in counts.iter().copied().enumerate() {
        if c != 0 {
            print!("({i}u16, {c}), ");
        }
    }
    println!("]");
}
