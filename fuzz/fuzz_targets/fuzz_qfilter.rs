#![no_main]
use libfuzzer_sys::arbitrary;
use libfuzzer_sys::arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

const FUZZ_REMOVES: bool = true;
const CHECK_EVERY: usize = 8;
const CHECK_SHRUNK: bool = true;

#[derive(Debug, Arbitrary)]
struct Input {
    cap: u16,
    max_cap: u16,
    fp_exp: u16,
    ops: Vec<(bool, u16)>,
}

fuzz_target!(|input: Input| {
    let Input {
        cap,
        max_cap,
        fp_exp,
        ops,
    } = input;
    let max_cap = max_cap.max(cap) as u64;
    let cap = cap as u64;
    let fp = 2f64.powi(-(fp_exp.leading_ones() as i32));
    // The "Model", tracks the count for each item
    let mut counts = [0u64; (u16::MAX as usize) + 1];
    let mut f = qfilter::Filter::new_resizeable(cap, max_cap, fp);
    for i in 0..ops.len() {
        // print_sample(&counts);
        // dbg!(ops[i]);

        let (add, item) = ops[i];
        if !FUZZ_REMOVES || add {
            if f.insert_duplicated(item).is_err() {
                continue;
            }
            counts[item as usize] += 1;
        } else if counts[item as usize] != 0 && f.remove(item) {
            counts[item as usize] -= 1;
        } else {
            continue;
        }

        if i % CHECK_EVERY == 0 {
            for &(_add, e) in &ops[..=i] {
                let min = counts[e as usize];
                // Since we can only check for >= due to collisions skip min = 0
                if min != 0 {
                    let est = f.count(e);
                    assert!(est >= min, "{e}: est {est} < min {min}");
                }
            }
        }
    }

    for shrunk in [false, true] {
        for &(_add, e) in &ops {
            let min = counts[e as usize];
            let est = f.count(e);
            assert!(est >= min, "{e}: est {est} < min {min} shrunk {shrunk:?}");
        }
        if !CHECK_SHRUNK {
            break;
        }
        f.shrink_to_fit();
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
