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
    fp_size: u8,
    ops: Vec<(bool, u16)>,
}

fuzz_target!(|input: Input| {
    let Input { cap, ops, fp_size } = input;
    // The "Model", tracks the count for each item
    let mut counts = [0u64; (u16::MAX as usize) + 1];
    let Ok(mut f) = qfilter::Filter::with_fingerprint_size(cap as u64, fp_size.clamp(7, 64)) else {
        return;
    };
    for i in 0..ops.len() {
        // print_sample(&counts);
        // dbg!(ops[i]);

        let (add, item) = ops[i];
        let item = item as u64;
        if !FUZZ_REMOVES || add {
            if f.insert_fingerprint(true, item).is_err() {
                continue;
            }
            counts[item as usize] += 1;
        } else if counts[item as usize] != 0 && f.remove_fingerprint(item) {
            counts[item as usize] -= 1;
        } else {
            continue;
        }

        if i % CHECK_EVERY == 0 {
            for &(_add, e) in &ops[..=i] {
                let min = counts[e as usize];
                // Since we can only check for >= due to collisions skip min = 0
                if min != 0 {
                    let est = f.count_fingerprint(e as u64);
                    assert!(est >= min, "{e}: est {est} < min {min}");
                }
            }
        }
    }

    for shrunk in [false, true] {
        for &(_add, e) in &ops {
            let min = counts[e as usize];
            let est = f.count_fingerprint(e as u64);
            assert!(est >= min, "{e}: est {est} < min {min} shrunk {shrunk:?}");
        }
        let prints = f.fingerprints().collect::<Vec<_>>();
        let mut expected_prints = counts
            .iter()
            .enumerate()
            .flat_map(|(i, n)| {
                let t = (i as u64) << (64 - f.fingerprint_size()) >> (64 - f.fingerprint_size());
                std::iter::repeat(t).take(*n as usize)
            })
            .collect::<Vec<_>>();
        expected_prints.sort_unstable();
        assert_eq!(prints.len(), f.len() as usize);
        assert_eq!(prints, expected_prints);
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
