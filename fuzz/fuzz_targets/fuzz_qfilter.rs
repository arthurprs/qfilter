#![no_main]
use libfuzzer_sys::fuzz_target;

const FUZZ_REMOVES: Option<&str> = option_env!("FUZZ_REMOVES");

fuzz_target!(|data: Vec<i16>| {
    if data.len() < 2 {
        return;
    }
    let cap = (data[0] as u64).min(data.len() as u64 / 2);
    let fp = (0.1f64).powi(data[1].leading_ones() as i32).min(0.1);
    let ops = data
        .into_iter()
        .map(|i| {
            if i < 0 && FUZZ_REMOVES.is_some() {
                (false, i.checked_neg().unwrap_or(0) as u16)
            } else {
                (true, i as u16)
            }
        })
        .collect::<Vec<(bool, u16)>>();
    let mut counts = [0u64; (u16::MAX as usize) +1 ];
    let mut f = qfilter::Filter::new(cap, fp);
    for i in 0..ops.len() {
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
        for &(_add, e) in &ops[..=i] {
            let est = f.count(e);
            let min = counts[e as usize];
            assert!(est >= min, "{}: est {} min {}", e, est, min);
        }
    }
    for &(_add, e) in &ops {
        let est = f.count(e);
        let min = counts[e as usize];
        assert!(est >= min, "{}: est {} min {}", e, est, min);
    }
});
