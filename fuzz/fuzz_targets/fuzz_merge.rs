#![no_main]
use libfuzzer_sys::arbitrary;
use libfuzzer_sys::arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct Input {
    cap1: u16,
    cap2: u16,
    fp_size: u8,
    items1: Vec<u16>,
    items2: Vec<u16>,
    keep_duplicates: bool,
    resizable: bool,
}

fuzz_target!(|input: Input| {
    let Input {
        cap1,
        cap2,
        fp_size,
        items1,
        items2,
        keep_duplicates,
        resizable,
    } = input;

    let fp_size = fp_size.clamp(7, 64);
    let max_cap = (cap1 as u64)
        .max(cap2 as u64)
        .saturating_add(items1.len() as u64)
        .saturating_add(items2.len() as u64);

    let (mut f1, mut f2) = if resizable {
        let a = qfilter::Filter::new_resizeable(cap1 as u64, max_cap, 0.01);
        let b = qfilter::Filter::new_resizeable(cap2 as u64, max_cap, 0.01);
        match (a, b) {
            (Ok(a), Ok(b)) => (a, b),
            _ => return,
        }
    } else {
        let a = qfilter::Filter::with_fingerprint_size(cap1 as u64, fp_size);
        let b = qfilter::Filter::with_fingerprint_size(cap2 as u64, fp_size);
        match (a, b) {
            (Ok(a), Ok(b)) => (a, b),
            _ => return,
        }
    };

    // Filters must have compatible fingerprint sizes for merge
    if f1.fingerprint_size() != f2.fingerprint_size() {
        return;
    }

    // Populate filters
    for &item in &items1 {
        let _ = f1.insert_duplicated(item);
    }
    for &item in &items2 {
        let _ = f2.insert_duplicated(item);
    }

    let f1_before = f1.clone();

    // Merge f2 into f1
    let Ok(()) = f1.merge(keep_duplicates, &f2) else {
        return;
    };

    // All items from f1 must still be present
    for &item in &items1 {
        assert!(f1.contains(item), "item {item} from f1 missing after merge");
    }
    // All items from f2 must be present
    for &item in &items2 {
        assert!(f1.contains(item), "item {item} from f2 missing after merge");
    }

    // Count invariants (>= not == because fingerprint collisions from
    // other items can add extra matches in the merged filter).
    if keep_duplicates {
        for &item in &items1 {
            let before = f1_before.count(item);
            let from_f2 = f2.count(item);
            let after = f1.count(item);
            assert!(
                after >= before + from_f2,
                "item {item}: count after dup merge ({after}) < f1 ({before}) + f2 ({from_f2})"
            );
        }
    }

    // Merged fingerprints must contain all fingerprints from both original filters.
    let merged_fps: Vec<u64> = f1.fingerprints().collect();
    let mut expected: Vec<u64> = f1_before.fingerprints().chain(f2.fingerprints()).collect();
    expected.sort_unstable();
    if !keep_duplicates {
        expected.dedup();
    }
    assert_eq!(
        merged_fps, expected,
        "merged fingerprints don't match expected"
    );
});
