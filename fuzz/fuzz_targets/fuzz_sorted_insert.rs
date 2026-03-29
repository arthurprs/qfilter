#![no_main]
use libfuzzer_sys::arbitrary;
use libfuzzer_sys::arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct Input {
    cap: u16,
    fp_size: u8,
    /// Fingerprints to insert (will be sorted before insertion)
    items: Vec<u16>,
    /// Whether to allow duplicates
    duplicate: bool,
    /// Whether the filter should be resizable (exercises rebuild-based growth logic)
    resizable: bool,
}

fuzz_target!(|input: Input| {
    let Input {
        cap,
        fp_size,
        items,
        duplicate,
        resizable,
    } = input;

    let fp_size = fp_size.clamp(7, 64);

    // When resizable, use cap as initial and items.len() as max to ensure
    // growth is exercised when there are more items than initial capacity.
    let max_cap = if resizable {
        (items.len() as u64).max(cap as u64)
    } else {
        0 // unused
    };

    // Construct filters first so we can use the actual fingerprint_size
    // for masking. The resizable path derives fp_size from fp_rate, which
    // may differ from the input fp_size.
    let regular_filter = if resizable {
        qfilter::Filter::new_resizeable(cap as u64, max_cap, 0.01)
    } else {
        qfilter::Filter::with_fingerprint_size(cap as u64, fp_size)
    };
    let Ok(mut regular) = regular_filter else {
        return;
    };
    let builder_filter = if resizable {
        qfilter::Filter::new_resizeable(cap as u64, max_cap, 0.01)
    } else {
        qfilter::Filter::with_fingerprint_size(cap as u64, fp_size)
    };
    let Ok(builder_filter) = builder_filter else {
        return;
    };
    let mut inserter = qfilter::Builder::new(builder_filter);

    // Use the actual fingerprint size from the constructed filter
    let actual_fp_size = regular.fingerprint_size();
    assert_eq!(actual_fp_size, inserter.fingerprint_size());

    // Build sorted fingerprints, masked to actual fp_size bits
    let mask = if actual_fp_size >= 64 {
        u64::MAX
    } else {
        (1u64 << actual_fp_size) - 1
    };
    let mut fingerprints: Vec<u64> = items.iter().map(|&i| (i as u64) & mask).collect();
    fingerprints.sort_unstable();

    // Regular insertion
    for &h in &fingerprints {
        if regular.insert_fingerprint(duplicate, h).is_err() {
            break;
        }
    }

    // Sorted insertion
    for &h in &fingerprints {
        if inserter.insert_fingerprint(duplicate, h).is_err() {
            break;
        }
    }
    let sorted = inserter.into_filter();

    // Both should produce identical results
    assert_eq!(regular.len(), sorted.len());

    let reg_fps: Vec<u64> = regular.fingerprints().collect();
    let sort_fps: Vec<u64> = sorted.fingerprints().collect();
    assert_eq!(reg_fps, sort_fps);

    // Verify all items are queryable
    for &h in &fingerprints {
        let rc = regular.count_fingerprint(h);
        let sc = sorted.count_fingerprint(h);
        assert_eq!(rc, sc, "count mismatch for fingerprint {h}");
    }
});
