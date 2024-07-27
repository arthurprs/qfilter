# Qfilter

Efficient bloom filter like data structure, based on the [Rank Select Quotient Filter (RSQF)](https://dl.acm.org/doi/pdf/10.1145/3035918.3035963).

This is a small and flexible general-purpose [AMQ-Filter](https://en.wikipedia.org/wiki/Approximate_Membership_Query_Filter).
It not only supports approximate membership testing like a bloom filter but also deletions, merging,
resizing and [serde](https://crates.io/crates/serde) serialization.

* High performance
* Supports removals
* Extremely compact, more so than comparable filters
* Can be created with a initial small capacity and grow as needed
* (De)Serializable with [serde](https://crates.io/crates/serde)
* Portable Rust implementation
* Only verifiable usages of unsafe

This data structure is similar to a hash table that stores fingerprints in a very compact way.
Fingerprints are similar to a hash values, but are possibly truncated.
The reason for false positives is that multiple items can map to the same fingerprint.
For more information see the [quotient filter Wikipedia page](https://en.wikipedia.org/wiki/Quotient_filter)
that describes a similar but less optimized version of the data structure.
The actual implementation is based on the [Rank Select Quotient Filter (RSQF)](https://dl.acm.org/doi/pdf/10.1145/3035918.3035963).

The public API also exposes a fingerprint API, which can be used to succinctly store u64 hash values.

### Example

```rust
let mut f = qfilter::Filter::new(1000000, 0.01);
for i in 0..1000 {
    f.insert(i).unwrap();
}
for i in 0..1000 {
    assert!(f.contains(i));
}
```

### Hasher

The hashing algorithm used is [xxhash3](https://crates.io/crates/xxhash-rust) which offers both high performance and stability across platforms.

### Filter size

For a given capacity and error probability the RSQF may require significantly less space than the equivalent bloom filter or other AMQ-Filters.

| Bits per item | Error probability when full | Bits per item (Cont.) | Error (cont.) |
|:---:|:---:|:---:|---|
| 3.125 | 0.362 | 19.125 | 6.87e-06 |
| 4.125 | 0.201 | 20.125 | 3.43e-06 |
| 5.125 | 0.106 | 21.125 | 1.72e-06 |
| 6.125 | 0.0547 | 22.125 | 8.58e-07 |
| 7.125 | 0.0277 | 23.125 | 4.29e-07 |
| 8.125 | 0.014 | 24.125 | 2.15e-07 |
| 9.125 | 0.00701 | 25.125 | 1.07e-07 |
| 10.125 | 0.00351 | 26.125 | 5.36e-08 |
| 11.125 | 0.00176 | 27.125 | 2.68e-08 |
| 12.125 | 0.000879 | 28.125 | 1.34e-08 |
| 13.125 | 0.000439 | 29.125 | 6.71e-09 |
| 14.125 | 0.00022 | 30.125 | 3.35e-09 |
| 15.125 | 0.00011 | 31.125 | 1.68e-09 |
| 16.125 | 5.49e-05 | 32.125 | 8.38e-10 |
| 17.125 | 2.75e-05 | .. | .. |
| 18.125 | 1.37e-05 | .. | .. |

### Compatibility between versions 0.1 and 0.2

Version 0.2 changed public APIs (e.g. fallible constructors) which required a major version bump.

Serialization is bidirectionally compatible between versions 0.1 and 0.2.

### Not implemented

- [ ] Fingerprint attached values
- [ ] Counting with fingerprint values, not fingerprint duplication
- [ ] More advanced growth strategies (InfiniFilter).

### Legacy x86_64 CPUs support

The implementation assumes the `popcnt` instruction (equivalent to `integer.count_ones()`) is present
when compiling for x86_64 targets. This is theoretically not guaranteed as the instruction is only
available on AMD/Intel CPUs released after 2007/2008. If that's not the case the Filter constructor will panic.

Support for such legacy x86_64 CPUs can be optionally enabled with the `legacy_x86_64_support`
which incurs a ~10% performance penalty.

### License

This project is licensed under the MIT license.
