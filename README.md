# Qfilter

Approximate Membership Query Filter ([AMQ-Filter](https://en.wikipedia.org/wiki/Approximate_Membership_Query_Filter))
based on the [Rank Select Quotient Filter (RSQF)](https://dl.acm.org/doi/pdf/10.1145/3035918.3035963).

This is a small and flexible general-purpose AMQ-Filter, it not only supports approximate membership testing like a bloom filter
but also deletions, merging (not implemented), resizing and [serde](https://crates.io/crates/serde) serialization.

* High performance
* Supports removals
* Extremely compact, more so than comparable filters
* Can be created with a initial small capacity and grow as needed
* (De)Serializable with [serde](https://crates.io/crates/serde)
* Portable Rust implementation

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

| Bits per item | Error probability when full |
|--------|----------|
| 3.125  | 0.362    |
| 4.125  | 0.201    |
| 5.125  | 0.106    |
| 6.125  | 0.0547   |
| 7.125  | 0.0277   |
| 8.125  | 0.014    |
| 9.125  | 0.00701  |
| 10.125 | 0.00351  |
| 11.125 | 0.00176  |
| 12.125 | 0.000879 |
| 13.125 | 0.000439 |
| 14.125 | 0.00022  |
| 15.125 | 0.00011  |
| 16.125 | 5.49e-05 |
| 17.125 | 2.75e-05 |
| 18.125 | 1.37e-05 |
| 19.125 | 6.87e-06 |
| 20.125 | 3.43e-06 |
| 21.125 | 1.72e-06 |
| 22.125 | 8.58e-07 |
| 23.125 | 4.29e-07 |
| 24.125 | 2.15e-07 |
| 25.125 | 1.07e-07 |
| 26.125 | 5.36e-08 |
| 27.125 | 2.68e-08 |
| 28.125 | 1.34e-08 |
| 29.125 | 6.71e-09 |
| 30.125 | 3.35e-09 |
| 31.125 | 1.68e-09 |
| 32.125 | 8.38e-10 |

### Not implemented

- [ ] Merging
- [ ] Shrink to fit
- [ ] Counting
- [ ] Smoother resizing by chaining exponentially larger and more precise filters

### Legacy x86_64 CPUs support

The implementation assumes the `popcnt` instruction (equivalent to `integer.count_ones()`) is present
when compiling for x86_64 targets. This is theoretically not guaranteed as the instruction is only
available on AMD/Intel CPUs released after 2007/2008. If that's not the case the Filter constructor will panic.

Support for such legacy x86_64 CPUs can be optionally enabled with the `legacy_x86_64_support`
which incurs a ~10% performance penalty.

### License

This project is licensed under the MIT license.
