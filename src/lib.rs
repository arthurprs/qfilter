//! Approximate Membership Query Filter ([AMQ-Filter](https://en.wikipedia.org/wiki/Approximate_Membership_Query_Filter))
//! based on the [Rank Select Quotient Filter (RSQF)](https://dl.acm.org/doi/pdf/10.1145/3035918.3035963).
//!
//! This is a small and flexible general-purpose AMQ-Filter, it not only supports approximate membership testing like a bloom filter
//! but also deletions, merging, resizing and [serde](https://crates.io/crates/serde) serialization.
//!
//! ### Example
//!
//! ```rust
//! let mut f = qfilter::Filter::new(1000000, 0.01).unwrap();
//! for i in 0..1000 {
//!     f.insert(i).unwrap();
//! }
//! for i in 0..1000 {
//!     assert!(f.contains(i));
//! }
//! ```
//!
//! ### Hasher
//!
//! Methods accepting `T: Hash` are provided for convenience using
//! [foldhash-portable](https://crates.io/crates/foldhash-portable), which offers high performance
//! and stability across platforms. Note that a fixed seed is used (no DoS resistance)
//! and `#[derive(Hash)]` output is [not guaranteed stable](https://github.com/hoxxep/portable-hash#whats-wrong-with-the-stdhash-traits)
//! across Rust compiler versions.
//!
//! ### Custom hasher
//!
//! [`Filter`] supports a custom [`BuildHasher`] via its `S` type
//! parameter (similar to [`HashMap`](std::collections::HashMap)). Use
//! [`Filter::new_with_hasher()`] and related constructors. Caveats:
//!
//! - The hasher is **not serialized**. On deserialization it is reconstructed via `S::default()`.
//!   If that doesn't produce the correct hasher (e.g. random-seeded hashers), use
//!   [`Filter::with_hasher()`] to restore it.
//! - Filters being **merged** must use the same hasher and seed.
//! - The **same hasher instance** (or equivalent seed) must be used for all operations.
//!
//! ### Filter size
//!
//! For a given capacity and error probability the RSQF may require significantly less space than the equivalent bloom filter or other AMQ-Filters.
//!
//! | Bits per item | Error probability when full | Bits per item (cont.) | Error (cont.) |
//! |:---:|:---:|:---:|---|
//! | 3.125 | 0.362 | 19.125 | 6.87e-06 |
//! | 4.125 | 0.201 | 20.125 | 3.43e-06 |
//! | 5.125 | 0.106 | 21.125 | 1.72e-06 |
//! | 6.125 | 0.0547 | 22.125 | 8.58e-07 |
//! | 7.125 | 0.0277 | 23.125 | 4.29e-07 |
//! | 8.125 | 0.014 | 24.125 | 2.15e-07 |
//! | 9.125 | 0.00701 | 25.125 | 1.07e-07 |
//! | 10.125 | 0.00351 | 26.125 | 5.36e-08 |
//! | 11.125 | 0.00176 | 27.125 | 2.68e-08 |
//! | 12.125 | 0.000879 | 28.125 | 1.34e-08 |
//! | 13.125 | 0.000439 | 29.125 | 6.71e-09 |
//! | 14.125 | 0.00022 | 30.125 | 3.35e-09 |
//! | 15.125 | 0.00011 | 31.125 | 1.68e-09 |
//! | 16.125 | 5.49e-05 | 32.125 | 8.38e-10 |
//! | 17.125 | 2.75e-05 | .. | .. |
//! | 18.125 | 1.37e-05 | .. | .. |
//!
//! ### Performance
//!
//! - **Lookup/Insert/Delete**: O(1) expected case, O(n) worst case with pathological hash collisions
//! - **Memory overhead**: 2.125 bits per slot for metadata (occupieds, runends, offset)
//! - **Cache efficiency**: Block-based layout with 64-slot blocks improves cache locality
//!
//! Performance degrades gracefully as occupancy increases. The filter automatically
//! limits occupancy to 95% to maintain good performance.
//!
//! ### Legacy x86_64 CPUs support
//!
//! The implementation assumes `popcnt` and BMI2 (`pdep`, `tzcnt`) instructions are available
//! when compiling for x86_64 targets. These instructions are available on CPUS released after 2015.
//! If they are not available, the Filter constructor will panic.
//!
//! The `legacy_x86_64_support` feature enables support for older x86_64 CPUs by using
//! portable fallbacks.
#![cfg_attr(docsrs, feature(doc_cfg))]

use std::{
    cmp::Ordering,
    hash::{BuildHasher, Hash},
    num::{NonZeroU64, NonZeroU8},
    ops::{RangeBounds, RangeFrom},
};

#[cfg(feature = "jsonschema")]
use schemars::JsonSchema;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
pub use stable_hasher::StableBuildHasher;

mod portable_select;
mod stable_hasher;

/// Truncates a hash to a fingerprint of the given bit size.
///
/// Only the lower `fingerprint_bits` bits of `hash` are returned.
/// This is useful when pre-computing fingerprints for [`Builder`] sorted insertion,
/// which requires truncated fingerprints for correct sort order.
///
/// In most other cases you don't need this — the fingerprint-based APIs
/// ([`Filter::contains_fingerprint()`], [`Filter::insert_fingerprint()`], etc.) accept
/// full untruncated hashes and truncate internally.
///
/// - `fingerprint_bits == 0` returns `0`
/// - `fingerprint_bits >= 64` returns the full hash unchanged
#[inline]
pub fn truncate_to_fingerprint(hash: u64, fingerprint_bits: u8) -> u64 {
    if fingerprint_bits >= 64 {
        hash
    } else {
        hash & ((1u64 << fingerprint_bits) - 1)
    }
}

/// Hashes an item and truncates it to a fingerprint of the given bit size.
///
/// Equivalent to `truncate_to_fingerprint(build_hasher.hash_one(&item), fingerprint_bits)`.
///
/// See [`truncate_to_fingerprint()`] for details on when truncation is needed.
#[inline]
pub fn compute_fingerprint_with_hasher<T: Hash, S: BuildHasher>(
    build_hasher: &S,
    item: T,
    fingerprint_bits: u8,
) -> u64 {
    truncate_to_fingerprint(build_hasher.hash_one(&item), fingerprint_bits)
}

// Private convenience wrapper using the default hasher (test only).
#[cfg(test)]
#[inline]
fn compute_fingerprint<T: Hash>(item: T, fingerprint_bits: u8) -> u64 {
    compute_fingerprint_with_hasher(&StableBuildHasher, item, fingerprint_bits)
}

/// Specifications for a filter with given parameters.
///
/// Returned by [`filter_specs()`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilterSpecs {
    /// Actual maximum capacity (may be higher than requested due to rounding).
    pub max_capacity: u64,
    /// Maximum false positive rate when at capacity (may be lower than requested due to rounding).
    pub max_error_ratio: f64,
    /// Storage cost per item in bits when at max capacity (remainder bits + 2.125 metadata overhead).
    pub bits_per_item: f64,
    /// Memory usage in bytes at minimum capacity (smallest resizable filter).
    pub memory_bytes_min: usize,
    /// Memory usage in bytes at maximum capacity.
    pub memory_bytes_max: usize,
    /// Internal fingerprint size in bits.
    ///
    /// This is an implementation detail exposed for advanced use cases like
    /// pre-computing fingerprints with [`compute_fingerprint_with_hasher()`]. It represents
    /// the full hash width (qbits + rbits), not the storage cost per item.
    pub fingerprint_bits: u8,
}

/// Computes filter specifications for a given capacity and false positive rate.
///
/// Returns the fingerprint size and memory usage bounds, useful for planning
/// memory allocation or pre-computing fingerprints before creating a filter.
///
/// # Parameters
///
/// - `capacity`: The number of items the filter should hold.
/// - `fp_rate`: Upper bound on false positive probability (clamped to (0, 0.5]).
///   The actual rate may be lower due to internal rounding.
///
/// # Errors
///
/// - [`Error::CapacityTooLarge`] if capacity exceeds [`Filter::MAX_CAPACITY`].
/// - [`Error::NotEnoughFingerprintBits`] if the configuration requires more than 64 bits.
///
/// # Example
///
/// ```rust
/// use qfilter::{filter_specs, compute_fingerprint_with_hasher, StableBuildHasher, Filter};
///
/// let capacity = 10000;
/// let fp_rate = 0.01;
///
/// // Get filter specifications
/// let specs = filter_specs(capacity, fp_rate).unwrap();
///
/// // Pre-compute fingerprints using the fingerprint size
/// let fingerprints: Vec<u64> = (0..100)
///     .map(|i| compute_fingerprint_with_hasher(&StableBuildHasher, i, specs.fingerprint_bits))
///     .collect();
///
/// // Memory will be between min and max depending on how the filter grows
/// let filter = Filter::new(capacity, fp_rate).unwrap();
/// assert_eq!(filter.fingerprint_size(), specs.fingerprint_bits);
/// assert_eq!(filter.memory_usage(), specs.memory_bytes_max);
/// ```
pub fn filter_specs(capacity: u64, fp_rate: f64) -> Result<FilterSpecs, Error> {
    let slots = calculate_needed_slots(capacity)?;
    let qbits = slots.trailing_zeros() as u8;
    let fp_rate = fp_rate.clamp(MIN_FP_RATE, 0.5);
    let rbits = (-fp_rate.log2()).ceil().max(1.0) as u8;
    let fingerprint_bits = qbits + rbits;
    if fingerprint_bits > 64 {
        return Err(Error::NotEnoughFingerprintBits);
    }

    let memory_bytes_max = usize::try_from(calculate_memory_bytes(qbits, rbits))
        .map_err(|_| Error::CapacityTooLarge)?;
    // Smallest resizable filter has 64 slots (qbits=6)
    let memory_bytes_min = usize::try_from(calculate_memory_bytes(6, fingerprint_bits - 6))
        .map_err(|_| Error::CapacityTooLarge)?;
    // Max capacity: 95% of slots
    let capacity = (slots * 19).div_ceil(20);
    // Max capacity error ratio based on rbits
    let max_error_ratio = 2f64.powi(-(rbits as i32));
    // Storage cost per item: remainder bits + metadata overhead
    let bits_per_item = rbits as f64 + 2.125;

    Ok(FilterSpecs {
        fingerprint_bits,
        memory_bytes_min,
        memory_bytes_max,
        max_capacity: capacity,
        max_error_ratio,
        bits_per_item,
    })
}

/// Calculates memory usage in bytes for a filter with given qbits and rbits.
///
/// Memory layout per block (64 slots):
/// - 1 byte: offset
/// - 8 bytes: occupieds bitmap
/// - 8 bytes: runends bitmap
/// - 8 * rbits bytes: remainders (64 slots * rbits bits / 8)
///
/// Total: num_blocks * (17 + 8 * rbits) + 8 bytes padding.
/// The +8 padding allows branchless remainder reads (always reading 2 u64s).
///
/// Returns `u64` to avoid truncation on 32-bit targets. All callers that need `usize`
/// use checked conversion (`usize::try_from`) and return an error on overflow.
fn calculate_memory_bytes(qbits: u8, rbits: u8) -> u64 {
    let num_blocks = (1u64 << qbits) / 64;
    let block_bytes = 17 + 8 * rbits as u64; // see block layout in doc comment above
    num_blocks * block_bytes + 8
}

/// Calculates the number of slots needed to fit the desired capacity with 95% max occupancy.
/// Returns the number of slots rounded to the next power of two, but always >= 64.
fn calculate_needed_slots(desired: u64) -> Result<u64, Error> {
    let mut slots = desired
        .checked_next_power_of_two()
        .ok_or(Error::CapacityTooLarge)?
        .max(64);
    loop {
        let capacity = slots
            .checked_mul(19)
            .ok_or(Error::CapacityTooLarge)?
            .div_ceil(20);
        if capacity >= desired {
            return Ok(slots);
        }
        slots = slots.checked_mul(2).ok_or(Error::CapacityTooLarge)?;
    }
}

/// Approximate Membership Query Filter (AMQ-Filter) based on the Rank Select Quotient Filter (RSQF).
///
/// This data structure is similar to a hash table that stores fingerprints in a very compact way.
/// Fingerprints are similar to a hash values, but are possibly truncated.
/// The reason for false positives is that multiple items can map to the same fingerprint.
/// For more information see the [quotient filter Wikipedia page](https://en.wikipedia.org/wiki/Quotient_filter)
/// that describes a similar but less optimized version of the data structure.
/// The actual implementation is based on the [Rank Select Quotient Filter (RSQF)](https://dl.acm.org/doi/pdf/10.1145/3035918.3035963).
///
/// The public API also exposes a fingerprint API, which can be used to succinctly store u64
/// hash values.
///
/// The type parameter `B` controls the buffer storage. Use `Filter` (defaults to `Box<[u8]>`)
/// for an owned, mutable filter or [`FilterRef`] (`Filter<&[u8]>`) for a borrowed, read-only view
/// that supports zero-copy deserialization.
#[derive(Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize),
    serde(bound(serialize = "B: serde_bytes::Serialize"))
)]
#[cfg_attr(feature = "jsonschema", derive(JsonSchema))]
pub struct Filter<B = Box<[u8]>, S = StableBuildHasher> {
    #[cfg_attr(
        feature = "serde",
        serde(rename = "b", serialize_with = "serde_bytes::serialize",)
    )]
    #[cfg_attr(feature = "jsonschema", schemars(with = "Vec<u8>"))]
    buffer: B,
    #[cfg_attr(feature = "serde", serde(skip))]
    #[cfg_attr(feature = "jsonschema", schemars(skip))]
    build_hasher: S,
    #[cfg_attr(feature = "serde", serde(rename = "l"))]
    len: u64,
    #[cfg_attr(feature = "serde", serde(rename = "q"))]
    qbits: NonZeroU8,
    #[cfg_attr(feature = "serde", serde(rename = "r"))]
    rbits: NonZeroU8,
    #[cfg_attr(feature = "serde", serde(rename = "m"))]
    max_qbits: Option<NonZeroU8>,
}

/// Raw deserialization target for [`Filter`] that is validated before construction.
#[cfg(feature = "serde")]
#[derive(Deserialize)]
#[serde(bound(deserialize = "B: serde_bytes::Deserialize<'de>"))]
struct FilterUnchecked<B> {
    #[serde(rename = "b", deserialize_with = "serde_bytes::deserialize")]
    buffer: B,
    #[serde(rename = "l")]
    len: u64,
    #[serde(rename = "q")]
    qbits: NonZeroU8,
    #[serde(rename = "r")]
    rbits: NonZeroU8,
    #[serde(rename = "m")]
    max_qbits: Option<NonZeroU8>,
}

#[cfg(feature = "serde")]
impl<B: AsRef<[u8]>> FilterUnchecked<B> {
    fn try_into_filter<S: Default>(self) -> Result<Filter<B, S>, &'static str> {
        let qbits = self.qbits.get();
        let rbits = self.rbits.get();
        // qbits must be at least 6 (64 slots per block), within MAX_QBITS,
        // and fingerprint must fit in 64 bits
        if !(6..=MAX_QBITS).contains(&qbits) || qbits as u16 + rbits as u16 > 64 {
            return Err("invalid qbits/rbits");
        }
        // buffer length must match the expected size for this qbits/rbits
        // Compare as u64 to avoid truncation on 32-bit targets
        let expected_bytes = calculate_memory_bytes(qbits, rbits);
        if self.buffer.as_ref().len() as u64 != expected_bytes {
            return Err("buffer length mismatch");
        }
        // len must not exceed total number of buckets
        let total_buckets = 1u64 << qbits;
        if self.len > total_buckets {
            return Err("len exceeds total buckets");
        }
        // max_qbits must be >= qbits, <= MAX_QBITS, and within fingerprint bounds
        if let Some(max_qbits) = self.max_qbits {
            let mq = max_qbits.get();
            if mq < qbits || mq > MAX_QBITS || mq as u16 > qbits as u16 + rbits as u16 - 1 {
                return Err("invalid max_qbits");
            }
        }
        Ok(Filter {
            buffer: self.buffer,
            build_hasher: S::default(),
            len: self.len,
            qbits: self.qbits,
            rbits: self.rbits,
            max_qbits: self.max_qbits,
        })
    }
}

/// The `AsRef<[u8]>` bound is required to validate the buffer length during deserialization.
///
/// The hasher is not serialized — it is reconstructed via `S::default()` on deserialization.
/// Use [`Filter::with_hasher()`] to set a specific hasher after deserialization.
#[cfg(feature = "serde")]
impl<'de, B: AsRef<[u8]> + serde_bytes::Deserialize<'de>, S: Default> Deserialize<'de>
    for Filter<B, S>
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        FilterUnchecked::<B>::deserialize(deserializer)?
            .try_into_filter()
            .map_err(serde::de::Error::custom)
    }
}

/// A read-only, borrowed view of a [`Filter`].
///
/// This is a type alias for `Filter<&[u8], S>` with `S` defaulting to [`StableBuildHasher`].
/// It supports all read-only operations ([`contains`](Filter::contains),
/// [`count`](Filter::count), [`fingerprints`](Filter::fingerprints), etc.) and enables
/// zero-copy deserialization from binary formats like CBOR, bincode, or postcard via serde.
///
/// # Zero-copy deserialization
///
/// ```rust,ignore
/// // Deserialize without copying the buffer
/// let filter_ref: FilterRef = serde_cbor::from_slice(&bytes).unwrap();
/// assert!(filter_ref.contains("hello"));
/// ```
///
/// Use [`FilterRef::to_owned()`](Filter::to_owned) to convert to a [`Filter`] when mutation is needed.
pub type FilterRef<'a, S = StableBuildHasher> = Filter<&'a [u8], S>;

impl Copy for Filter<&[u8], StableBuildHasher> {}

/// Errors returned by [`Filter`] operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The filter is full and cannot grow further.
    CapacityExceeded,
    /// Filters have incompatible fingerprint sizes (e.g., during [`Filter::merge()`]).
    IncompatibleFingerprintSize,
    /// The requested configuration requires more than 64 bits per fingerprint.
    NotEnoughFingerprintBits,
    /// The requested capacity exceeds [`Filter::MAX_CAPACITY`] or the platform's addressable range.
    CapacityTooLarge,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for Error {}

#[derive(Debug)]
struct Block {
    offset: u64,
    occupieds: u64,
    runends: u64,
}

trait BitExt {
    fn is_bit_set(&self, i: usize) -> bool;
    fn set_bit(&mut self, i: usize);
    fn clear_bit(&mut self, i: usize);
    fn shift_right(&self, bits: usize, b: &Self, b_start: usize, b_end: usize) -> Self;
    fn shift_left(&self, bits: usize, b: &Self, b_start: usize, b_end: usize) -> Self;
    /// Number of set bits (1s) in the range
    fn popcnt(&self, range: impl RangeBounds<u64>) -> u64;
    /// Index of nth set bits in the range
    fn select(&self, range: RangeFrom<u64>, n: u64) -> Option<u64>;

    #[inline]
    fn update_bit(&mut self, i: usize, value: bool) {
        if value {
            self.set_bit(i)
        } else {
            self.clear_bit(i)
        }
    }
}

impl BitExt for u64 {
    #[inline]
    fn is_bit_set(&self, i: usize) -> bool {
        (*self & (1 << i)) != 0
    }

    #[inline]
    fn set_bit(&mut self, i: usize) {
        *self |= 1 << i
    }

    #[inline]
    fn clear_bit(&mut self, i: usize) {
        *self &= !(1 << i)
    }

    #[inline]
    fn shift_right(&self, bits: usize, b: &Self, b_start: usize, b_end: usize) -> Self {
        let a_component = *self >> (64 - bits); // select the highest `bits` from A to become lowest
        let width = b_end - b_start;
        let b_shifted_mask = if width >= 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        } << b_start;
        let b_shifted = ((b_shifted_mask & b) << bits) & b_shifted_mask;
        let b_mask = !b_shifted_mask;

        a_component | b_shifted | (b & b_mask)
    }

    #[inline]
    fn shift_left(&self, bits: usize, b: &Self, b_start: usize, b_end: usize) -> Self {
        let a_component = *self << (64 - bits); // select the lowest `bits` from A to become highest
        let width = b_end - b_start;
        let b_shifted_mask = if width >= 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        } << b_start;
        let b_shifted = ((b_shifted_mask & b) >> bits) & b_shifted_mask;
        let b_mask = !b_shifted_mask;

        a_component | b_shifted | (b & b_mask)
    }

    #[inline]
    fn popcnt(&self, range: impl RangeBounds<u64>) -> u64 {
        let mut v = match range.start_bound() {
            std::ops::Bound::Included(&i) => *self & (u64::MAX << i),
            std::ops::Bound::Excluded(&i) => *self & (u64::MAX << (i + 1)),
            _ => *self,
        };
        v = match range.end_bound() {
            std::ops::Bound::Included(&i) => v & (u64::MAX >> (63 - i)),
            std::ops::Bound::Excluded(&i) => v & ((1u64 << i) - 1),
            _ => v,
        };

        #[cfg(all(
            target_arch = "x86_64",
            not(feature = "legacy_x86_64_support"),
            not(target_feature = "popcnt")
        ))]
        let result = unsafe {
            // Using intrinsics introduce a function call, and the resulting code
            // ends up slower than the inline assembly below.
            // Any calls to is_x86_feature_detected also significantly affect performance.
            // Given this is available on all x64 cpus starting 2008 we assume it's present
            // (unless legacy_x86_64_support is set) and panic elsewhere otherwise.
            let popcnt;
            std::arch::asm!(
                "popcnt {popcnt}, {v}",
                v = in(reg) v,
                popcnt = out(reg) popcnt,
                options(pure, nomem, nostack)
            );
            popcnt
        };
        #[cfg(any(
            not(target_arch = "x86_64"),
            feature = "legacy_x86_64_support",
            target_feature = "popcnt"
        ))]
        let result = v.count_ones() as u64;

        result
    }

    #[inline]
    fn select(&self, range: RangeFrom<u64>, n: u64) -> Option<u64> {
        debug_assert!(range.start < 64);
        let v = *self & (u64::MAX << range.start);

        // x86_64: use BMI2 (assumed at runtime, or guaranteed at compile time)
        #[cfg(all(
            target_arch = "x86_64",
            any(not(feature = "legacy_x86_64_support"), target_feature = "bmi2")
        ))]
        unsafe {
            let result: u64;
            std::arch::asm!(
                "mov     {tmp}, 1",
                "shlx    {tmp}, {tmp}, {n}",
                "pdep    {tmp}, {tmp}, {v}",
                "tzcnt   {tmp}, {tmp}",
                n = in(reg) n,
                v = in(reg) v,
                tmp = out(reg) result,
                options(pure, nomem, nostack)
            );
            if result != 64 {
                Some(result)
            } else {
                None
            }
        }

        // Fallback: non-x86_64 or legacy without BMI2
        #[cfg(not(all(
            target_arch = "x86_64",
            any(not(feature = "legacy_x86_64_support"), target_feature = "bmi2")
        )))]
        {
            crate::portable_select::select(v, n)
        }
    }
}

trait CastNonZeroU8 {
    fn u64(&self) -> u64;
    fn usize(&self) -> usize;
}

impl CastNonZeroU8 for NonZeroU8 {
    #[inline]
    fn u64(&self) -> u64 {
        self.get() as u64
    }

    #[inline]
    fn usize(&self) -> usize {
        self.get() as usize
    }
}

// Read-only methods available on any Filter<B> where the buffer can be read as &[u8].
// This covers both Filter (owned, B=Box<[u8]>) and FilterRef (borrowed, B=&[u8]).
impl<B: AsRef<[u8]>, S> Filter<B, S> {
    /// Replaces the hasher, returning a filter with the new hasher type.
    ///
    /// This is a zero-cost operation — only the hasher is swapped, the stored
    /// fingerprints are unchanged. Useful after deserialization to restore the
    /// hasher that was used when the filter was originally populated.
    ///
    /// **Warning:** On a non-empty filter, `T: Hash` methods will only work
    /// correctly if the new hasher produces the same hashes as the one used
    /// at insertion time. The fingerprint-based API is unaffected.
    ///
    /// ```rust,ignore
    /// // Deserialize, then restore the original hasher used at insertion time.
    /// let f: Filter = serde_cbor::from_slice(&bytes)?;
    /// let f = f.with_hasher(original_hasher);
    /// ```
    pub fn with_hasher<S2>(self, build_hasher: S2) -> Filter<B, S2> {
        Filter {
            buffer: self.buffer,
            build_hasher,
            len: self.len,
            qbits: self.qbits,
            rbits: self.rbits,
            max_qbits: self.max_qbits,
        }
    }

    /// Returns the fingerprint size in bits.
    ///
    /// Use this with [`compute_fingerprint_with_hasher()`] to create compatible fingerprints.
    #[inline]
    pub fn fingerprint_size(&self) -> u8 {
        self.qbits.get() + self.rbits.get()
    }

    /// Returns `true` if the filter contains no items.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of items in the filter.
    #[inline]
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Returns the memory usage in bytes.
    #[inline]
    pub fn memory_usage(&self) -> usize {
        self.buffer.as_ref().len()
    }

    /// Returns the maximum capacity after all possible growth.
    #[inline]
    pub fn max_capacity(&self) -> u64 {
        // Overflow is not possible here as it'd have overflowed in the constructor.
        ((1u64 << self.max_qbits.unwrap_or(self.qbits).get()) * 19).div_ceil(20)
    }

    /// Returns the current capacity (before next growth).
    #[inline]
    pub fn capacity(&self) -> u64 {
        if cfg!(fuzzing) {
            // 100% occupancy is not realistic but stresses the algorithm much more.
            // To generate real counter examples this "pessimisation" must be removed.
            self.total_buckets().get()
        } else {
            // Up to 95% occupancy
            // 19/20 == 0.95
            // Overflow is not possible here as it'd have overflowed in the constructor.
            (self.total_buckets().get() * 19).div_ceil(20)
        }
    }

    /// Returns the false positive rate when fully grown (`len == max_capacity()`).
    pub fn max_error_ratio_resizeable(&self) -> f64 {
        let extra_rbits = self.max_qbits.unwrap_or(self.qbits).get() - self.qbits.get();
        2f64.powi(-((self.rbits.get() - extra_rbits) as i32))
    }

    /// Returns the false positive rate at current capacity (`len == capacity()`).
    pub fn max_error_ratio(&self) -> f64 {
        2f64.powi(-(self.rbits.get() as i32))
    }

    /// Returns the estimated false positive rate at current occupancy.
    pub fn current_error_ratio(&self) -> f64 {
        let occupancy = self.len as f64 / self.total_buckets().get() as f64;
        1.0 - std::f64::consts::E.powf(-occupancy / 2f64.powi(self.rbits.get() as i32))
    }

    /// Returns `true` if the item is probably in the filter, `false` if definitely not.
    ///
    /// May return false positives but never false negatives.
    pub fn contains<T: Hash>(&self, item: T) -> bool
    where
        S: BuildHasher,
    {
        self.contains_fingerprint(self.hash_item(item))
    }

    /// Returns `true` if the fingerprint is probably in the filter, `false` if definitely not.
    ///
    /// Only the lower [`Self::fingerprint_size()`] bits of `hash` are used.
    pub fn contains_fingerprint(&self, hash: u64) -> bool {
        let (hash_bucket_idx, hash_remainder) = self.calc_qr(hash);
        if !self.is_occupied(hash_bucket_idx) {
            return false;
        }
        let mut runstart_idx = self.run_start(hash_bucket_idx);
        loop {
            if hash_remainder == self.get_remainder(runstart_idx) {
                return true;
            }
            if self.is_runend(runstart_idx) {
                return false;
            }
            runstart_idx += 1;
        }
    }

    /// Returns how many times the item appears in the filter (approximate).
    ///
    /// Only meaningful if duplicates were inserted via [`Filter::insert_duplicated()`].
    pub fn count<T: Hash>(&self, item: T) -> u64
    where
        S: BuildHasher,
    {
        self.count_fingerprint(self.hash_item(item))
    }

    /// Returns how many times the fingerprint appears in the filter (approximate).
    ///
    /// Only the lower [`Self::fingerprint_size()`] bits of `hash` are used.
    pub fn count_fingerprint(&self, hash: u64) -> u64 {
        let (hash_bucket_idx, hash_remainder) = self.calc_qr(hash);
        if !self.is_occupied(hash_bucket_idx) {
            return 0;
        }

        let mut count = 0u64;
        let mut runstart_idx = self.run_start(hash_bucket_idx);
        loop {
            if hash_remainder == self.get_remainder(runstart_idx) {
                count += 1;
            }
            if self.is_runend(runstart_idx) {
                return count;
            }
            runstart_idx += 1;
        }
    }

    /// Returns an iterator over the fingerprints stored in the filter.
    ///
    /// Fingerprints are yielded in ascending order. Each value has only the lower
    /// [`Self::fingerprint_size()`] bits set (upper bits are zero).
    ///
    /// This is useful for serialization, migrating data between filters, or
    /// inspecting stored values. Use [`compute_fingerprint_with_hasher()`] to compute a
    /// fingerprint compatible with this filter's size.
    pub fn fingerprints(&self) -> FingerprintIter<'_> {
        FingerprintIter::new(self)
    }

    /// Returns a borrowed, read-only view of this filter.
    ///
    /// When using a custom hasher, `S` must implement `Clone` so it can be
    /// shared with the returned view.
    #[inline]
    pub fn as_filter_ref(&self) -> FilterRef<'_, S>
    where
        S: Clone,
    {
        Filter {
            buffer: self.buffer.as_ref(),
            build_hasher: self.build_hasher.clone(),
            len: self.len,
            qbits: self.qbits,
            rbits: self.rbits,
            max_qbits: self.max_qbits,
        }
    }

    #[inline]
    fn block_byte_size(&self) -> usize {
        1 + 8 + 8 + 64 * self.rbits.usize() / 8
    }

    // SAFETY: Caller must ensure offset + 8 <= buffer.len()
    #[inline(always)]
    unsafe fn read_u64_unchecked(&self, offset: usize) -> u64 {
        debug_assert!(offset + 8 <= self.buffer.as_ref().len());
        u64::from_le_bytes(
            self.buffer
                .as_ref()
                .get_unchecked(offset..offset + 8)
                .try_into()
                .unwrap_unchecked(),
        )
    }

    #[inline]
    fn raw_block(&self, block_num: u64) -> Block {
        let block_num = block_num % self.total_blocks();
        let block_start = block_num as usize * self.block_byte_size();
        // SAFETY: block_num % total_blocks() guarantees valid block index
        unsafe {
            Block {
                offset: *self.buffer.as_ref().get_unchecked(block_start) as u64,
                occupieds: self.read_u64_unchecked(block_start + 1),
                runends: self.read_u64_unchecked(block_start + 1 + 8),
            }
        }
    }

    #[inline(always)]
    fn is_occupied(&self, hash_bucket_idx: u64) -> bool {
        let hash_bucket_idx = hash_bucket_idx % self.total_buckets();
        let block_start = (hash_bucket_idx / 64) as usize * self.block_byte_size();
        let occupieds = unsafe { self.read_u64_unchecked(block_start + 1) };
        occupieds.is_bit_set((hash_bucket_idx % 64) as usize)
    }

    #[inline(always)]
    fn is_runend(&self, hash_bucket_idx: u64) -> bool {
        let hash_bucket_idx = hash_bucket_idx % self.total_buckets();
        let block_start = (hash_bucket_idx / 64) as usize * self.block_byte_size();
        let runends = unsafe { self.read_u64_unchecked(block_start + 1 + 8) };
        runends.is_bit_set((hash_bucket_idx % 64) as usize)
    }

    #[inline(always)]
    fn get_remainder(&self, hash_bucket_idx: u64) -> u64 {
        debug_assert!(self.rbits.get() > 0 && self.rbits.get() < 64);
        let hash_bucket_idx = hash_bucket_idx % self.total_buckets();
        let remainders_start = (hash_bucket_idx / 64) as usize * self.block_byte_size() + 1 + 8 + 8;
        let start_bit_idx = self.rbits.usize() * (hash_bucket_idx % 64) as usize;
        let start_u64 = start_bit_idx / 64;
        let extra_low = start_bit_idx - start_u64 * 64;

        // SAFETY: Always safe due to 8 extra bytes padding at end of buffer
        let rem_part0 = unsafe { self.read_u64_unchecked(remainders_start + start_u64 * 8) };
        let rem_part1 = unsafe { self.read_u64_unchecked(remainders_start + (start_u64 + 1) * 8) };

        let combined = (rem_part0 as u128) | ((rem_part1 as u128) << 64);
        let remainder = (combined >> extra_low) as u64 & ((1u64 << self.rbits.get()) - 1);

        debug_assert!(remainder.leading_zeros() >= 64 - self.rbits.get() as u32);
        remainder
    }

    #[inline]
    fn is_slot_empty(&self, hash_bucket_idx: u64) -> bool {
        let bucket_block_idx = hash_bucket_idx / 64;
        let bucket_intrablock_offset = hash_bucket_idx % 64;
        let bucket_block = self.raw_block(bucket_block_idx);

        if bucket_block.offset > bucket_intrablock_offset {
            return false;
        }

        let num_occupied = bucket_block.occupieds.popcnt(..=bucket_intrablock_offset);
        let num_runends = bucket_block
            .runends
            .popcnt(bucket_block.offset..bucket_intrablock_offset);
        num_occupied == num_runends
    }

    #[inline]
    fn find_first_empty_slot(&self, mut hash_bucket_idx: u64) -> u64 {
        let mut bucket_intrablock_offset = hash_bucket_idx % 64;
        let mut bucket_block = self.raw_block(hash_bucket_idx / 64);

        loop {
            let num_occupied = bucket_block.occupieds.popcnt(..=bucket_intrablock_offset);
            let olb = if bucket_block.offset <= bucket_intrablock_offset {
                num_occupied
                    - bucket_block
                        .runends
                        .popcnt(bucket_block.offset..bucket_intrablock_offset)
            } else {
                bucket_block.offset + num_occupied - bucket_intrablock_offset
            };

            if olb == 0 {
                return hash_bucket_idx % self.total_buckets();
            }

            hash_bucket_idx += olb;
            bucket_intrablock_offset += olb;

            if bucket_intrablock_offset >= 64 {
                bucket_intrablock_offset %= 64;
                bucket_block = self.raw_block(hash_bucket_idx / 64);
            }
        }
    }

    fn find_first_not_shifted_slot(&self, mut hash_bucket_idx: u64) -> u64 {
        loop {
            let run_end = self.run_end(hash_bucket_idx);
            if run_end == hash_bucket_idx {
                return hash_bucket_idx;
            }
            hash_bucket_idx = run_end;
        }
    }

    #[cold]
    fn calc_offset(&self, block_num: u64) -> u64 {
        let total_buckets = self.total_buckets();
        let total_blocks = self.total_blocks();
        let block_num = block_num % total_blocks;

        // Compute offset for a block given run_start of its first bucket.
        let offset_from_run_start = |block: u64, mut run_start: u64| -> u64 {
            let block_start = (block * 64) % total_buckets;
            if run_start < block_start {
                run_start += total_buckets.get();
            }
            run_start - block_start
        };

        // Walk backwards to find a block whose previous block has a cached offset.
        let mut cur_block = block_num;
        loop {
            let prev_block = (cur_block + total_blocks.get() - 1) % total_blocks;
            if self.raw_block(prev_block).offset < u8::MAX as u64 {
                break;
            }
            cur_block = prev_block;
            debug_assert!(cur_block != block_num, "all blocks have overflowed offsets");
        }

        // Bootstrap: compute offset for cur_block via run_start/run_end.
        // Safe: the previous block has a cached offset, so run_end won't recurse.
        let block_start = (cur_block * 64) % total_buckets;
        let mut offset = offset_from_run_start(cur_block, self.run_start(block_start));

        // Walk forward, computing each block's offset using run_end_with_offset
        // to avoid recursion through blocks with overflowed cached offsets.
        while cur_block != block_num {
            let next_block = (cur_block + 1) % total_blocks;
            let last_bucket = (cur_block * 64 + 63) % total_buckets;
            let block = self.raw_block(cur_block);
            let run_end = self.run_end_with_offset(last_bucket, cur_block, &block, offset);
            offset = offset_from_run_start(next_block, (run_end + 1) % total_buckets);
            cur_block = next_block;
        }

        offset
    }

    #[inline]
    fn run_start(&self, hash_bucket_idx: u64) -> u64 {
        let prev_bucket = hash_bucket_idx.wrapping_sub(1) % self.total_buckets();
        (self.run_end(prev_bucket) + 1) % self.total_buckets()
    }

    #[cfg_attr(
        all(
            target_arch = "x86_64",
            any(not(feature = "legacy_x86_64_support"), target_feature = "bmi2")
        ),
        inline(always)
    )]
    fn run_end(&self, hash_bucket_idx: u64) -> u64 {
        let hash_bucket_idx = hash_bucket_idx % self.total_buckets();
        let bucket_block_idx = hash_bucket_idx / 64;
        let bucket_block = self.raw_block(bucket_block_idx);

        let offset = if bucket_block.offset >= u8::MAX as u64 {
            self.calc_offset(bucket_block_idx)
        } else {
            bucket_block.offset
        };
        self.run_end_with_offset(hash_bucket_idx, bucket_block_idx, &bucket_block, offset)
    }

    #[inline(always)]
    fn run_end_with_offset(
        &self,
        hash_bucket_idx: u64,
        bucket_block_idx: u64,
        bucket_block: &Block,
        offset: u64,
    ) -> u64 {
        let bucket_intrablock_offset = hash_bucket_idx % 64;
        let bucket_intrablock_rank = bucket_block.occupieds.popcnt(..=bucket_intrablock_offset);

        if bucket_intrablock_rank == 0 {
            return if offset <= bucket_intrablock_offset {
                hash_bucket_idx
            } else {
                (bucket_block_idx * 64 + offset - 1) % self.total_buckets()
            };
        }

        let mut runend_block_idx = bucket_block_idx + offset / 64;
        let mut runend_ignore_bits = offset % 64;
        let mut runend_block = self.raw_block(runend_block_idx);
        let mut runend_rank = bucket_intrablock_rank - 1;

        loop {
            if let Some(off) = runend_block
                .runends
                .select(runend_ignore_bits.., runend_rank)
            {
                let runend_idx = runend_block_idx * 64 + off;
                return runend_idx.max(hash_bucket_idx) % self.total_buckets();
            }
            runend_rank -= runend_block.runends.popcnt(runend_ignore_bits..);
            runend_block_idx += 1;
            runend_ignore_bits = 0;
            runend_block = self.raw_block(runend_block_idx);
        }
    }

    #[inline]
    fn hash_item<T: Hash>(&self, item: T) -> u64
    where
        S: BuildHasher,
    {
        self.build_hasher.hash_one(&item)
    }

    #[inline]
    fn calc_qr(&self, hash: u64) -> (u64, u64) {
        let hash_bucket_idx = (hash >> self.rbits.get()) & ((1 << self.qbits.get()) - 1);
        let remainder = hash & ((1 << self.rbits.get()) - 1);
        (hash_bucket_idx, remainder)
    }

    #[inline]
    fn total_blocks(&self) -> NonZeroU64 {
        // The way this is calculated ensures the compilers sees that the result is both != 0 and a power of 2,
        // both of which allow the optimizer to generate much faster division/remainder code.
        // Safety: qbits in 6..=63, so (1 << qbits) / 64 is in 1..=2^57
        unsafe { NonZeroU64::new_unchecked((1u64 << self.qbits.get()) / 64) }
    }

    #[inline]
    fn total_buckets(&self) -> NonZeroU64 {
        // The way this is calculated ensures the compilers sees that the result is both != 0 and a power of 2,
        // both of which allow the optimizer to generate much faster division/remainder code.
        // Safety: qbits in 6..=63, so 1 << qbits is in 64..=2^63
        unsafe { NonZeroU64::new_unchecked(1u64 << self.qbits.get()) }
    }
}

impl<B: AsRef<[u8]>, S> std::fmt::Debug for Filter<B, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Filter")
            .field("buffer", &"[..]")
            .field("len", &self.len)
            .field("qbits", &self.qbits)
            .field("rbits", &self.rbits)
            .field("max_qbits", &self.max_qbits)
            .finish()
    }
}

/// Converts a borrowed filter view into an owned [`Filter`].
impl<S: Clone> Filter<&[u8], S> {
    /// Converts this borrowed filter view into an owned filter.
    pub fn to_owned(&self) -> Filter<Box<[u8]>, S> {
        Filter {
            buffer: self.buffer.into(),
            build_hasher: self.build_hasher.clone(),
            len: self.len,
            qbits: self.qbits,
            rbits: self.rbits,
            max_qbits: self.max_qbits,
        }
    }
}

impl<'a> IntoIterator for FilterRef<'a> {
    type Item = u64;
    type IntoIter = FingerprintIter<'a>;

    /// Returns an iterator over the fingerprints stored in the filter.
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        FingerprintIter::from_ref(self)
    }
}

/// An iterator over the fingerprints stored in a [`Filter`] or [`FilterRef`].
///
/// Fingerprints are yielded in ascending order. Each value is a `u64` where only
/// the lower [`Filter::fingerprint_size()`] bits are meaningful (upper bits are zero).
///
/// If duplicates were inserted, the same fingerprint value may be yielded multiple times.
///
/// Created by [`Filter::fingerprints()`] or [`FilterRef::fingerprints()`].
pub struct FingerprintIter<'a> {
    filter: FilterRef<'a>,
    q_bucket_idx: u64,
    r_bucket_idx: u64,
    remaining: u64,
    q_block_idx: u64,
    r_block_idx: u64,
    cached_occupieds: u64,
    cached_runends: u64,
}

impl<'a> FingerprintIter<'a> {
    fn new<B: AsRef<[u8]>, S>(filter: &'a Filter<B, S>) -> Self {
        Self::from_ref(FilterRef {
            buffer: filter.buffer.as_ref(),
            build_hasher: StableBuildHasher,
            len: filter.len,
            qbits: filter.qbits,
            rbits: filter.rbits,
            max_qbits: filter.max_qbits,
        })
    }

    fn from_ref(filter: FilterRef<'a>) -> Self {
        let mut q_block_idx = 0u64;
        let mut cached_occupieds = filter.raw_block(0).occupieds;
        if !filter.is_empty() {
            while cached_occupieds == 0 {
                q_block_idx += 1;
                cached_occupieds = filter.raw_block(q_block_idx).occupieds;
            }
        }
        let q_bucket_idx = q_block_idx * 64 + cached_occupieds.trailing_zeros() as u64;
        let r_bucket_idx = filter.run_start(q_bucket_idx);
        let r_block_idx = r_bucket_idx / 64;
        let cached_runends = filter.raw_block(r_block_idx).runends;

        Self {
            filter,
            q_bucket_idx,
            r_bucket_idx,
            remaining: filter.len,
            q_block_idx,
            r_block_idx,
            cached_occupieds,
            cached_runends,
        }
    }
}

impl Iterator for FingerprintIter<'_> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.remaining = self.remaining.checked_sub(1)?;

        let hash = (self.q_bucket_idx << self.filter.rbits.get())
            | self.filter.get_remainder(self.r_bucket_idx);

        let is_runend = self
            .cached_runends
            .is_bit_set((self.r_bucket_idx % 64) as usize);
        if is_runend {
            // Move to next occupied quotient bucket
            self.q_bucket_idx += 1;
            if self.q_bucket_idx / 64 != self.q_block_idx {
                self.q_block_idx = self.q_bucket_idx / 64;
                self.cached_occupieds = self.filter.raw_block(self.q_block_idx).occupieds;
            }

            // Find next occupied using trailing_zeros
            let mut masked = self.cached_occupieds & (u64::MAX << (self.q_bucket_idx % 64));
            while masked == 0 {
                self.q_block_idx += 1;
                self.cached_occupieds = self.filter.raw_block(self.q_block_idx).occupieds;
                masked = self.cached_occupieds;
            }
            self.q_bucket_idx = self.q_block_idx * 64 + masked.trailing_zeros() as u64;
            self.r_bucket_idx = (self.r_bucket_idx + 1).max(self.q_bucket_idx);
        } else {
            self.r_bucket_idx += 1;
        }

        // Update runends cache if needed
        if self.r_bucket_idx / 64 != self.r_block_idx {
            self.r_block_idx = self.r_bucket_idx / 64;
            self.cached_runends = self.filter.raw_block(self.r_block_idx).runends;
        }

        Some(hash)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.remaining.try_into().unwrap_or(usize::MAX),
            self.remaining.try_into().ok(),
        )
    }
}

#[cfg(target_pointer_width = "64")]
impl ExactSizeIterator for FingerprintIter<'_> {}

impl std::iter::FusedIterator for FingerprintIter<'_> {}

impl<'a> IntoIterator for &'a Filter {
    type Item = u64;
    type IntoIter = FingerprintIter<'a>;

    /// Returns an iterator over the fingerprints stored in the filter.
    ///
    /// Equivalent to calling [`Filter::fingerprints()`].
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.fingerprints()
    }
}

/// Builder for constructing a [`Filter`] from fingerprints in sorted (ascending) order.
///
/// This is significantly faster than repeated [`Filter::insert_fingerprint()`] calls
/// when fingerprints are known to be in non-decreasing order, as it avoids run boundary
/// lookups, linear scans, and element shifting — reducing each insertion to a simple
/// sequential write.
///
/// Create a builder by passing an empty [`Filter`] to [`Builder::new()`], then
/// call [`Self::into_filter()`] to retrieve the populated filter.
///
/// # Example
///
/// ```rust
/// let mut inserter = qfilter::Builder::new(qfilter::Filter::new(1000, 0.01).unwrap());
/// let fp_size = inserter.fingerprint_size();
///
/// // Compute truncated fingerprints and sort them.
/// // Builder requires fingerprints in non-decreasing order.
/// let mut fps: Vec<u64> = (0..100u64)
///     .map(|i| qfilter::compute_fingerprint_with_hasher(&qfilter::StableBuildHasher, i, fp_size))
///     .collect();
/// fps.sort();
///
/// for fp in fps {
///     inserter.insert_fingerprint(false, fp).unwrap();
/// }
/// let filter = inserter.into_filter();
///
/// for i in 0..100u64 {
///     assert!(filter.contains(i));
/// }
/// ```
#[derive(Debug)]
pub struct Builder<S = StableBuildHasher> {
    filter: Filter<Box<[u8]>, S>,
    next_slot: u64,
    last_quotient: u64,
    last_remainder: u64,
}

impl<S> Builder<S> {
    /// Creates a builder from an empty filter.
    ///
    /// Use this with [`Filter::new_with_hasher()`] or other constructors to
    /// build a filter with a custom hasher using sorted fingerprint insertion.
    ///
    /// # Panics
    ///
    /// Panics if the filter is not empty.
    pub fn new(filter: Filter<Box<[u8]>, S>) -> Self {
        assert!(filter.is_empty(), "Builder requires an empty filter");
        Self {
            filter,
            next_slot: 0,
            last_quotient: 0,
            last_remainder: 0,
        }
    }

    /// Returns the fingerprint size (in bits) of the filter being built.
    pub fn fingerprint_size(&self) -> u8 {
        self.filter.fingerprint_size()
    }

    /// Returns the current capacity of the filter being built.
    pub fn capacity(&self) -> u64 {
        self.filter.capacity()
    }

    /// Consumes the builder and returns the constructed filter.
    pub fn into_filter(self) -> Filter<Box<[u8]>, S> {
        self.filter
    }
}

impl<S: Clone> Builder<S> {
    /// Creates a builder for internal rebuild paths (grow, shrink).
    fn with_qr_and_hasher(
        qbits: NonZeroU8,
        rbits: NonZeroU8,
        max_qbits: Option<NonZeroU8>,
        build_hasher: S,
    ) -> Result<Self, Error> {
        let mut filter = Filter::with_qr_and_hasher(qbits, rbits, build_hasher)?;
        filter.max_qbits = max_qbits;
        Ok(Self::new(filter))
    }

    /// Inserts a fingerprint that must be >= all previously inserted fingerprints.
    ///
    /// # Parameters
    ///
    /// - `duplicate`: If `true`, insert even if the fingerprint equals the previous one.
    ///   If `false`, skip duplicates (return `Ok(false)`).
    /// - `hash`: The fingerprint value. Only the lower
    ///   [`Filter::fingerprint_size()`] bits are used.
    ///
    /// # Returns
    ///
    /// - `Ok(true)` if inserted successfully.
    /// - `Ok(false)` if already present and `duplicate` is `false`.
    /// - `Err(Error::CapacityExceeded)` if the filter is full and cannot grow.
    ///
    /// # Panics
    ///
    /// Panics if `hash` is not in non-decreasing order relative to the
    /// previously inserted fingerprint.
    #[inline] // see insert_impl
    pub fn insert_fingerprint(&mut self, duplicate: bool, hash: u64) -> Result<bool, Error> {
        match self.insert_impl(duplicate, hash) {
            Ok(inserted) => Ok(inserted),
            Err(Error::CapacityExceeded) => {
                *self = self.filter.rebuild_grown()?;
                self.insert_impl(duplicate, hash)
            }
            Err(e) => Err(e),
        }
    }

    /// Core sorted insertion. Returns `Ok(true)` if a new element was inserted,
    /// `Ok(false)` if it was a duplicate and `duplicate` is `false`.
    // This is optimized enough that the function call overhead is noticeable,
    // so we force inline it into the caller (insert_fingerprint) always.
    // For instance, this makes merge about 25% faster.
    #[inline(always)]
    fn insert_impl(&mut self, duplicate: bool, hash: u64) -> Result<bool, Error> {
        let (quotient, remainder) = self.filter.calc_qr(hash);

        if !self.filter.is_empty() {
            assert!(
                quotient > self.last_quotient
                    || (quotient == self.last_quotient && remainder >= self.last_remainder),
                "fingerprints must be in non-decreasing order: ({quotient}, {remainder}) < ({}, {})",
                self.last_quotient,
                self.last_remainder
            );

            // Duplicate detection — since input is sorted, duplicates are consecutive.
            if quotient == self.last_quotient && remainder == self.last_remainder && !duplicate {
                return Ok(false);
            }

            if self.filter.len >= self.filter.capacity() {
                return Err(Error::CapacityExceeded);
            }

            // When next_slot wraps past total_buckets, the simple sequential-append
            // assumption breaks down — fall back to the general insertion path.
            if self.next_slot >= self.filter.total_buckets().get() {
                self.last_quotient = quotient;
                self.last_remainder = remainder;
                let max_count = if duplicate { u64::MAX } else { 1 };
                let count = self.filter.insert_impl(max_count, hash)?;
                return Ok(count < max_count);
            }
        }

        // Place the element at the appropriate slot.
        let slot = if quotient >= self.next_slot {
            // Canonical slot is free — start a new run here.
            self.filter.set_occupied(quotient, true);
            quotient
        } else {
            if quotient == self.last_quotient {
                // Extending current run: clear previous runend.
                self.filter.set_runend(self.next_slot - 1, false);
            } else {
                // New quotient, but spilled past canonical slot.
                self.filter.set_occupied(quotient, true);
            }
            self.filter.inc_offsets(quotient, self.next_slot);
            self.next_slot
        };
        self.filter.set_runend(slot, true);
        self.filter.set_remainder(slot, remainder);
        self.next_slot = slot + 1;
        self.last_quotient = quotient;
        self.last_remainder = remainder;
        self.filter.len += 1;
        Ok(true)
    }
}

/// Maximum log2 number of slots that can be used in the filter.
/// Effectively, the largest power of 2 that can be multiplied by 19 without overflowing u64.
const MAX_QBITS: u8 = 59;

/// Smallest representable false positive rate (2^-64). A lower rate would imply more
/// than 64 fingerprint bits, which can't be addressed and would overflow the `u8` bit
/// counts; such requests return [`Error::NotEnoughFingerprintBits`] instead of panicking.
const MIN_FP_RATE: f64 = 1.0 / (1u64 << 63) as f64 / 2.0;

impl<B, S> Filter<B, S> {
    /// Maximum number of items that can be stored in the filter: ceil(2^59 * 19 / 20)
    pub const MAX_CAPACITY: u64 = crate::MAX_CAPACITY;
}

/// Maximum number of items that can be stored in the filter: ceil(2^59 * 19 / 20)
const MAX_CAPACITY: u64 = (2u64.pow(MAX_QBITS as u32) * 19).div_ceil(20);

impl Filter {
    /// Creates a new filter with the given capacity and false positive rate.
    ///
    /// # Parameters
    ///
    /// - `capacity`: Minimum number of items the filter can hold.
    /// - `fp_rate`: Upper bound on false positive probability (clamped to (0, 0.5]).
    ///   The actual rate may be lower due to internal rounding.
    ///
    /// # Errors
    ///
    /// - [`Error::CapacityTooLarge`] if capacity exceeds [`Self::MAX_CAPACITY`].
    /// - [`Error::NotEnoughFingerprintBits`] if the configuration isn't achievable.
    #[inline]
    pub fn new(capacity: u64, fp_rate: f64) -> Result<Self, Error> {
        Self::new_resizeable(capacity, capacity, fp_rate)
    }

    /// Creates a resizeable filter that can grow from `initial_capacity` to `max_capacity`.
    ///
    /// The filter starts small and automatically grows when full. The `fp_rate` applies
    /// at `max_capacity`; smaller sizes have proportionally lower error rates.
    ///
    /// # Parameters
    ///
    /// - `initial_capacity`: Starting capacity.
    /// - `max_capacity`: Maximum capacity after growth (must be >= `initial_capacity`).
    /// - `fp_rate`: Upper bound on false positive rate at max capacity (clamped to (0, 0.5]).
    ///   The actual rate may be lower due to internal rounding.
    ///
    /// # Errors
    ///
    /// - [`Error::CapacityTooLarge`] if `max_capacity` exceeds [`Self::MAX_CAPACITY`].
    /// - [`Error::NotEnoughFingerprintBits`] if the configuration isn't achievable.
    pub fn new_resizeable(
        initial_capacity: u64,
        max_capacity: u64,
        fp_rate: f64,
    ) -> Result<Self, Error> {
        assert!(max_capacity >= initial_capacity);
        let slots_for_capacity = calculate_needed_slots(initial_capacity)?;
        let qbits = slots_for_capacity.trailing_zeros() as u8;
        let slots_for_max_capacity = calculate_needed_slots(max_capacity)?;
        let max_qbits = slots_for_max_capacity.trailing_zeros() as u8;
        let fp_rate = fp_rate.clamp(MIN_FP_RATE, 0.5);
        let rbits = (-fp_rate.log2()).ceil().max(1.0) as u8 + (max_qbits - qbits);
        let mut result = Self::with_qr(qbits.try_into().unwrap(), rbits.try_into().unwrap())?;
        if max_qbits > qbits {
            result.max_qbits = Some(max_qbits.try_into().unwrap());
        }
        Ok(result)
    }

    /// Creates a new resizeable filter with a specific fingerprint bit size.
    ///
    /// Use this when storing pre-computed fingerprints via [`Self::insert_fingerprint()`].
    /// Use [`compute_fingerprint_with_hasher()`] to compute fingerprints with a specific bit size.
    ///
    /// # Parameters
    ///
    /// - `initial_capacity`: Minimum number of items the filter can hold initially.
    /// - `fingerprint_bits`: Bits per fingerprint (7..=64). Larger values reduce false positives.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotEnoughFingerprintBits`] if `fingerprint_bits` is outside 7..=64
    /// or too small for the requested capacity.
    pub fn with_fingerprint_size(
        initial_capacity: u64,
        fingerprint_bits: u8,
    ) -> Result<Self, Error> {
        if !(7..=64).contains(&fingerprint_bits) {
            return Err(Error::NotEnoughFingerprintBits);
        }
        let slots_for_capacity = calculate_needed_slots(initial_capacity)?;
        let qbits = slots_for_capacity.trailing_zeros() as u8;
        if fingerprint_bits <= qbits {
            return Err(Error::NotEnoughFingerprintBits);
        }
        let rbits = fingerprint_bits - qbits;
        let mut result = Self::with_qr(qbits.try_into().unwrap(), rbits.try_into().unwrap())?;
        if rbits > 1 {
            result.max_qbits = Some((qbits + rbits - 1).min(MAX_QBITS).try_into().unwrap());
        }
        Ok(result)
    }

    fn with_qr(qbits: NonZeroU8, rbits: NonZeroU8) -> Result<Self, Error> {
        Self::with_qr_and_hasher(qbits, rbits, StableBuildHasher)
    }

    fn check_cpu_support() {
        #[cfg(all(
            target_arch = "x86_64",
            not(feature = "legacy_x86_64_support"),
            not(target_feature = "popcnt")
        ))]
        assert!(
            std::is_x86_feature_detected!("popcnt"),
            "CPU doesn't support the popcnt instruction"
        );
        #[cfg(all(
            target_arch = "x86_64",
            not(feature = "legacy_x86_64_support"),
            not(target_feature = "bmi2")
        ))]
        assert!(
            std::is_x86_feature_detected!("bmi2"),
            "CPU doesn't support the bmi2 instructions"
        );
    }
}

impl<S> Filter<Box<[u8]>, S> {
    fn with_qr_and_hasher(
        qbits: NonZeroU8,
        rbits: NonZeroU8,
        build_hasher: S,
    ) -> Result<Self, Error> {
        Filter::check_cpu_support();
        if qbits.get() + rbits.get() > 64 {
            return Err(Error::NotEnoughFingerprintBits);
        }
        let buffer_bytes = usize::try_from(calculate_memory_bytes(qbits.get(), rbits.get()))
            .map_err(|_| Error::CapacityTooLarge)?;
        let buffer = vec![0u8; buffer_bytes].into_boxed_slice();
        Ok(Self {
            buffer,
            build_hasher,
            qbits,
            rbits,
            len: 0,
            max_qbits: None,
        })
    }
}

impl<S: Clone> Filter<Box<[u8]>, S> {
    /// Creates a new filter with the given capacity, false positive rate, and custom hasher.
    ///
    /// The same hasher (and seed) must be used for all operations on this filter.
    /// The hasher is not serialized. On deserialization it is reconstructed via
    /// `S::default()`. If that doesn't produce the correct hasher, use
    /// [`Filter::with_hasher()`] to restore it.
    ///
    /// See [`Filter::new()`] for parameter details.
    #[inline]
    pub fn new_with_hasher(capacity: u64, fp_rate: f64, build_hasher: S) -> Result<Self, Error> {
        Self::new_resizeable_with_hasher(capacity, capacity, fp_rate, build_hasher)
    }

    /// Creates a resizeable filter with a custom hasher.
    ///
    /// See [`Filter::new_resizeable()`] and [`Filter::new_with_hasher()`] for details.
    pub fn new_resizeable_with_hasher(
        initial_capacity: u64,
        max_capacity: u64,
        fp_rate: f64,
        build_hasher: S,
    ) -> Result<Self, Error> {
        assert!(max_capacity >= initial_capacity);
        let slots_for_capacity = calculate_needed_slots(initial_capacity)?;
        let qbits = slots_for_capacity.trailing_zeros() as u8;
        let slots_for_max_capacity = calculate_needed_slots(max_capacity)?;
        let max_qbits = slots_for_max_capacity.trailing_zeros() as u8;
        let fp_rate = fp_rate.clamp(MIN_FP_RATE, 0.5);
        let rbits = (-fp_rate.log2()).ceil().max(1.0) as u8 + (max_qbits - qbits);
        let mut result = Self::with_qr_and_hasher(
            qbits.try_into().unwrap(),
            rbits.try_into().unwrap(),
            build_hasher,
        )?;
        if max_qbits > qbits {
            result.max_qbits = Some(max_qbits.try_into().unwrap());
        }
        Ok(result)
    }

    /// Creates a filter with a specific fingerprint bit size and custom hasher.
    ///
    /// See [`Filter::with_fingerprint_size()`] and [`Filter::new_with_hasher()`] for details.
    pub fn with_fingerprint_size_and_hasher(
        initial_capacity: u64,
        fingerprint_bits: u8,
        build_hasher: S,
    ) -> Result<Self, Error> {
        if !(7..=64).contains(&fingerprint_bits) {
            return Err(Error::NotEnoughFingerprintBits);
        }
        let slots_for_capacity = calculate_needed_slots(initial_capacity)?;
        let qbits = slots_for_capacity.trailing_zeros() as u8;
        if fingerprint_bits <= qbits {
            return Err(Error::NotEnoughFingerprintBits);
        }
        let rbits = fingerprint_bits - qbits;
        let mut result = Self::with_qr_and_hasher(
            qbits.try_into().unwrap(),
            rbits.try_into().unwrap(),
            build_hasher,
        )?;
        if rbits > 1 {
            result.max_qbits = Some((qbits + rbits - 1).min(MAX_QBITS).try_into().unwrap());
        }
        Ok(result)
    }

    /// Removes all items from the filter.
    pub fn clear(&mut self) {
        self.buffer.fill(0);
        self.len = 0;
    }

    #[inline]
    fn set_block_runends(&mut self, block_num: u64, runends: u64) {
        let block_num = block_num % self.total_blocks();
        let block_start = block_num as usize * self.block_byte_size();
        // SAFETY: block_num % total_blocks() guarantees valid block index
        unsafe { self.write_u64_unchecked(block_start + 1 + 8, runends) };
    }

    /// Write u64 to buffer at given offset without bounds checking.
    /// SAFETY: Caller must ensure offset + 8 <= buffer.len()
    #[inline(always)]
    unsafe fn write_u64_unchecked(&mut self, offset: usize, value: u64) {
        debug_assert!(offset + 8 <= self.buffer.len());
        self.buffer
            .get_unchecked_mut(offset..offset + 8)
            .copy_from_slice(&value.to_le_bytes());
    }

    #[inline]
    fn adjust_block_offset(&mut self, block_num: u64, inc: bool) {
        let block_num = block_num % self.total_blocks();
        let block_start = block_num as usize * self.block_byte_size();
        // SAFETY: block_num % total_blocks() guarantees valid block index
        let current = unsafe { *self.buffer.get_unchecked(block_start) };
        let new_value = if inc {
            current.saturating_add(1)
        } else if current != u8::MAX {
            current - 1
        } else {
            self.calc_offset(block_num).try_into().unwrap_or(u8::MAX)
        };
        unsafe { *self.buffer.get_unchecked_mut(block_start) = new_value };
    }

    #[inline]
    fn inc_offsets(&mut self, start_bucket: u64, end_bucket: u64) {
        let original_block = start_bucket / 64;
        let mut last_affected_block = end_bucket / 64;
        if end_bucket < start_bucket {
            last_affected_block += self.total_blocks().get();
        }
        for b in original_block + 1..=last_affected_block {
            self.adjust_block_offset(b, true);
        }
    }

    #[inline]
    fn dec_offsets(&mut self, start_bucket: u64, end_bucket: u64) {
        let original_block = start_bucket / 64;
        let mut last_affected_block = end_bucket / 64;
        if end_bucket < start_bucket {
            last_affected_block += self.total_blocks().get();
        }

        // As an edge case we may decrement the offsets of 2+ blocks and the block B' offset
        // may be saturated and depend on a previous Block B" with a non saturated offset.
        // But B" offset may also(!) be affected by the decremented operation, so we must
        // decrement B" offset first before the remaining offsets.
        if last_affected_block - original_block >= 2
            && self.raw_block(original_block + 1).offset >= u8::MAX as u64
        {
            // last affected block offset is always <= 64 (BLOCK SIZE)
            // otherwise the decrement operation would be to affecting a subsequent block
            debug_assert!(self.raw_block(last_affected_block).offset <= 64);
            self.adjust_block_offset(last_affected_block, false);
            last_affected_block -= 1;
        }
        for b in original_block + 1..=last_affected_block {
            self.adjust_block_offset(b, false);
        }

        #[cfg(fuzzing)]
        self.validate_offsets(original_block, last_affected_block);
    }

    #[cfg(any(fuzzing, test))]
    fn validate_offsets(&mut self, original_block: u64, last_affected_block: u64) {
        for b in original_block..=last_affected_block {
            let raw_offset = self.raw_block(b).offset;
            let offset = self.calc_offset(b);
            debug_assert!(
                (raw_offset >= u8::MAX as u64 && offset >= u8::MAX as u64)
                    || (offset == raw_offset),
                "block {} offset {} calc {}",
                b,
                raw_offset,
                offset,
            );
        }
    }

    #[inline(always)]
    fn set_occupied(&mut self, hash_bucket_idx: u64, value: bool) {
        let hash_bucket_idx = hash_bucket_idx % self.total_buckets();
        let block_start = (hash_bucket_idx / 64) as usize * self.block_byte_size();
        // SAFETY: hash_bucket_idx % total_buckets() guarantees valid block index
        let mut occupieds = unsafe { self.read_u64_unchecked(block_start + 1) };
        occupieds.update_bit((hash_bucket_idx % 64) as usize, value);
        unsafe { self.write_u64_unchecked(block_start + 1, occupieds) };
    }

    #[inline(always)]
    fn set_runend(&mut self, hash_bucket_idx: u64, value: bool) {
        let hash_bucket_idx = hash_bucket_idx % self.total_buckets();
        let block_start = (hash_bucket_idx / 64) as usize * self.block_byte_size();
        // SAFETY: hash_bucket_idx % total_buckets() guarantees valid block index
        let mut runends = unsafe { self.read_u64_unchecked(block_start + 1 + 8) };
        runends.update_bit((hash_bucket_idx % 64) as usize, value);
        unsafe { self.write_u64_unchecked(block_start + 1 + 8, runends) };
    }

    #[inline(always)]
    fn set_remainder(&mut self, hash_bucket_idx: u64, remainder: u64) {
        debug_assert!(self.rbits.get() > 0 && self.rbits.get() < 64);
        debug_assert!(remainder.leading_zeros() >= 64 - self.rbits.get() as u32);
        let hash_bucket_idx = hash_bucket_idx % self.total_buckets();
        let remainders_start = (hash_bucket_idx / 64) as usize * self.block_byte_size() + 1 + 8 + 8;
        let start_bit_idx = self.rbits.usize() * (hash_bucket_idx % 64) as usize;
        let start_u64 = start_bit_idx / 64;
        let extra_low = start_bit_idx - start_u64 * 64;

        // SAFETY: Always safe due to 8 extra bytes padding at end of buffer
        let offset = remainders_start + start_u64 * 8;
        let rem_part0 = unsafe { self.read_u64_unchecked(offset) };
        let rem_part1 = unsafe { self.read_u64_unchecked(offset + 8) };

        // Combine as 128-bit, clear remainder region, set new remainder (branchless)
        let combined = (rem_part0 as u128) | ((rem_part1 as u128) << 64);
        let rbits_mask = ((1u128 << self.rbits.get()) - 1) << extra_low;
        let new_combined = (combined & !rbits_mask) | ((remainder as u128) << extra_low);

        // Split back and write both
        unsafe {
            self.write_u64_unchecked(offset, new_combined as u64);
            self.write_u64_unchecked(offset + 8, (new_combined >> 64) as u64);
        }
    }

    /// Converts an inclusive end index to exclusive, handling wrap-around.
    #[inline]
    fn wrap_end_exclusive(&self, start: u64, end_inclusive: u64) -> u64 {
        if end_inclusive < start {
            end_inclusive + self.total_buckets().get() + 1
        } else {
            end_inclusive + 1
        }
    }

    fn shift_remainders_by_1(&mut self, start: u64, end_inc: u64) {
        let end = self.wrap_end_exclusive(start, end_inc);
        let mut end_u64 = end * self.rbits.u64() / 64;
        let mut bend = (end * self.rbits.u64() % 64) as usize;
        let start_u64 = start * self.rbits.u64() / 64;
        let bstart = (start * self.rbits.u64() % 64) as usize;

        let rbits = self.rbits.get() as usize;
        let total_blocks = self.total_blocks().get();
        let block_byte_size = self.block_byte_size();

        // Track position as (block_idx, rem_idx) to avoid division in loop
        let mut block_idx = (end_u64 / rbits as u64) % total_blocks;
        let mut rem_idx = (end_u64 % rbits as u64) as usize;

        // Byte offset for a given (block_idx, rem_idx)
        let offset = |b: u64, r: usize| (b as usize * block_byte_size) + 17 + r * 8;

        // Shift from end to start, caching the previous value
        if end_u64 != start_u64 {
            let mut cached_rem = unsafe { self.read_u64_unchecked(offset(block_idx, rem_idx)) };

            while end_u64 != start_u64 {
                // Compute previous position
                let (prev_block, prev_rem) = if rem_idx > 0 {
                    (block_idx, rem_idx - 1)
                } else {
                    (
                        if block_idx > 0 {
                            block_idx - 1
                        } else {
                            total_blocks - 1
                        },
                        rbits - 1,
                    )
                };

                let prev = unsafe { self.read_u64_unchecked(offset(prev_block, prev_rem)) };
                let shifted = prev.shift_right(rbits, &cached_rem, 0, bend);
                unsafe { self.write_u64_unchecked(offset(block_idx, rem_idx), shifted) };

                cached_rem = prev;
                block_idx = prev_block;
                rem_idx = prev_rem;
                end_u64 -= 1;
                bend = 64;
            }
        }

        // Handle start position (block_idx/rem_idx already there after loop)
        let mut rem_val = unsafe { self.read_u64_unchecked(offset(block_idx, rem_idx)) };
        rem_val = 0u64.shift_right(rbits, &rem_val, bstart, bend);
        unsafe { self.write_u64_unchecked(offset(block_idx, rem_idx), rem_val) };
    }

    fn shift_remainders_back_by_1(&mut self, start: u64, end_inc: u64) {
        let end = self.wrap_end_exclusive(start, end_inc);
        let end_u64 = end * self.rbits.u64() / 64;
        let bend = (end * self.rbits.u64() % 64) as usize;
        let mut start_u64 = start * self.rbits.u64() / 64;
        let mut bstart = (start * self.rbits.u64() % 64) as usize;

        let rbits = self.rbits.get() as usize;
        let total_blocks = self.total_blocks().get();
        let block_byte_size = self.block_byte_size();

        // Track position as (block_idx, rem_idx) to avoid division in loop
        let mut block_idx = (start_u64 / rbits as u64) % total_blocks;
        let mut rem_idx = (start_u64 % rbits as u64) as usize;

        // Byte offset for a given (block_idx, rem_idx)
        let offset = |b: u64, r: usize| (b as usize * block_byte_size) + 17 + r * 8;

        // Shift from start to end, caching the previous value
        if end_u64 != start_u64 {
            let mut cached_rem = unsafe { self.read_u64_unchecked(offset(block_idx, rem_idx)) };

            while end_u64 != start_u64 {
                // Compute next position
                let (next_block, next_rem) = if rem_idx + 1 < rbits {
                    (block_idx, rem_idx + 1)
                } else {
                    ((block_idx + 1) % total_blocks, 0)
                };

                let next = unsafe { self.read_u64_unchecked(offset(next_block, next_rem)) };
                let shifted = next.shift_left(rbits, &cached_rem, bstart, 64);
                unsafe { self.write_u64_unchecked(offset(block_idx, rem_idx), shifted) };

                cached_rem = next;
                block_idx = next_block;
                rem_idx = next_rem;
                start_u64 += 1;
                bstart = 0;
            }
        }

        // Handle end position (block_idx/rem_idx already there after loop)
        let mut rem_val = unsafe { self.read_u64_unchecked(offset(block_idx, rem_idx)) };
        rem_val = 0u64.shift_left(rbits, &rem_val, bstart, bend);
        unsafe { self.write_u64_unchecked(offset(block_idx, rem_idx), rem_val) };
    }

    fn shift_runends_by_1(&mut self, start: u64, end_inc: u64) {
        let end = self.wrap_end_exclusive(start, end_inc);
        let mut end_block = end / 64;
        let mut bend = (end % 64) as usize;
        let start_block = start / 64;
        let bstart = (start % 64) as usize;
        while end_block != start_block {
            let prev_block_runends = self.raw_block(end_block - 1).runends;
            let mut block_runends = self.raw_block(end_block).runends;
            block_runends = prev_block_runends.shift_right(1, &block_runends, 0, bend);
            self.set_block_runends(end_block, block_runends);
            end_block -= 1;
            bend = 64;
        }
        let mut block_runends = self.raw_block(start_block).runends;
        block_runends = 0u64.shift_right(1, &block_runends, bstart, bend);
        self.set_block_runends(start_block, block_runends);
    }

    fn shift_runends_back_by_1(&mut self, start: u64, end_inc: u64) {
        let end = self.wrap_end_exclusive(start, end_inc);
        let end_block = end / 64;
        let bend = (end % 64) as usize;
        let mut start_block = start / 64;
        let mut bstart = (start % 64) as usize;
        while start_block != end_block {
            let next_block_runends = self.raw_block(start_block + 1).runends;
            let mut block_runends = self.raw_block(start_block).runends;
            block_runends = next_block_runends.shift_left(1, &block_runends, bstart, 64);
            self.set_block_runends(start_block, block_runends);
            start_block += 1;
            bstart = 0;
        }
        let mut block_runends = self.raw_block(end_block).runends;
        block_runends = 0u64.shift_left(1, &block_runends, bstart, bend);
        self.set_block_runends(end_block, block_runends);
    }

    /// Removes `item` from the filter. Returns `true` if found and removed.
    ///
    /// **Warning:** Removing an item that wasn't inserted may cause false negatives
    /// for other items with colliding fingerprints.
    pub fn remove<T: Hash>(&mut self, item: T) -> bool
    where
        S: BuildHasher,
    {
        self.remove_fingerprint(self.hash_item(item))
    }

    /// Removes the fingerprint from the filter. Returns `true` if found and removed.
    ///
    /// Only the lower [`Self::fingerprint_size()`] bits of `hash` are used.
    ///
    /// **Warning:** Removing a fingerprint that wasn't inserted may cause false negatives
    /// for other items with colliding fingerprints.
    pub fn remove_fingerprint(&mut self, hash: u64) -> bool {
        let (hash_bucket_idx, hash_remainder) = self.calc_qr(hash);
        if !self.is_occupied(hash_bucket_idx) {
            return false;
        }
        let mut run_start = self.run_start(hash_bucket_idx);
        // adjust run_start so we can have
        // hash_bucket_idx <= run_start <= found_idx <= run_end
        if run_start < hash_bucket_idx {
            run_start += self.total_buckets().get();
        }
        let mut run_end = run_start;
        let mut found_idx = None;
        let found_idx = loop {
            if hash_remainder == self.get_remainder(run_end) {
                found_idx = Some(run_end);
            }
            if self.is_runend(run_end) {
                if let Some(i) = found_idx {
                    break i;
                } else {
                    return false;
                };
            }
            run_end += 1;
        };

        let mut last_bucket_shifted_run_end = run_end;
        if last_bucket_shifted_run_end != hash_bucket_idx {
            last_bucket_shifted_run_end = self.find_first_not_shifted_slot(run_end);
            if last_bucket_shifted_run_end < run_end {
                last_bucket_shifted_run_end += self.total_buckets().get();
            }
        }

        // run_end points to the end of the run (inc) which contains the target remainder (found_idx)
        // If we had a single remainder in the run the run is no more
        if run_end == run_start {
            self.set_occupied(hash_bucket_idx, false);
        } else {
            // More than one remainder in the run.
            // If the removed rem is the last one in the run
            // the before last remainder becomes the new runend.
            if found_idx == run_end {
                self.set_runend(run_end - 1, true);
            }
        }
        if found_idx != last_bucket_shifted_run_end {
            self.set_remainder(found_idx, 0);
            self.shift_remainders_back_by_1(found_idx, last_bucket_shifted_run_end);
            self.shift_runends_back_by_1(found_idx, last_bucket_shifted_run_end);
        }
        self.set_runend(last_bucket_shifted_run_end, false);
        self.set_remainder(last_bucket_shifted_run_end, 0);
        self.dec_offsets(hash_bucket_idx, last_bucket_shifted_run_end);
        self.len -= 1;
        true
    }

    /// Inserts `item`, allowing duplicates.
    ///
    /// Use this when the filter supports removals and you're re-adding a previously
    /// removed item. See also [`Self::insert_counting()`] for bounded duplicates.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CapacityExceeded`] if the filter is full.
    #[inline]
    pub fn insert_duplicated<T: Hash>(&mut self, item: T) -> Result<(), Error>
    where
        S: BuildHasher,
    {
        self.insert_counting(u64::MAX, item).map(|_| ())
    }

    /// Inserts `item` if not already present (probabilistically).
    ///
    /// # Returns
    ///
    /// - `Ok(true)` if inserted.
    /// - `Ok(false)` if already present (may be a false positive).
    /// - `Err(Error::CapacityExceeded)` if the filter is full.
    #[inline]
    pub fn insert<T: Hash>(&mut self, item: T) -> Result<bool, Error>
    where
        S: BuildHasher,
    {
        self.insert_counting(1, item).map(|count| count == 0)
    }

    /// Inserts `item` up to `max_count` times.
    ///
    /// # Returns
    ///
    /// - `Ok(prev_count)` where `prev_count` is how many times the item was already present.
    ///   A new copy is inserted only if `prev_count < max_count`.
    /// - `Err(Error::CapacityExceeded)` if the filter is full.
    pub fn insert_counting<T: Hash>(&mut self, max_count: u64, item: T) -> Result<u64, Error>
    where
        S: BuildHasher,
    {
        let hash = self.hash_item(item);
        match self.insert_impl(max_count, hash) {
            Ok(count) => Ok(count),
            Err(_) => {
                self.grow_if_possible()?;
                self.insert_impl(max_count, hash)
            }
        }
    }

    /// Inserts the fingerprint specified by `hash` in the filter.
    ///
    /// Use this instead of [`Self::insert()`] when you have pre-computed fingerprints,
    /// are migrating data between filters, or need deterministic behavior with specific
    /// hash values. Use [`compute_fingerprint_with_hasher()`] to compute compatible fingerprints.
    ///
    /// # Parameters
    ///
    /// - `duplicate`: If `true`, insert even if fingerprint already exists.
    /// - `hash`: The fingerprint value. Only the lower [`Self::fingerprint_size()`] bits are used.
    ///
    /// # Returns
    ///
    /// - `Ok(true)` if inserted successfully.
    /// - `Ok(false)` if already present and `duplicate` is `false`.
    /// - `Err(Error::CapacityExceeded)` if the filter is full.
    #[inline]
    pub fn insert_fingerprint(&mut self, duplicate: bool, hash: u64) -> Result<bool, Error> {
        let max_count = if duplicate { u64::MAX } else { 1 };
        self.insert_fingerprint_counting(max_count, hash)
            .map(|count| count < max_count)
    }

    /// Inserts the fingerprint up to `max_count` times.
    ///
    /// Only the lower [`Self::fingerprint_size()`] bits of `hash` are used.
    ///
    /// # Returns
    ///
    /// - `Ok(prev_count)` where `prev_count` is how many times the fingerprint was present.
    ///   A new copy is inserted only if `prev_count < max_count`.
    /// - `Err(Error::CapacityExceeded)` if the filter is full.
    pub fn insert_fingerprint_counting(&mut self, max_count: u64, hash: u64) -> Result<u64, Error> {
        match self.insert_impl(max_count, hash) {
            Ok(count) => Ok(count),
            Err(_) => {
                self.grow_if_possible()?;
                self.insert_impl(max_count, hash)
            }
        }
    }

    /// Inserts the fingerprint specified by `hash` in the filter.
    /// `max_count` specifies how many occurences of the fingerprint can be added to the filter.
    /// It's up to the caller to grow the filter if needed and retry the insert.
    ///
    /// Returns `Ok(count)` of how many equal fingerprints _were_ in the filter.
    /// Returns `Err(Error::CapacityExceeded)` if the filter cannot admit the new item.
    fn insert_impl(&mut self, max_count: u64, hash: u64) -> Result<u64, Error> {
        enum Operation {
            NewRun,
            BeforeRunend,
            NewRunend,
        }

        let (hash_bucket_idx, hash_remainder) = self.calc_qr(hash);
        if self.is_slot_empty(hash_bucket_idx) {
            if self.len >= self.capacity() {
                return Err(Error::CapacityExceeded);
            }
            debug_assert!(!self.is_occupied(hash_bucket_idx));
            debug_assert!(!self.is_runend(hash_bucket_idx));
            self.set_occupied(hash_bucket_idx, true);
            self.set_runend(hash_bucket_idx, true);
            self.set_remainder(hash_bucket_idx, hash_remainder);
            self.len += 1;
            return Ok(0);
        }

        let mut runstart_idx = self.run_start(hash_bucket_idx);
        let mut runend_idx = self.run_end(hash_bucket_idx);
        let mut fingerprint_count = 0;
        let insert_idx;
        let operation;
        if self.is_occupied(hash_bucket_idx) {
            // adjust runend so its >= runstart even if it wrapped around
            if runend_idx < runstart_idx {
                runend_idx += self.total_buckets().get();
            }
            while runstart_idx <= runend_idx {
                match self.get_remainder(runstart_idx).cmp(&hash_remainder) {
                    Ordering::Equal => {
                        fingerprint_count += 1;
                        if fingerprint_count >= max_count {
                            return Ok(fingerprint_count);
                        }
                    }
                    Ordering::Greater => break,
                    Ordering::Less => (),
                }

                runstart_idx += 1;
            }

            insert_idx = runstart_idx % self.total_buckets();
            if runstart_idx > runend_idx {
                /* new remainder is >= than any remainder in the run. */
                operation = Operation::NewRunend;
            } else {
                /* there are larger remainders already in the run. */
                operation = Operation::BeforeRunend; /* Inserting */
            }
        } else {
            insert_idx = (runend_idx + 1) % self.total_buckets();
            operation = Operation::NewRun; /* Insert into empty bucket */
        }

        if self.len >= self.capacity() {
            return Err(Error::CapacityExceeded);
        }
        let empty_slot_idx = self.find_first_empty_slot(runend_idx + 1);
        if insert_idx != empty_slot_idx {
            self.shift_remainders_by_1(insert_idx, empty_slot_idx);
            self.shift_runends_by_1(insert_idx, empty_slot_idx);
        }
        self.set_remainder(insert_idx, hash_remainder);
        match operation {
            Operation::NewRun => {
                /* Insert into empty bucket */
                self.set_runend(insert_idx, true);
                self.set_occupied(hash_bucket_idx, true);
            }
            Operation::NewRunend => {
                /*  new remainder it is >= than any remainder in the run. */
                self.set_runend(insert_idx.wrapping_sub(1) % self.total_buckets(), false);
                self.set_runend(insert_idx, true);
            }
            Operation::BeforeRunend => { /* there are larger remainders already in the run. */ }
        }

        self.inc_offsets(hash_bucket_idx, empty_slot_idx);
        self.len += 1;
        Ok(fingerprint_count)
    }

    /// Shrinks memory usage if occupancy is low enough.
    ///
    /// Preserves the fingerprint size and false positive guarantees.
    /// Shrinking occurs when occupancy drops below 50% of capacity.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut filter = qfilter::Filter::new(1000, 0.01).unwrap();
    /// let initial_memory = filter.memory_usage();
    ///
    /// // Fill filter to capacity
    /// for i in 0..filter.capacity() {
    ///     filter.insert(i).unwrap();
    /// }
    ///
    /// // Remove most items to reduce occupancy below 50%
    /// for i in 0..filter.capacity() * 3 / 4 {
    ///     filter.remove(i);
    /// }
    ///
    /// filter.shrink_to_fit();
    /// assert!(filter.memory_usage() < initial_memory);
    /// ```
    pub fn shrink_to_fit(&mut self) {
        if self.total_blocks().get() > 1 && self.len() <= self.capacity() / 2 {
            let mut inserter = Builder::with_qr_and_hasher(
                (self.qbits.get() - 1).try_into().unwrap(),
                (self.rbits.get() + 1).try_into().unwrap(),
                self.max_qbits,
                self.build_hasher.clone(),
            )
            .unwrap();
            // Use insert_impl directly: the shrunk filter has capacity >= len (50% occupancy).
            for hash in self.fingerprints() {
                inserter
                    .insert_impl(true, hash)
                    .expect("Shrinking should not fail");
            }
            let new = inserter.into_filter();
            debug_assert_eq!(new.len, self.len);
            debug_assert_eq!(new.fingerprint_size(), self.fingerprint_size());
            *self = new;
        }
    }

    /// Merges all fingerprints from `other` into `self`.
    ///
    /// Both filters must have been built with the same hasher (and seed) for the
    /// result to be meaningful when using `T: Hash` methods.
    ///
    /// # Parameters
    ///
    /// - `keep_duplicates`: If `true`, the result is a multiset sum — each fingerprint
    ///   appears `count_self + count_other` times. If `false`, the result is a set union —
    ///   each unique fingerprint appears exactly once, discarding any duplicates that
    ///   existed in either filter.
    /// - `other`: Source filter. Must have `fingerprint_size() >= self.fingerprint_size()`.
    ///
    /// # Errors
    ///
    /// - [`Error::IncompatibleFingerprintSize`] if `other` has smaller fingerprints.
    /// - [`Error::CapacityExceeded`] if the merged result exceeds capacity.
    ///   When fingerprint sizes match, the merge is atomic (on error, `self` is unchanged).
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut filter1 = qfilter::Filter::new(100, 0.01).unwrap();
    /// let mut filter2 = qfilter::Filter::new(100, 0.01).unwrap();
    ///
    /// // Insert different items into each filter
    /// for i in 0..10 {
    ///     filter1.insert(i).unwrap();
    /// }
    /// for i in 10..20 {
    ///     filter2.insert(i).unwrap();
    /// }
    ///
    /// // Merge filter2 into filter1
    /// filter1.merge(false, &filter2).unwrap();
    ///
    /// // filter1 now contains items from both filters
    /// for i in 0..20 {
    ///     assert!(filter1.contains(i));
    /// }
    /// ```
    pub fn merge<B2: AsRef<[u8]>, S2>(
        &mut self,
        keep_duplicates: bool,
        other: &Filter<B2, S2>,
    ) -> Result<(), Error> {
        if self.fingerprint_size() == other.fingerprint_size() {
            *self = self.merge_sorted(keep_duplicates, other)?;
        } else if other.fingerprint_size() >= self.fingerprint_size() {
            // Different fingerprint sizes: truncation changes sort order,
            // so fall back to one-by-one insertion.
            let max_count = if keep_duplicates { u64::MAX } else { 1 };
            for hash in other.fingerprints() {
                self.insert_impl(max_count, hash)?;
            }
        } else {
            return Err(Error::IncompatibleFingerprintSize);
        }
        Ok(())
    }

    /// Two-iterator sorted merge into a fresh filter. O(n + m).
    ///
    /// Both iterators yield fingerprints in the same sorted order because
    /// the filters have identical fingerprint sizes (same qbits/rbits split).
    fn merge_sorted<B2: AsRef<[u8]>, S2>(
        &self,
        keep_duplicates: bool,
        other: &Filter<B2, S2>,
    ) -> Result<Self, Error> {
        debug_assert_eq!(self.fingerprint_size(), other.fingerprint_size());
        // Preserve growth headroom from self.
        // Pre-size to avoid intermediate growths, capped by self's max_qbits.
        let needed = if keep_duplicates {
            self.len.saturating_add(other.len)
        } else {
            // This may underestimate if there are no duplicates,
            // but it's a safe upper bound and avoids overallocation.
            self.len.max(other.len)
        };
        let needed_qbits = calculate_needed_slots(needed)
            .map_err(|_| Error::CapacityExceeded)?
            .trailing_zeros() as u8;
        let max_qbits_allowed = self.max_qbits.map_or(self.qbits.get(), |m| m.get());
        if needed_qbits > max_qbits_allowed {
            return Err(Error::CapacityExceeded);
        }
        let qbits = self.qbits.get().max(needed_qbits);
        let rbits = self.fingerprint_size() - qbits;
        let mut builder = Builder::with_qr_and_hasher(
            NonZeroU8::new(qbits).unwrap(),
            NonZeroU8::new(rbits).unwrap(),
            self.max_qbits,
            self.build_hasher.clone(),
        )?;
        let mut a = self.fingerprints().peekable();
        let mut b = other.fingerprints().peekable();
        loop {
            let hash = match (a.peek(), b.peek()) {
                (Some(&a_val), Some(&b_val)) if a_val <= b_val => {
                    a.next();
                    a_val
                }
                (Some(_), Some(_)) => b.next().unwrap(),
                (Some(_), None) => a.next().unwrap(),
                (None, Some(_)) => b.next().unwrap(),
                (None, None) => break,
            };
            builder.insert_fingerprint(keep_duplicates, hash)?;
        }
        Ok(builder.into_filter())
    }

    #[inline]
    fn grow_if_possible(&mut self) -> Result<(), Error> {
        *self = self.rebuild_grown()?.into_filter();
        Ok(())
    }

    /// Rebuilds this filter into a grown [`Builder`] with `qbits + 1`.
    ///
    /// Returns a builder (not a filter) so that callers can either continue
    /// the fast sequential-append path or consume it via [`Builder::into_filter()`].
    #[cold]
    fn rebuild_grown(&self) -> Result<Builder<S>, Error> {
        let max = self.max_qbits.ok_or(Error::CapacityExceeded)?;
        if max <= self.qbits {
            return Err(Error::CapacityExceeded);
        }
        let qbits = self.qbits.checked_add(1).ok_or(Error::CapacityExceeded)?;
        let rbits = NonZeroU8::new(self.rbits.get() - 1).ok_or(Error::CapacityExceeded)?;
        let mut inserter =
            Builder::with_qr_and_hasher(qbits, rbits, self.max_qbits, self.build_hasher.clone())?;
        // Use insert_impl directly: the new filter has 2x capacity so growth cannot happen.
        for hash in self.fingerprints() {
            inserter
                .insert_impl(true, hash)
                .expect("Growth should not fail");
        }
        debug_assert_eq!(self.len, inserter.filter.len);
        Ok(inserter)
    }

    #[doc(hidden)]
    #[cfg(any(fuzzing, test))]
    pub fn printout(&self) {
        eprintln!(
            "=== q {} r {} len {} cap {} ===",
            self.qbits,
            self.rbits,
            self.len(),
            self.capacity()
        );
        for b in 0..self.total_blocks().get() {
            let block = self.raw_block(b);
            eprintln!(
                "block {} offset {:?}\noccup {:064b}\nrunen {:064b}",
                b, block.offset, block.occupieds, block.runends
            );
            eprintln!(
                "      3210987654321098765432109876543210987654321098765432109876543210 {}",
                b * 64
            );
            eprint!("rem   ");
            for i in (0..64).rev() {
                let r = self.get_remainder(b * 64 + i);
                eprint!("{}", r % 100 / 10);
            }
            eprint!("\nrem   ");
            for i in (0..64).rev() {
                let r = self.get_remainder(b * 64 + i);
                eprint!("{}", r % 10);
            }
            println!();
        }
        eprintln!("===");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_fingerprint() {
        // Same input produces same hash
        assert_eq!(
            compute_fingerprint("hello", 64),
            compute_fingerprint("hello", 64)
        );
        // Different inputs produce different hashes
        assert_ne!(
            compute_fingerprint("hello", 64),
            compute_fingerprint("world", 64)
        );
        // Truncation works correctly
        let full = compute_fingerprint("test", 64);
        assert_eq!(compute_fingerprint("test", 8), full & 0xFF);
        assert_eq!(compute_fingerprint("test", 16), full & 0xFFFF);
        assert_eq!(compute_fingerprint("test", 32), full & 0xFFFF_FFFF);
        // Edge cases
        assert_eq!(compute_fingerprint("test", 0), 0);
        assert_eq!(compute_fingerprint("test", 1), full & 1);
        assert_eq!(compute_fingerprint("test", 65), full); // saturates at 64
                                                           // Matches filter behavior - confirm via fingerprints() iterator
        let mut filter = Filter::new(100, 0.01).unwrap();
        filter.insert("hello").unwrap();
        let fp = compute_fingerprint("hello", filter.fingerprint_size());
        assert_eq!(filter.fingerprints().collect::<Vec<_>>(), vec![fp]);
    }

    #[test]
    fn run_end_simple() {
        let mut f = Filter::new(50, 0.01).unwrap();
        f.set_occupied(5, true);
        f.set_runend(5, true);
        assert_eq!(f.run_end(4), 4);
        assert_eq!(f.run_end(5), 5);
        assert_eq!(f.run_end(6), 6);

        f.set_occupied(6, true);
        f.set_runend(6, true);
        assert_eq!(f.run_end(4), 4);
        assert_eq!(f.run_end(5), 5);
        assert_eq!(f.run_end(6), 6);

        f.set_runend(6, false);
        f.set_runend(7, true);
        assert_eq!(f.run_end(4), 4);
        assert_eq!(f.run_end(5), 5);
        assert_eq!(f.run_end(6), 7);

        f.set_runend(7, false);
        f.set_runend(8, true);
        assert_eq!(f.run_end(4), 4);
        assert_eq!(f.run_end(5), 5);
        assert_eq!(f.run_end(6), 8);

        f.set_occupied(10, true);
        f.set_runend(12, true);
        f.set_occupied(12, true);
        f.set_runend(13, true);
        assert_eq!(f.run_end(10), 12);
        assert_eq!(f.run_end(12), 13);

        f.set_occupied(11, true);
        f.set_runend(14, true);
        assert_eq!(f.run_end(10), 12);
        assert_eq!(f.run_end(11), 13);
        assert_eq!(f.run_end(12), 14);
    }

    #[test]
    fn run_end_eob() {
        let mut f = Filter::new(50, 0.01).unwrap();
        assert_eq!(f.total_buckets().get(), 64);
        f.set_occupied(63, true);
        f.set_runend(63, true);
        assert_eq!(f.run_end(62), 62);
        assert_eq!(f.run_end(63), 63);
        assert_eq!(f.find_first_empty_slot(62), 62);
        assert_eq!(f.find_first_empty_slot(63), 0);
    }

    #[test]
    fn run_end_crossing() {
        let mut f = Filter::new(50, 0.01).unwrap();
        f.set_occupied(0, true);
        f.set_runend(0, true);
        f.set_occupied(63, true);
        f.set_runend(63, true);
        assert_eq!(f.run_end(0), 0);
        assert_eq!(f.run_end(1), 1);
        assert_eq!(f.run_end(62), 62);
        assert_eq!(f.run_end(63), 63);

        f.set_runend(63, false);
        f.set_runend(1, true);
        f.adjust_block_offset(1, true);
        assert_eq!(f.run_end(0), 1);
        assert_eq!(f.run_end(1), 1);
        assert_eq!(f.run_end(62), 62);
        assert_eq!(f.run_end(63), 0);

        f.set_runend(1, false);
        f.set_runend(2, true);
        assert_eq!(f.run_end(63), 0);
        assert_eq!(f.run_end(0), 2);
        assert_eq!(f.run_end(1), 2);

        f.set_runend(2, false);
        f.set_runend(3, true);
        assert_eq!(f.run_end(63), 0);
        assert_eq!(f.run_end(1), 3);
        assert_eq!(f.run_end(2), 3);

        f.set_occupied(65, true);
        f.set_runend(68, true);
        assert_eq!(f.run_end(63), 0);
        assert_eq!(f.run_end(0), 3);
        assert_eq!(f.run_end(1), 4);
    }

    #[test]
    fn test_insert_duplicated() {
        for cap in [100, 200, 500, 1000] {
            let mut f = Filter::new(cap, 0.01).unwrap();
            for i in 0..f.capacity() / 2 {
                f.insert_duplicated(-1).unwrap();
                f.insert_duplicated(i).unwrap();
                assert!(f.count(-1) >= i);
                assert!(f.count(i) >= 1);
            }
        }
    }

    #[test]
    fn test_insert_duplicated_two() {
        for s in 0..10 {
            for c in [200, 800, 1500] {
                let mut f = Filter::new(c, 0.001).unwrap();
                for i in 0..f.capacity() / 2 {
                    f.insert_duplicated(-1).unwrap();
                    assert_eq!(f.count(-1), i + 1);
                    assert_eq!(f.count(s), i);
                    f.insert_duplicated(s).unwrap();
                    assert_eq!(f.count(-1), i + 1);
                    assert_eq!(f.count(s), i + 1);
                }
            }
        }
    }

    #[test]
    fn test_insert_duplicated_one() {
        for s in 0..10 {
            for cap in [100, 200, 500, 1000] {
                let mut f = Filter::new(cap, 0.01).unwrap();
                for i in 0..f.capacity() {
                    f.insert_duplicated(s).unwrap();
                    assert!(f.count(s) > i);
                }
                assert_eq!(f.count(s), f.capacity());
            }
        }
    }

    #[test]
    fn test_auto_resize_two() {
        let mut f = Filter::new_resizeable(50, 1000, 0.01).unwrap();
        for _ in 0..50 {
            f.insert_duplicated(0).unwrap();
        }
        for _ in 0..3 {
            f.insert_duplicated(1).unwrap();
        }
        f.grow_if_possible().unwrap();
        f.grow_if_possible().unwrap();
        f.grow_if_possible().unwrap();
        assert_eq!(f.count(0), 50);
        assert_eq!(f.count(1), 3);
    }

    #[test]
    fn test_new_resizeable() {
        let mut f = Filter::new_resizeable(100, 100, 0.01).unwrap();
        assert!(f.grow_if_possible().is_err());
        let mut f = Filter::new_resizeable(0, 100, 0.01).unwrap();
        assert!(f.grow_if_possible().is_ok());
    }

    #[test]
    fn test_new_resizeable_fp_rate_ceiling() {
        let f = Filter::new_resizeable(0, 1000, 0.05).unwrap();
        let ratio = f.max_error_ratio_resizeable();
        assert!(
            ratio <= 0.05,
            "max error ratio {} exceeds requested fp rate",
            ratio
        );
    }

    #[test]
    #[should_panic]
    fn test_new_capacity_overflow() {
        Filter::new_resizeable(100, u64::MAX, 0.01).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_new_hash_overflow() {
        Filter::new_resizeable(100, u64::MAX / 20, 0.01).unwrap();
    }

    #[test]
    fn test_auto_resize_one() {
        let mut f = Filter::new_resizeable(100, 500, 0.01).unwrap();
        for i in 0u64.. {
            if f.insert_duplicated(i).is_err() {
                assert_eq!(f.len(), i);
                break;
            }
        }
        assert!(f.len() >= 500);
        for i in 0u64..f.len() {
            assert!(f.contains(i), "{}", i);
        }
    }

    #[test]
    fn test_remainders_and_shifts() {
        let mut f = Filter::new(200, 0.01).unwrap();
        let c = f.capacity();
        for j in 0..c {
            f.set_remainder(j, 0b1011101);
            assert_eq!(f.get_remainder(j), 0b1011101);
            f.set_runend(j, true);
            assert!(f.is_runend(j));
        }
        for j in 0..c {
            f.set_remainder(j, 0b1111111);
            assert_eq!(f.get_remainder(j), 0b1111111);
            f.set_runend(j, false);
            assert!(!f.is_runend(j));
        }
        for j in 0..c {
            f.set_remainder(j, 0b1101101);
            assert_eq!(f.get_remainder(j), 0b1101101);
            f.set_runend(j, true);
            assert!(f.is_runend(j));
        }
        f.shift_remainders_by_1(0, c);
        f.shift_runends_by_1(0, c);

        for j in 1..=c {
            assert_eq!(f.get_remainder(j), 0b1101101);
        }
        assert!(!f.is_runend(0));
        for j in 1..=c {
            assert_eq!(f.get_remainder(j), 0b1101101);
            assert!(f.is_runend(j));
        }
    }

    #[test]
    fn test_remove() {
        for fp in [0.0001, 0.00001, 0.000001] {
            for cap in [0, 100, 200, 400, 1000] {
                let mut f = Filter::new(cap, fp).unwrap();
                dbg!(f.rbits, f.capacity());
                let c = f.capacity();
                for i in 0..c {
                    assert!(f.insert(i).unwrap());
                }
                assert_eq!(f.len(), c);
                for i in 0..c {
                    for j in 0..c {
                        assert_eq!(f.count(j), (j >= i) as u64, "{}", j);
                    }
                    // f.printout();
                    assert!(f.remove(i));
                    // f.printout();
                }
                assert!(f.is_empty());
            }
        }
    }
    #[test]
    fn test_remove_dup_one() {
        for s in 0..10 {
            for cap in [0, 100, 200, 500, 1000] {
                let mut f = Filter::new(cap, 0.0001).unwrap();
                let c = f.capacity();
                for _ in 0..c {
                    f.insert_duplicated(s).unwrap();
                }
                assert_eq!(f.len(), c);
                for i in 0..c {
                    assert_eq!(f.count(s), c - i);
                    assert!(f.remove(s));
                }
                assert!(f.is_empty());
            }
        }
    }
    #[test]
    fn test_remove_dup_two() {
        for s in 0..10 {
            dbg!(s);
            for cap in [100, 200, 500, 1000] {
                let mut f = Filter::new(cap, 0.0001).unwrap();
                let c = f.capacity();
                for _ in 0..c / 2 {
                    f.insert_duplicated(-1).unwrap();
                    f.insert_duplicated(s).unwrap();
                }
                assert_eq!(f.count(-1), c / 2);
                assert_eq!(f.count(s), c / 2);
                for i in 0..c / 2 {
                    assert_eq!(f.count(-1), c / 2 - i);
                    assert_eq!(f.count(s), c / 2 - i);
                    assert!(f.remove(-1));
                    assert_eq!(f.count(-1), c / 2 - i - 1);
                    assert_eq!(f.count(s), c / 2 - i);
                    assert!(f.remove(s));
                    assert_eq!(f.count(-1), c / 2 - i - 1);
                    assert_eq!(f.count(s), c / 2 - i - 1);
                }
                assert!(f.is_empty());
            }
        }
    }

    #[test]
    fn test_it_works() {
        for fp_rate_arg in [0.01, 0.001, 0.0001] {
            let mut f = Filter::new(100_000, fp_rate_arg).unwrap();
            assert!(!f.contains(0));
            assert_eq!(f.len(), 0);
            for i in 0..f.capacity() {
                f.insert_duplicated(i).unwrap();
            }
            for i in 0..f.capacity() {
                assert!(f.contains(i));
            }
            let fp_test_start = f.capacity();
            let est_fp_rate = (fp_test_start..)
                .take(50_000)
                .filter(|i| f.contains(i))
                .count() as f64
                / 50_000.0;
            dbg!(f.max_error_ratio(), est_fp_rate);
            assert!(est_fp_rate <= f.max_error_ratio());
        }
    }

    #[test]
    fn test_with_fingerprint_size_resizes() {
        let mut f = Filter::with_fingerprint_size(0, 8).unwrap();
        assert_eq!(f.fingerprint_size(), 8);
        assert_eq!(f.max_capacity(), (128u64 * 19).div_ceil(20));
        assert_eq!(f.capacity(), (64u64 * 19).div_ceil(20));
        for i in 0..f.max_capacity() {
            f.insert_fingerprint(false, i).unwrap();
        }
        assert_eq!(f.len(), f.max_capacity());
        assert!(f.insert_fingerprint(false, f.max_capacity()).is_err());
    }

    #[test]
    fn test_with_fingerprint_size() {
        let fingerprints = [
            0u64,
            0,
            1,
            1,
            1,
            1,
            1,
            0x777777777777,
            u32::MAX as u64 - 1,
            u32::MAX as u64 - 1,
            u32::MAX as u64,
            u64::MAX - 1,
            u64::MAX - 1,
            u64::MAX,
            u64::MAX,
        ];
        for fip_size in [7, 16, 24, 31, 49, 64] {
            let mut filter = Filter::with_fingerprint_size(1, fip_size).unwrap();
            for h in fingerprints {
                filter.insert_fingerprint(true, h).unwrap();
            }
            let out: Vec<u64> = filter.fingerprints().collect::<Vec<_>>();
            let mut expect = fingerprints.map(|h| h << (64 - fip_size) >> (64 - fip_size));
            expect.sort_unstable();
            assert_eq!(out, expect);
        }
    }

    #[test]
    fn test_merge() {
        fn test(mut f1: Filter, mut f2: Filter, mut f3: Filter) {
            assert!(f1.merge(true, &f1.clone()).is_ok());
            assert!(f1.merge(true, &f2).is_ok());
            assert!(f1.merge(true, &f3).is_ok());
            assert!(f2.merge(true, &f1).is_err());
            assert!(f2.merge(true, &f2.clone()).is_ok());
            assert!(f2.merge(true, &f3).is_ok());
            assert!(f3.merge(true, &f1).is_err());
            assert!(f3.merge(true, &f2).is_err());
            assert!(f3.merge(true, &f3.clone()).is_ok());

            f1.insert_fingerprint(true, 1).unwrap();
            f2.insert_fingerprint(true, 1).unwrap();
            f2.insert_fingerprint(true, 2).unwrap();
            f3.insert_fingerprint(true, 1).unwrap();
            f3.insert_fingerprint(true, 2).unwrap();
            f3.insert_fingerprint(true, 3).unwrap();
            assert_eq!(f1.len(), 1);
            assert_eq!(f2.len(), 2);
            assert_eq!(f3.len(), 3);

            f1.merge(false, &f1.clone()).unwrap();
            assert_eq!(f1.len(), 1);
            f1.merge(true, &f2.clone()).unwrap();
            assert_eq!(f1.len(), 3);
            f1.merge(false, &f3.clone()).unwrap();
            assert_eq!(f1.len(), 4);

            for _ in f1.len()..f1.capacity() {
                f1.insert_fingerprint(true, 1).unwrap();
            }
            assert_eq!(f1.len(), f1.capacity());
            assert!(matches!(
                f1.insert_impl(u64::MAX, 1),
                Err(Error::CapacityExceeded)
            ));
            if f1.max_capacity() > f1.capacity() {
                // Resizable: merge grows to accommodate.
                let len_before = f1.len();
                f1.merge(true, &f1.clone()).unwrap();
                assert_eq!(f1.len(), len_before * 2);
            } else {
                // Non-resizable: merge fails when capacity exceeded.
                assert!(matches!(
                    f1.merge(true, &f1.clone()),
                    Err(Error::CapacityExceeded)
                ));
            }
            assert!(matches!(f1.insert_fingerprint(false, 1), Ok(false)));
            assert!(matches!(f1.merge(false, &f1.clone()), Ok(())));
        }
        test(
            Filter::with_fingerprint_size(1, 10).unwrap(),
            Filter::with_fingerprint_size(1, 11).unwrap(),
            Filter::with_fingerprint_size(1, 12).unwrap(),
        );
        test(
            Filter::new(1, 0.01).unwrap(),
            Filter::new(1, 0.001).unwrap(),
            Filter::new(1, 0.0001).unwrap(),
        );
    }

    #[test]
    fn test_merge_correctness() {
        let max_cap = 10000u64;
        for (cap, fp_rate) in [(100, 0.01), (500, 0.001), (1000, 0.01)] {
            let new = |c: u64| Filter::new_resizeable(c, max_cap, fp_rate).unwrap();
            let n = cap as u64;

            // Build two filters with 50% overlap:
            // f1 has [0..n), f2 has [n/2..n*3/2)
            let mut f1 = new(n);
            let mut f2 = new(n);
            for i in 0..n {
                f1.insert_duplicated(i).unwrap();
            }
            for i in n / 2..n + n / 2 {
                f2.insert_duplicated(i).unwrap();
            }

            // Dedup merge: result should contain all unique items
            let mut merged_dedup = f1.clone();
            merged_dedup.merge(false, &f2).unwrap();
            for i in 0..n + n / 2 {
                assert!(merged_dedup.contains(i), "missing {i} after dedup merge");
            }

            // Self-merge with dedup shouldn't change len
            let mut self_merge = f1.clone();
            self_merge.merge(false, &f1).unwrap();
            assert_eq!(self_merge.len(), f1.len());

            // Duplicate merge: len should be sum of both
            let mut merged_dup = f1.clone();
            merged_dup.merge(true, &f2).unwrap();
            assert_eq!(merged_dup.len(), f1.len() + f2.len());
            for i in 0..n + n / 2 {
                assert!(merged_dup.contains(i), "missing {i} after dup merge");
            }

            // Merge two disjoint full filters (exercises pre-sizing growth)
            let mut r1 = new(n);
            let mut r2 = new(n);
            for i in 0..n {
                r1.insert_duplicated(i).unwrap();
            }
            for i in n..n * 2 {
                r2.insert_duplicated(i).unwrap();
            }
            let r1_len = r1.len();
            let r2_len = r2.len();
            r1.merge(true, &r2).unwrap();
            assert_eq!(r1.len(), r1_len + r2_len);

            // Fingerprints must remain sorted after every merge
            for f in [&merged_dedup, &self_merge, &merged_dup, &r1] {
                let fps: Vec<u64> = f.fingerprints().collect();
                for w in fps.windows(2) {
                    assert!(w[0] <= w[1], "unsorted fingerprints: {} > {}", w[0], w[1]);
                }
            }
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde() {
        for capacity in [100, 1000, 10000] {
            for fp_ratio in [0.2, 0.1, 0.01, 0.001, 0.0001] {
                let mut f = Filter::new(capacity, fp_ratio).unwrap();
                for i in 0..f.capacity() {
                    f.insert(i).unwrap();
                }

                let ser = serde_cbor::to_vec(&f).unwrap();
                f = serde_cbor::from_slice(&ser).unwrap();
                for i in 0..f.capacity() {
                    f.contains(i);
                }
                dbg!(
                    f.current_error_ratio(),
                    f.max_error_ratio(),
                    f.capacity(),
                    f.len(),
                    ser.len()
                );
            }
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_filter_ref_zero_copy() {
        for capacity in [100, 1000, 10000] {
            for fp_ratio in [0.2, 0.1, 0.01, 0.001] {
                let mut f = Filter::new(capacity, fp_ratio).unwrap();
                for i in 0..f.capacity() {
                    f.insert(i).unwrap();
                }

                let ser = serde_cbor::to_vec(&f).unwrap();
                let fr: FilterRef<'_> = serde_cbor::from_slice(&ser).unwrap();

                // All read-only accessors match
                assert_eq!(fr.len(), f.len());
                assert_eq!(fr.is_empty(), f.is_empty());
                assert_eq!(fr.fingerprint_size(), f.fingerprint_size());
                assert_eq!(fr.memory_usage(), f.memory_usage());
                assert_eq!(fr.capacity(), f.capacity());
                assert_eq!(fr.max_capacity(), f.max_capacity());
                assert_eq!(fr.max_error_ratio(), f.max_error_ratio());
                assert_eq!(fr.current_error_ratio(), f.current_error_ratio());

                // contains and count match
                for i in 0..f.capacity() {
                    assert_eq!(fr.contains(i), f.contains(i));
                    assert_eq!(fr.count(i), f.count(i));
                }

                // fingerprint-based queries match
                let fp = compute_fingerprint(42u64, f.fingerprint_size());
                assert_eq!(fr.contains_fingerprint(fp), f.contains_fingerprint(fp));
                assert_eq!(fr.count_fingerprint(fp), f.count_fingerprint(fp));

                // fingerprints iterator matches
                let owned_fps: Vec<u64> = f.fingerprints().collect();
                let ref_fps: Vec<u64> = fr.fingerprints().collect();
                assert_eq!(owned_fps, ref_fps);

                // IntoIterator matches
                let ref_iter_fps: Vec<u64> = fr.into_iter().collect();
                assert_eq!(owned_fps, ref_iter_fps);

                // to_owned roundtrip
                let f2 = fr.to_owned();
                assert_eq!(f2.len(), f.len());
                assert_eq!(f2.fingerprint_size(), f.fingerprint_size());
                for i in 0..f.capacity() {
                    assert_eq!(f2.contains(i), f.contains(i));
                }
            }
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_filter_ref_as_ref() {
        let mut f = Filter::new(100, 0.01).unwrap();
        for i in 0..50u64 {
            f.insert(i).unwrap();
        }

        let fr = f.as_filter_ref();
        assert_eq!(fr.len(), f.len());
        assert_eq!(fr.fingerprint_size(), f.fingerprint_size());
        for i in 0..50u64 {
            assert!(fr.contains(i));
        }

        let fps: Vec<u64> = f.fingerprints().collect();
        let ref_fps: Vec<u64> = fr.fingerprints().collect();
        assert_eq!(fps, ref_fps);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_rejects_invalid() {
        // Valid filter to use as a base for tampering
        let f = Filter::new(100, 0.01).unwrap();
        let ser = serde_cbor::to_vec(&f).unwrap();

        // Valid deserialization works
        let _: Filter = serde_cbor::from_slice(&ser).unwrap();

        // Tamper with qbits to an invalid value (too small)
        let mut tampered: serde_cbor::Value = serde_cbor::from_slice(&ser).unwrap();
        if let serde_cbor::Value::Map(ref mut map) = tampered {
            map.insert(
                serde_cbor::Value::Text("q".into()),
                serde_cbor::Value::Integer(1), // qbits=1, way too small
            );
        }
        let tampered_bytes = serde_cbor::to_vec(&tampered).unwrap();
        assert!(serde_cbor::from_slice::<Filter>(&tampered_bytes).is_err());

        // Tamper with buffer length (truncate)
        let mut tampered: serde_cbor::Value = serde_cbor::from_slice(&ser).unwrap();
        if let serde_cbor::Value::Map(ref mut map) = tampered {
            map.insert(
                serde_cbor::Value::Text("b".into()),
                serde_cbor::Value::Bytes(vec![0; 8]),
            );
        }
        let tampered_bytes = serde_cbor::to_vec(&tampered).unwrap();
        assert!(serde_cbor::from_slice::<Filter>(&tampered_bytes).is_err());

        // Tamper with len to exceed total buckets
        let mut tampered: serde_cbor::Value = serde_cbor::from_slice(&ser).unwrap();
        if let serde_cbor::Value::Map(ref mut map) = tampered {
            map.insert(
                serde_cbor::Value::Text("l".into()),
                serde_cbor::Value::Integer(i128::from(u64::MAX)),
            );
        }
        let tampered_bytes = serde_cbor::to_vec(&tampered).unwrap();
        assert!(serde_cbor::from_slice::<Filter>(&tampered_bytes).is_err());

        // Tamper with qbits to exceed MAX_QBITS
        let mut tampered: serde_cbor::Value = serde_cbor::from_slice(&ser).unwrap();
        if let serde_cbor::Value::Map(ref mut map) = tampered {
            map.insert(
                serde_cbor::Value::Text("q".into()),
                serde_cbor::Value::Integer(63),
            );
        }
        let tampered_bytes = serde_cbor::to_vec(&tampered).unwrap();
        assert!(serde_cbor::from_slice::<Filter>(&tampered_bytes).is_err());

        // Tamper with max_qbits to exceed qbits + rbits - 1
        let mut tampered: serde_cbor::Value = serde_cbor::from_slice(&ser).unwrap();
        if let serde_cbor::Value::Map(ref mut map) = tampered {
            map.insert(
                serde_cbor::Value::Text("m".into()),
                serde_cbor::Value::Integer(60),
            );
        }
        let tampered_bytes = serde_cbor::to_vec(&tampered).unwrap();
        assert!(serde_cbor::from_slice::<Filter>(&tampered_bytes).is_err());
    }

    #[test]
    fn test_dec_offset_edge_case() {
        // case found in fuzz testing, exercises offset decrement across blocks
        #[rustfmt::skip]
        let sample = [(0u16, 287), (2u16, 1), (9u16, 2), (10u16, 1), (53u16, 5), (61u16, 5), (127u16, 2), (232u16, 1), (255u16, 21), (314u16, 2), (317u16, 2), (384u16, 2), (511u16, 3), (512u16, 2), (1599u16, 2), (2303u16, 5), (2559u16, 2), (2568u16, 3), (2815u16, 2), (6400u16, 2), (9211u16, 2), (9728u16, 2), (10790u16, 1), (10794u16, 94), (10797u16, 2), (10999u16, 2), (11007u16, 2), (11520u16, 1), (12800u16, 4), (12842u16, 2), (13823u16, 1), (14984u16, 2), (15617u16, 2), (15871u16, 4), (16128u16, 3), (16383u16, 2), (16394u16, 1), (18167u16, 2), (23807u16, 1), (32759u16, 2)];
        let template = Filter::new(400, 0.1).unwrap();
        let fingerprint_bits = template.qbits.get() + 3;
        let mut f = Filter::with_fingerprint_size(400, fingerprint_bits).unwrap();
        for (i, c) in sample {
            for _ in 0..c {
                f.insert_duplicated(i).unwrap();
            }
        }
        f.validate_offsets(0, f.total_buckets().get());
        f.remove(0u16);
        f.validate_offsets(0, f.total_buckets().get());
    }

    #[test]
    fn test_capacity_edge_cases() {
        // Cap at 28 to avoid buffer allocation overflow on 32-bit platforms
        let max_n = if cfg!(target_pointer_width = "32") {
            28
        } else {
            32
        };
        for n in 1..max_n {
            let base = (1u64 << n) * 19 / 20;
            // Test numbers around the edge
            for i in [base - 1, base, base + 1] {
                let filter = Filter::new(i, 0.01).unwrap();
                assert!(
                    filter.capacity() >= i,
                    "Requested capacity {} but got {}",
                    i,
                    filter.capacity()
                );
                assert_eq!(filter.capacity(), filter.max_capacity());
            }
        }
    }

    #[test]
    fn test_max_capacity() {
        for i in 7..=64 {
            let f = Filter::with_fingerprint_size(0, i).unwrap();
            assert!(f.capacity() <= f.max_capacity());
            assert_eq!(
                f.max_capacity(),
                ((1u64 << (i - 1).min(MAX_QBITS)) * 19).div_ceil(20)
            );
        }
        for i in 1..MAX_QBITS {
            let f = Filter::new_resizeable(0, 2u64.pow(i as u32), 0.5).unwrap();
            assert_eq!(f.capacity(), 61);
            assert!(f.capacity() <= f.max_capacity());
        }
        // Test the maximum capacity
        let f = Filter::new_resizeable(0, MAX_CAPACITY, 0.5).unwrap();
        assert_eq!(f.capacity(), 61);
        assert_eq!(f.max_capacity(), MAX_CAPACITY);
        // Test the maximum capacity + 1, which should fail
        Filter::new_resizeable(0, MAX_CAPACITY + 1, 0.5).unwrap_err();
    }

    #[test]
    fn test_builder_matches_regular() {
        for fp_rate in [0.01, 0.001, 0.0001] {
            for cap in [100, 500, 1000, 5000] {
                let mut regular = Filter::new(cap, fp_rate).unwrap();
                let mut fingerprints: Vec<u64> = (0..regular.capacity())
                    .map(|i| compute_fingerprint(i, regular.fingerprint_size()))
                    .collect();
                fingerprints.sort_unstable();

                for &h in &fingerprints {
                    regular.insert_fingerprint(true, h).unwrap();
                }

                let mut inserter = Builder::new(Filter::new(cap, fp_rate).unwrap());
                for &h in &fingerprints {
                    inserter.insert_fingerprint(true, h).unwrap();
                }
                let sorted = inserter.into_filter();

                assert_eq!(regular.len(), sorted.len());
                let reg_fps: Vec<u64> = regular.fingerprints().collect();
                let sort_fps: Vec<u64> = sorted.fingerprints().collect();
                assert_eq!(reg_fps, sort_fps);
            }
        }
    }

    #[test]
    fn test_builder_no_duplicates() {
        let mut inserter = Builder::new(Filter::new(1000, 0.01).unwrap());
        let fp_size = inserter.fingerprint_size();
        let mut fingerprints: Vec<u64> = (0..500u64)
            .map(|i| compute_fingerprint(i, fp_size))
            .collect();
        fingerprints.sort_unstable();
        let mut inserted = 0;
        for &h in &fingerprints {
            if inserter.insert_fingerprint(false, h).unwrap() {
                inserted += 1;
            }
        }
        let f = inserter.into_filter();

        // With duplicate=false, count of inserts should equal the number of distinct fingerprints
        let distinct: std::collections::HashSet<u64> = fingerprints.iter().copied().collect();
        assert_eq!(inserted, distinct.len());
        assert_eq!(f.len(), distinct.len() as u64);
    }

    #[test]
    fn test_builder_with_duplicates() {
        let mut inserter = Builder::new(Filter::new(1000, 0.01).unwrap());
        let fp_size = inserter.fingerprint_size();

        // Insert the same fingerprint multiple times
        let hash = compute_fingerprint(42u64, fp_size);
        for _ in 0..10 {
            inserter.insert_fingerprint(true, hash).unwrap();
        }
        let f = inserter.into_filter();

        assert_eq!(f.len(), 10);
        assert_eq!(f.count_fingerprint(hash), 10);
    }

    #[test]
    fn test_builder_auto_growth() {
        let mut inserter = Builder::new(Filter::new_resizeable(100, 5000, 0.01).unwrap());
        let initial_cap = inserter.capacity();
        let fp_size = inserter.fingerprint_size();

        let mut fingerprints: Vec<u64> = (0..3000u64)
            .map(|i| compute_fingerprint(i, fp_size))
            .collect();
        fingerprints.sort_unstable();
        for &h in &fingerprints {
            inserter.insert_fingerprint(true, h).unwrap();
        }
        let f = inserter.into_filter();

        assert!(f.capacity() > initial_cap, "filter should have grown");
        assert_eq!(f.len(), 3000);

        // Verify all items are present
        for i in 0..3000u64 {
            assert!(f.contains(i), "missing item {}", i);
        }
    }

    #[test]
    fn test_builder_empty_filter() {
        let inserter = Builder::new(Filter::new(100, 0.01).unwrap());
        let f = inserter.into_filter();
        assert!(f.is_empty());
    }

    #[test]
    fn test_builder_various_fingerprint_sizes() {
        for fp_size in [14, 16, 20, 24, 32] {
            let mut inserter = Builder::new(Filter::with_fingerprint_size(500, fp_size).unwrap());
            let cap = inserter.capacity();
            let mut fingerprints: Vec<u64> =
                (0..cap).map(|i| compute_fingerprint(i, fp_size)).collect();
            fingerprints.sort_unstable();
            for &h in &fingerprints {
                inserter.insert_fingerprint(true, h).unwrap();
            }
            let f = inserter.into_filter();

            assert_eq!(f.len(), cap);

            // Verify via fingerprints() iteration
            let stored: Vec<u64> = f.fingerprints().collect();
            assert_eq!(stored.len(), cap as usize);
            // Should be sorted
            for w in stored.windows(2) {
                assert!(w[0] <= w[1]);
            }
        }
    }

    #[test]
    fn test_builder_single_block() {
        // Small filter that fits in a single block (64 slots)
        let mut inserter = Builder::new(Filter::new(50, 0.01).unwrap());
        let fp_size = inserter.fingerprint_size();
        let cap = inserter.capacity();
        let mut fingerprints: Vec<u64> =
            (0..cap).map(|i| compute_fingerprint(i, fp_size)).collect();
        fingerprints.sort_unstable();
        for &h in &fingerprints {
            inserter.insert_fingerprint(true, h).unwrap();
        }
        let f = inserter.into_filter();

        assert_eq!(f.total_buckets().get(), 64);
        assert_eq!(f.len(), cap);
        for i in 0..cap {
            assert!(f.contains(i));
        }
    }

    #[test]
    fn test_builder_multi_block_spillover() {
        // Use a filter with multiple blocks and a small fingerprint size to force spillover
        let mut inserter = Builder::new(Filter::with_fingerprint_size(500, 14).unwrap());
        let cap = inserter.capacity();
        let fp_size = inserter.fingerprint_size();
        let mask = (1u64 << fp_size) - 1;

        // Create fingerprints that will have many collisions in the same quotient
        // to force runs that spill across block boundaries
        let mut fingerprints: Vec<u64> = (0..cap).map(|i| i & mask).collect();
        fingerprints.sort_unstable();

        let mut regular = Filter::with_fingerprint_size(500, 14).unwrap();
        for &h in &fingerprints {
            regular.insert_fingerprint(true, h).unwrap();
        }
        for &h in &fingerprints {
            inserter.insert_fingerprint(true, h).unwrap();
        }
        let sorted = inserter.into_filter();

        assert_eq!(sorted.len(), regular.len());
        assert_eq!(
            sorted.fingerprints().collect::<Vec<_>>(),
            regular.fingerprints().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_builder_identical_state() {
        // Verifies that sorted insertion produces byte-identical filter state
        // compared to regular insertion, including at high occupancy where
        // the wrapping slow path is triggered.
        for (cap, fp_rate) in [(500, 0.001), (100, 0.01), (1000, 0.0001)] {
            let mut inserter = Builder::new(Filter::new(cap, fp_rate).unwrap());
            let fp_size = inserter.fingerprint_size();
            let capacity = inserter.capacity();

            let mut fingerprints: Vec<u64> = (0..capacity)
                .map(|i| compute_fingerprint(i, fp_size))
                .collect();
            fingerprints.sort_unstable();

            let mut regular = Filter::new(cap, fp_rate).unwrap();
            for &h in &fingerprints {
                regular.insert_fingerprint(true, h).unwrap();
            }
            for &h in &fingerprints {
                inserter.insert_fingerprint(true, h).unwrap();
            }
            let sorted = inserter.into_filter();

            assert_eq!(
                regular.buffer, sorted.buffer,
                "buffer mismatch for cap={cap}, fp_rate={fp_rate}"
            );
        }
    }

    #[test]
    fn test_builder_hash_collision_dedup() {
        // Regression test: two different hash values that truncate to the same
        // (quotient, remainder) must be treated as duplicates when duplicate=false.
        // See: fuzz_sorted_insert/minimized-from-c401dba0cd67a0a3c34c19ce181d23a0dbd37e8e
        let fp_size = 7u8;
        let cap = 5u64;

        let mut fingerprints: Vec<u64> = vec![0, 12032];
        fingerprints.sort_unstable();

        // Regular insertion
        let mut regular = Filter::with_fingerprint_size(cap, fp_size).unwrap();
        for &h in &fingerprints {
            let _ = regular.insert_fingerprint(false, h);
        }

        // Sorted insertion
        let mut inserter = Builder::new(Filter::with_fingerprint_size(cap, fp_size).unwrap());
        for &h in &fingerprints {
            let _ = inserter.insert_fingerprint(false, h);
        }
        let sorted_f = inserter.into_filter();

        assert_eq!(regular.len(), sorted_f.len());
        let reg_fps: Vec<u64> = regular.fingerprints().collect();
        let sort_fps: Vec<u64> = sorted_f.fingerprints().collect();
        assert_eq!(reg_fps, sort_fps);
    }

    #[test]
    fn test_custom_hasher() {
        use std::collections::hash_map::RandomState;

        let hasher = RandomState::new();
        let mut f = Filter::new_with_hasher(1000, 0.01, hasher).unwrap();
        for i in 0u64..100 {
            f.insert(i).unwrap();
        }
        for i in 0u64..100 {
            assert!(f.contains(i));
        }
        // Remove and verify
        assert!(f.remove(50u64));
        assert!(!f.contains(50u64));
    }

    #[test]
    fn test_random_build_hasher() {
        let hasher = foldhash_portable::quality::RandomState::default();
        let mut f = Filter::new_with_hasher(1000, 0.01, hasher).unwrap();
        for i in 0u64..100 {
            f.insert(i).unwrap();
        }
        for i in 0u64..100 {
            assert!(f.contains(i));
        }
    }

    #[test]
    fn test_stable_build_hasher_deterministic() {
        // Two filters with the default StableBuildHasher produce identical fingerprints
        let mut f1 = Filter::new(1000, 0.01).unwrap();
        let mut f2 = Filter::new(1000, 0.01).unwrap();
        for i in 0u64..100 {
            f1.insert(i).unwrap();
            f2.insert(i).unwrap();
        }
        let fps1: Vec<u64> = f1.fingerprints().collect();
        let fps2: Vec<u64> = f2.fingerprints().collect();
        assert_eq!(fps1, fps2);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_hash_stability() {
        // Build filters with known inputs and verify serialized bytes match
        // hardcoded values. This catches any change in the hash algorithm or
        // StableHasher normalization, and ensures cross-platform compatibility
        // (the same bytes must be produced on 32-bit BE and 64-bit LE).

        fn build_filter(items: &[usize], capacity: u64, fp_rate: f64) -> Vec<u8> {
            let mut f = Filter::new(capacity, fp_rate).unwrap();
            for &i in items {
                f.insert(i).unwrap();
            }
            serde_cbor::to_vec(&f).unwrap()
        }

        // Small filter with a few usize items (exercises write_usize → write_u64 normalization)
        let small = build_filter(&[1, 2, 3, 42, 100], 100, 0.01);
        // Larger filter with sequential usize items
        let seq = build_filter(&(0..50).collect::<Vec<_>>(), 100, 0.01);
        // Filter with string items
        let mut f = Filter::new(100, 0.01).unwrap();
        for s in ["hello", "world", "foo", "bar"] {
            f.insert(s).unwrap();
        }
        let strings = serde_cbor::to_vec(&f).unwrap();

        // Hardcoded expected bytes — if these change, the hash algorithm or
        // StableHasher normalization changed, breaking cross-platform compatibility.
        // usize items exercise write_usize → write_u64 normalization (32-bit vs 64-bit).
        assert_eq!(
            small,
            [
                165, 97, 98, 88, 154, 0, 0, 17, 0, 136, 0, 0, 0, 0, 0, 17, 0, 136, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 93, 0, 0, 160, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 224, 14, 0, 0,
                158, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                97, 108, 5, 97, 113, 7, 97, 114, 7, 97, 109, 246
            ]
        );
        assert_eq!(
            seq,
            [
                165, 97, 98, 88, 154, 0, 6, 249, 12, 228, 11, 2, 36, 134, 6, 121, 13, 228, 11, 2,
                68, 134, 0, 203, 30, 0, 0, 0, 0, 93, 0, 32, 162, 45, 167, 48, 91, 0, 189, 11, 0, 0,
                0, 0, 192, 5, 0, 184, 14, 159, 164, 22, 224, 2, 0, 0, 0, 128, 35, 0, 0, 0, 0, 0, 0,
                128, 0, 0, 184, 113, 1, 0, 11, 22, 0, 0, 0, 226, 0, 14, 74, 209, 193, 40, 72, 117,
                144, 14, 74, 145, 195, 40, 72, 117, 144, 128, 110, 44, 15, 0, 0, 0, 0, 40, 224, 7,
                0, 132, 1, 32, 0, 0, 0, 6, 220, 242, 126, 42, 0, 0, 0, 140, 1, 0, 0, 160, 11, 120,
                1, 0, 0, 0, 0, 2, 0, 4, 1, 59, 128, 21, 224, 137, 121, 0, 0, 0, 0, 80, 6, 0, 120,
                0, 0, 0, 0, 0, 0, 0, 0, 97, 108, 24, 50, 97, 113, 7, 97, 114, 7, 97, 109, 246
            ]
        );
        assert_eq!(
            strings,
            [
                165, 97, 98, 88, 154, 0, 32, 0, 0, 0, 0, 8, 0, 32, 32, 0, 0, 0, 0, 8, 0, 32, 0, 0,
                0, 0, 72, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 160, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 216, 0,
                0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                64, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                97, 108, 4, 97, 113, 7, 97, 114, 7, 97, 109, 246
            ]
        );
    }

    #[test]
    fn extreme_fp_rate_does_not_overflow() {
        // Regression for #25: a tiny `fp_rate` (0.0, f64::MIN_POSITIVE, or any
        // negative value) used to overflow the u8 fingerprint-bit arithmetic and
        // panic. Every constructor must instead return NotEnoughFingerprintBits.
        for &fp in &[0.0, f64::MIN_POSITIVE, -1.0] {
            assert!(matches!(
                Filter::new(1000, fp),
                Err(Error::NotEnoughFingerprintBits)
            ));
            assert!(matches!(
                Filter::new_resizeable(1000, 1_000_000, fp),
                Err(Error::NotEnoughFingerprintBits)
            ));
            assert!(matches!(
                filter_specs(1000, fp),
                Err(Error::NotEnoughFingerprintBits)
            ));
        }
        // Valid rates are unaffected, and the clamp bound is exactly 2^-64.
        assert!(Filter::new(1000, 0.01).is_ok());
        assert_eq!(MIN_FP_RATE, 2f64.powi(-64));
    }
}
