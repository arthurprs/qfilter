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
//! The hashing algorithm used is [xxhash3](https://crates.io/crates/xxhash-rust)
//! which offers both high performance and stability across platforms.
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
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

use std::{
    cmp::Ordering,
    hash::{Hash, Hasher},
    num::{NonZeroU64, NonZeroU8},
    ops::{RangeBounds, RangeFrom},
};

#[cfg(feature = "jsonschema")]
use schemars::JsonSchema;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use stable_hasher::StableHasher;

mod portable_select;
mod stable_hasher;

/// Computes a truncated fingerprint for the given item.
///
/// Uses the same stable hash algorithm (xxhash3) as [`Filter`].
/// Only the lower `fingerprint_bits` bits of the hash are returned.
///
/// - `fingerprint_bits == 0` returns `0`
/// - `fingerprint_bits >= 64` returns the full 64-bit hash
///
/// # Example
///
/// ```rust
/// use qfilter::{compute_fingerprint, Filter};
///
/// let mut filter = Filter::new(100, 0.01).unwrap();
/// filter.insert("hello").unwrap();
/// let fp = compute_fingerprint("hello", filter.fingerprint_size());
/// assert!(filter.contains_fingerprint(fp));
/// ```
#[inline]
pub fn compute_fingerprint<T: Hash>(item: T, fingerprint_bits: u8) -> u64 {
    let mut hasher = StableHasher::new();
    item.hash(&mut hasher);
    let hash = hasher.finish();
    if fingerprint_bits >= 64 {
        hash
    } else {
        hash & ((1u64 << fingerprint_bits) - 1)
    }
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
    /// pre-computing fingerprints with [`compute_fingerprint()`]. It represents
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
/// use qfilter::{filter_specs, compute_fingerprint, Filter};
///
/// let capacity = 10000;
/// let fp_rate = 0.01;
///
/// // Get filter specifications
/// let specs = filter_specs(capacity, fp_rate).unwrap();
///
/// // Pre-compute fingerprints using the fingerprint size
/// let fingerprints: Vec<u64> = (0..100)
///     .map(|i| compute_fingerprint(i, specs.fingerprint_bits))
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
    let fp_rate = fp_rate.clamp(f64::MIN_POSITIVE, 0.5);
    let rbits = (-fp_rate.log2()).ceil().max(1.0) as u8;
    let fingerprint_bits = qbits + rbits;
    if fingerprint_bits > 64 {
        return Err(Error::NotEnoughFingerprintBits);
    }

    let memory_bytes_max = calculate_memory_bytes(qbits, rbits);
    // Smallest resizable filter has 64 slots (qbits=6)
    let memory_bytes_min = calculate_memory_bytes(6, fingerprint_bits - 6);
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
fn calculate_memory_bytes(qbits: u8, rbits: u8) -> usize {
    let num_blocks = (1u64 << qbits) / 64;
    let block_bytes = 17 + 8 * rbits as u64;
    (num_blocks * block_bytes + 8) as usize
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
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "jsonschema", derive(JsonSchema))]
pub struct Filter {
    #[cfg_attr(
        feature = "serde",
        serde(
            rename = "b",
            serialize_with = "serde_bytes::serialize",
            deserialize_with = "serde_bytes::deserialize"
        )
    )]
    buffer: Box<[u8]>,
    #[cfg_attr(feature = "serde", serde(rename = "l"))]
    len: u64,
    #[cfg_attr(feature = "serde", serde(rename = "q"))]
    qbits: NonZeroU8,
    #[cfg_attr(feature = "serde", serde(rename = "r"))]
    rbits: NonZeroU8,
    #[cfg_attr(feature = "serde", serde(rename = "m"))]
    max_qbits: Option<NonZeroU8>,
}

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
    /// The requested capacity exceeds [`Filter::MAX_CAPACITY`].
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

/// An iterator over the fingerprints stored in a [`Filter`].
///
/// Fingerprints are yielded in ascending order. Each value is a `u64` where only
/// the lower [`Filter::fingerprint_size()`] bits are meaningful (upper bits are zero).
///
/// If duplicates were inserted, the same fingerprint value may be yielded multiple times.
///
/// Created by [`Filter::fingerprints()`].
pub struct FingerprintIter<'a> {
    filter: &'a Filter,
    q_bucket_idx: u64,
    r_bucket_idx: u64,
    remaining: u64,
    q_block_idx: u64,
    r_block_idx: u64,
    cached_occupieds: u64,
    cached_runends: u64,
}

impl<'a> FingerprintIter<'a> {
    fn new(filter: &'a Filter) -> Self {
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

impl Filter {
    /// Maximum log2 number of slots that can be used in the filter.
    /// Effectively, the largest power of 2 that can be multiplied by 19 without overflowing u64.
    const MAX_QBITS: u8 = 59;

    /// Maximum number of items that can be stored in the filter: ceil(2^59 * 19 / 20)
    pub const MAX_CAPACITY: u64 = (2u64.pow(Self::MAX_QBITS as u32) * 19).div_ceil(20);

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
        let fp_rate = fp_rate.clamp(f64::MIN_POSITIVE, 0.5);
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
    /// Use [`compute_fingerprint()`] to compute fingerprints with a specific bit size.
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
    ) -> Result<Filter, Error> {
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
            result.max_qbits = Some((qbits + rbits - 1).min(Self::MAX_QBITS).try_into().unwrap());
        }
        Ok(result)
    }

    fn with_qr(qbits: NonZeroU8, rbits: NonZeroU8) -> Result<Filter, Error> {
        Self::check_cpu_support();
        if qbits.get() + rbits.get() > 64 {
            return Err(Error::NotEnoughFingerprintBits);
        }
        let buffer_bytes = calculate_memory_bytes(qbits.get(), rbits.get());
        let buffer = vec![0u8; buffer_bytes].into_boxed_slice();
        Ok(Self {
            buffer,
            qbits,
            rbits,
            len: 0,
            max_qbits: None,
        })
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

    /// Returns the fingerprint size in bits.
    ///
    /// Use this with [`compute_fingerprint()`] to create compatible fingerprints.
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
        self.buffer.len()
    }

    /// Removes all items from the filter.
    pub fn clear(&mut self) {
        self.buffer.fill(0);
        self.len = 0;
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

    #[inline]
    fn block_byte_size(&self) -> usize {
        1 + 8 + 8 + 64 * self.rbits.usize() / 8
    }

    #[inline]
    fn set_block_runends(&mut self, block_num: u64, runends: u64) {
        let block_num = block_num % self.total_blocks();
        let block_start = block_num as usize * self.block_byte_size();
        // SAFETY: block_num % total_blocks() guarantees valid block index
        unsafe { self.write_u64_unchecked(block_start + 1 + 8, runends) };
    }

    /// Read u64 from buffer at given offset without bounds checking.
    /// SAFETY: Caller must ensure offset + 8 <= buffer.len()
    #[inline(always)]
    unsafe fn read_u64_unchecked(&self, offset: usize) -> u64 {
        debug_assert!(offset + 8 <= self.buffer.len());
        u64::from_le_bytes(
            self.buffer
                .get_unchecked(offset..offset + 8)
                .try_into()
                .unwrap_unchecked(),
        )
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
    fn raw_block(&self, block_num: u64) -> Block {
        let block_num = block_num % self.total_blocks();
        let block_start = block_num as usize * self.block_byte_size();
        // SAFETY: block_num % total_blocks() guarantees valid block index
        unsafe {
            Block {
                offset: *self.buffer.get_unchecked(block_start) as u64,
                occupieds: self.read_u64_unchecked(block_start + 1),
                runends: self.read_u64_unchecked(block_start + 1 + 8),
            }
        }
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
    fn is_occupied(&self, hash_bucket_idx: u64) -> bool {
        let hash_bucket_idx = hash_bucket_idx % self.total_buckets();
        let block_start = (hash_bucket_idx / 64) as usize * self.block_byte_size();
        // SAFETY: hash_bucket_idx % total_buckets() guarantees valid block index
        let occupieds = unsafe { self.read_u64_unchecked(block_start + 1) };
        occupieds.is_bit_set((hash_bucket_idx % 64) as usize)
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
    fn is_runend(&self, hash_bucket_idx: u64) -> bool {
        let hash_bucket_idx = hash_bucket_idx % self.total_buckets();
        let block_start = (hash_bucket_idx / 64) as usize * self.block_byte_size();
        // SAFETY: hash_bucket_idx % total_buckets() guarantees valid block index
        let runends = unsafe { self.read_u64_unchecked(block_start + 1 + 8) };
        runends.is_bit_set((hash_bucket_idx % 64) as usize)
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

        // Combine as 128-bit value and shift to extract the remainder bits (branchless)
        let combined = (rem_part0 as u128) | ((rem_part1 as u128) << 64);
        let remainder = (combined >> extra_low) as u64 & ((1u64 << self.rbits.get()) - 1);

        debug_assert!(remainder.leading_zeros() >= 64 - self.rbits.get() as u32);
        remainder
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

    #[cold]
    #[inline(never)]
    fn calc_offset(&self, block_num: u64) -> u64 {
        // The block offset can be calculated as the difference between its position and runstart.
        let block_start = (block_num * 64) % self.total_buckets();
        let mut run_start = self.run_start(block_start);
        if run_start < block_start {
            run_start += self.total_buckets().get();
        }
        run_start - block_start
    }

    /// Start idx of of the run (inclusive)
    #[inline]
    fn run_start(&self, hash_bucket_idx: u64) -> u64 {
        // runstart is equivalent to the runend of the previous bucket + 1.
        let prev_bucket = hash_bucket_idx.wrapping_sub(1) % self.total_buckets();
        (self.run_end(prev_bucket) + 1) % self.total_buckets()
    }

    /// End idx of the run for the given bucket (inclusive).
    /// Inlined when select is small (BMI2), not inlined when select is large (broadword).
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
        let bucket_intrablock_offset = hash_bucket_idx % 64;
        let bucket_block = self.raw_block(bucket_block_idx);

        // Handle offset overflow (255 = needs recalculation)
        let offset = if bucket_block.offset >= u8::MAX as u64 {
            self.calc_offset(bucket_block_idx)
        } else {
            bucket_block.offset
        };
        let bucket_intrablock_rank = bucket_block.occupieds.popcnt(..=bucket_intrablock_offset);

        // No occupied buckets up to this position
        if bucket_intrablock_rank == 0 {
            return if offset <= bucket_intrablock_offset {
                // Empty bucket unaffected by spillover, end == start
                hash_bucket_idx
            } else {
                // Bucket falls within spillover from previous block
                (bucket_block_idx * 64 + offset - 1) % self.total_buckets()
            };
        }

        // Search for the runend_rank'th runend bit across blocks
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
            // Not enough runends in this block, continue to next
            runend_rank -= runend_block.runends.popcnt(runend_ignore_bits..);
            runend_block_idx += 1;
            runend_ignore_bits = 0;
            runend_block = self.raw_block(runend_block_idx);
        }
    }

    /// Returns `true` if the item is probably in the filter, `false` if definitely not.
    ///
    /// May return false positives but never false negatives.
    pub fn contains<T: Hash>(&self, item: T) -> bool {
        self.contains_fingerprint(self.hash(item))
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
    /// Only meaningful if duplicates were inserted via [`Self::insert_duplicated()`].
    pub fn count<T: Hash>(&self, item: T) -> u64 {
        self.count_fingerprint(self.hash(item))
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

    /// Returns true if the slot at hash_bucket_idx is empty (not shifted into by other runs).
    #[inline]
    fn is_slot_empty(&self, hash_bucket_idx: u64) -> bool {
        let bucket_block_idx = hash_bucket_idx / 64;
        let bucket_intrablock_offset = hash_bucket_idx % 64;
        let bucket_block = self.raw_block(bucket_block_idx);

        // Spillover run from previous block extends past this position
        if bucket_block.offset > bucket_intrablock_offset {
            return false;
        }

        let num_occupied = bucket_block.occupieds.popcnt(..=bucket_intrablock_offset);
        let num_runends = bucket_block
            .runends
            .popcnt(bucket_block.offset..bucket_intrablock_offset);
        num_occupied == num_runends
    }

    /// Finds the first empty slot at or after `hash_bucket_idx`.
    /// Uses offset_lower_bound to skip over occupied slots efficiently.
    #[inline]
    fn find_first_empty_slot(&self, mut hash_bucket_idx: u64) -> u64 {
        let mut bucket_intrablock_offset = hash_bucket_idx % 64;
        let mut bucket_block = self.raw_block(hash_bucket_idx / 64);

        loop {
            // olb = (runs started at or before here) - (runs ended before here)
            let num_occupied = bucket_block.occupieds.popcnt(..=bucket_intrablock_offset);
            let olb = if bucket_block.offset <= bucket_intrablock_offset {
                // Normal case: count runends in [offset, position)
                num_occupied
                    - bucket_block
                        .runends
                        .popcnt(bucket_block.offset..bucket_intrablock_offset)
            } else {
                // Spillover run from previous block extends past this position
                bucket_block.offset + num_occupied - bucket_intrablock_offset
            };

            if olb == 0 {
                return hash_bucket_idx % self.total_buckets();
            }

            // Advance position by olb slots
            hash_bucket_idx += olb;
            bucket_intrablock_offset += olb;

            // Handle block boundary crossing
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

    /// Removes `item` from the filter. Returns `true` if found and removed.
    ///
    /// **Warning:** Removing an item that wasn't inserted may cause false negatives
    /// for other items with colliding fingerprints.
    pub fn remove<T: Hash>(&mut self, item: T) -> bool {
        self.remove_fingerprint(self.hash(item))
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
    pub fn insert_duplicated<T: Hash>(&mut self, item: T) -> Result<(), Error> {
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
    pub fn insert<T: Hash>(&mut self, item: T) -> Result<bool, Error> {
        self.insert_counting(1, item).map(|count| count == 0)
    }

    /// Inserts `item` up to `max_count` times.
    ///
    /// # Returns
    ///
    /// - `Ok(prev_count)` where `prev_count` is how many times the item was already present.
    ///   A new copy is inserted only if `prev_count < max_count`.
    /// - `Err(Error::CapacityExceeded)` if the filter is full.
    pub fn insert_counting<T: Hash>(&mut self, max_count: u64, item: T) -> Result<u64, Error> {
        let hash = self.hash(item);
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
    /// hash values. Use [`compute_fingerprint()`] to compute compatible fingerprints.
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

    /// Returns an iterator over the fingerprints stored in the filter.
    ///
    /// Fingerprints are yielded in ascending order. Each value has only the lower
    /// [`Self::fingerprint_size()`] bits set (upper bits are zero).
    ///
    /// This is useful for serialization, migrating data between filters, or
    /// inspecting stored values. Use [`compute_fingerprint()`] to compute a
    /// fingerprint compatible with this filter's size.
    pub fn fingerprints(&self) -> FingerprintIter<'_> {
        FingerprintIter::new(self)
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
            let mut new = Self::with_qr(
                (self.qbits.get() - 1).try_into().unwrap(),
                (self.rbits.get() + 1).try_into().unwrap(),
            )
            .unwrap();
            new.max_qbits = self.max_qbits;
            for hash in self.fingerprints() {
                let _ = new.insert_fingerprint(true, hash);
            }
            debug_assert_eq!(new.len, self.len);
            debug_assert_eq!(new.fingerprint_size(), self.fingerprint_size());
            *self = new;
        }
    }

    /// Merges all fingerprints from `other` into `self`.
    ///
    /// # Parameters
    ///
    /// - `keep_duplicates`: If `true`, allows duplicate fingerprints (for counting).
    /// - `other`: Source filter. Must have `fingerprint_size() >= self.fingerprint_size()`.
    ///
    /// # Errors
    ///
    /// - [`Error::IncompatibleFingerprintSize`] if `other` has smaller fingerprints.
    /// - [`Error::CapacityExceeded`] if `self` becomes full (partial merge may occur).
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
    pub fn merge(&mut self, keep_duplicates: bool, other: &Self) -> Result<(), Error> {
        if other.fingerprint_size() < self.fingerprint_size() {
            return Err(Error::IncompatibleFingerprintSize);
        }
        let max_count = if keep_duplicates { u64::MAX } else { 1 };
        for hash in other.fingerprints() {
            self.insert_impl(max_count, hash)?;
        }
        Ok(())
    }

    #[inline]
    fn grow_if_possible(&mut self) -> Result<(), Error> {
        if let Some(m) = self.max_qbits {
            if m > self.qbits {
                self.grow();
                return Ok(());
            }
        }
        Err(Error::CapacityExceeded)
    }

    #[cold]
    #[inline(never)]
    fn grow(&mut self) {
        let qbits = self.qbits.checked_add(1).unwrap();
        let rbits = NonZeroU8::new(self.rbits.get() - 1).unwrap();
        let mut new = Self::with_qr(qbits, rbits).unwrap();
        new.max_qbits = self.max_qbits;
        for hash in self.fingerprints() {
            new.insert_fingerprint(true, hash).unwrap();
        }
        assert_eq!(self.len, new.len);
        *self = new;
    }

    #[inline]
    fn hash<T: Hash>(&self, item: T) -> u64 {
        let mut hasher = StableHasher::new();
        item.hash(&mut hasher);
        hasher.finish()
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

impl std::fmt::Debug for Filter {
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
        f.grow();
        f.grow();
        f.grow();
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
            let est_fp_rate =
                (0..).take(50_000).filter(|i| f.contains(i)).count() as f64 / 50_000.0;
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
            assert!(matches!(
                f1.merge(true, &f1.clone()),
                Err(Error::CapacityExceeded)
            ));
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

    #[test]
    fn test_dec_offset_edge_case() {
        // case found in fuzz testing
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
        assert_eq!(f.raw_block(2).offset, 3);
        assert_eq!(f.raw_block(3).offset, u8::MAX as u64);
        f.validate_offsets(0, f.total_buckets().get());
        f.remove(0u16);
        assert_eq!(f.raw_block(2).offset, 2);
        assert_eq!(f.raw_block(3).offset, 254);
        f.validate_offsets(0, f.total_buckets().get());
    }

    #[test]
    fn test_capacity_edge_cases() {
        for n in 1..32 {
            let base = (1 << n) * 19 / 20;
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
                ((1u64 << (i - 1).min(Filter::MAX_QBITS)) * 19).div_ceil(20)
            );
        }
        for i in 1..Filter::MAX_QBITS {
            let f = Filter::new_resizeable(0, 2u64.pow(i as u32), 0.5).unwrap();
            assert_eq!(f.capacity(), 61);
            assert!(f.capacity() <= f.max_capacity());
        }
        // Test the maximum capacity
        let f = Filter::new_resizeable(0, Filter::MAX_CAPACITY, 0.5).unwrap();
        assert_eq!(f.capacity(), 61);
        assert_eq!(f.max_capacity(), Filter::MAX_CAPACITY);
        // Test the maximum capacity + 1, which should fail
        Filter::new_resizeable(0, Filter::MAX_CAPACITY + 1, 0.5).unwrap_err();
    }
}
