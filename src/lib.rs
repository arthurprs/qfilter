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
//! ### Legacy x86_64 CPUs support
//!
//! The implementation assumes the `popcnt` instruction (equivalent to `integer.count_ones()`) is present
//! when compiling for x86_64 targets. This is theoretically not guaranteed as the instruction in only
//! available on AMD/Intel CPUs released after 2007/2008. If that's not the case the Filter constructor will panic.
//!
//! Support for such legacy x86_64 CPUs can be optionally enabled with the `legacy_x86_64_support`
//! which incurs a ~10% performance penalty.
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

mod stable_hasher;

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
    #[cfg_attr(
        feature = "serde",
        serde(rename = "g", skip_serializing_if = "Option::is_none", default)
    )]
    max_qbits: Option<NonZeroU8>,
}

#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The filter cannot fit another fingerprint
    CapacityExceeded,
    /// The fingerprint sizes are not compatible
    IncompatibleFingerprintSize,
    /// The specified filter cannot be constructed with 64 bit hashes
    NotEnoughFingerprintBits,
    /// Capacity is too large (>= u64::MAX / 20)
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
        let bitmask = |n| !u64::MAX.checked_shl(n).unwrap_or(0);
        let a_component = *self >> (64 - bits); // select the highest `bits` from A to become lowest
        let b_shifted_mask = bitmask((b_end - b_start) as u32) << b_start;
        let b_shifted = ((b_shifted_mask & b) << bits) & b_shifted_mask;
        let b_mask = !b_shifted_mask;

        a_component | b_shifted | (b & b_mask)
    }

    #[inline]
    fn shift_left(&self, bits: usize, b: &Self, b_start: usize, b_end: usize) -> Self {
        let bitmask = |n| !u64::MAX.checked_shl(n).unwrap_or(0);
        let a_component = *self << (64 - bits); // select the lowest `bits` from A to become highest
        let b_shifted_mask = bitmask((b_end - b_start) as u32) << b_start;
        let b_shifted = ((b_shifted_mask & b) >> bits) & b_shifted_mask;
        let b_mask = !b_shifted_mask;

        a_component | b_shifted | (b & b_mask)
    }

    #[inline]
    fn popcnt(&self, range: impl RangeBounds<u64>) -> u64 {
        let mut v = match range.start_bound() {
            std::ops::Bound::Included(&i) => *self >> i << i,
            std::ops::Bound::Excluded(&i) => *self >> (i + 1) << (i + 1),
            _ => *self,
        };
        v = match range.end_bound() {
            std::ops::Bound::Included(&i) if i < 63 => v & ((2 << i) - 1),
            std::ops::Bound::Excluded(&i) if i <= 63 => v & ((1 << i) - 1),
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
        let v = *self >> range.start << range.start;

        #[cfg_attr(target_arch = "x86_64", cold)]
        #[cfg_attr(not(target_arch = "x86_64"), inline)]
        fn fallback(mut v: u64, n: u64) -> Option<u64> {
            for _ in 0..n / 8 {
                for _ in 0..8 {
                    v &= v.wrapping_sub(1); // remove the least significant bit
                }
            }
            for _ in 0..n % 8 {
                v &= v.wrapping_sub(1); // remove the least significant bit
            }

            if v == 0 {
                None
            } else {
                Some(v.trailing_zeros() as u64)
            }
        }

        #[cfg(target_arch = "x86_64")]
        let result = {
            // TODO: AMD CPUs up to Zen2 have slow BMI implementations
            if std::is_x86_feature_detected!("bmi2") {
                // This is the equivalent intrinsics version of the inline assembly below.
                // #[target_feature(enable = "bmi1")]
                // #[target_feature(enable = "bmi2")]
                // #[inline]
                // unsafe fn select_bmi2(x: u64, k: u64) -> Option<u64> {
                //     use std::arch::x86_64::{_pdep_u64, _tzcnt_u64};
                //     let result = _tzcnt_u64(_pdep_u64(1 << k, x));
                //     if result != 64 {
                //         Some(result)
                //     } else {
                //         None
                //     }
                // }
                // unsafe { select_bmi2(v, n) }

                let result: u64;
                unsafe {
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
                }
                if result != 64 {
                    Some(result)
                } else {
                    None
                }
            } else {
                fallback(v, n)
            }
        };
        #[cfg(not(target_arch = "x86_64"))]
        let result = fallback(v, n);

        result
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

/// An iterator over the fingerprints of a `Filter`.
pub struct FingerprintIter<'a> {
    filter: &'a Filter,
    q_bucket_idx: u64,
    r_bucket_idx: u64,
    remaining: u64,
}

impl<'a> FingerprintIter<'a> {
    fn new(filter: &'a Filter) -> Self {
        let mut iter = FingerprintIter {
            filter,
            q_bucket_idx: 0,
            r_bucket_idx: 0,
            remaining: filter.len,
        };
        if !filter.is_empty() {
            while !filter.is_occupied(iter.q_bucket_idx) {
                iter.q_bucket_idx += 1;
            }
            iter.r_bucket_idx = filter.run_start(iter.q_bucket_idx);
        }
        iter
    }
}

impl Iterator for FingerprintIter<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(r) = self.remaining.checked_sub(1) {
            self.remaining = r;
        } else {
            return None;
        }
        let hash = (self.q_bucket_idx << self.filter.rbits.get())
            | self.filter.get_remainder(self.r_bucket_idx);

        if self.filter.is_runend(self.r_bucket_idx) {
            self.q_bucket_idx += 1;
            while !self.filter.is_occupied(self.q_bucket_idx) {
                self.q_bucket_idx += 1;
            }
            self.r_bucket_idx = (self.r_bucket_idx + 1).max(self.q_bucket_idx);
        } else {
            self.r_bucket_idx += 1;
        }

        Some(hash)
    }
}

impl Filter {
    /// Creates a new filter that can hold at least `capacity` items
    /// and with a desired error rate of `fp_rate` (clamped to (0, 0.5]).
    ///
    /// Errors if capacity is >= `u64::MAX / 20` or if the specified filter isn't achievable using 64 bit hashes.
    #[inline]
    pub fn new(capacity: u64, fp_rate: f64) -> Result<Self, Error> {
        Self::new_resizeable(capacity, capacity, fp_rate)
    }

    /// Creates a new filter that can hold at least `initial_capacity` items initially
    /// and can resize to hold at least `max_capacity` when fully grown.
    /// The desired error rate `fp_rate` (clamped to (0, 0.5]) applies to the fully grown filter.
    ///
    /// This works by storing fingerprints large enough to satisfy the maximum requirements,
    /// so smaller filters will actually have lower error rates, which will increase
    /// (up to `fp_rate`) as the filter grows. In practice every time the filter doubles in
    /// capacity its error rate also doubles.
    ///
    /// Errors if capacity is >= `u64::MAX / 20` or if the specified filter isn't achievable using 64 bit hashes.
    pub fn new_resizeable(
        initial_capacity: u64,
        max_capacity: u64,
        fp_rate: f64,
    ) -> Result<Self, Error> {
        assert!(max_capacity >= initial_capacity);
        let fp_rate = fp_rate.clamp(f64::MIN_POSITIVE, 0.5);
        // Calculate necessary slots to achieve capacity with up to 95% occupancy
        // 19/20 == 0.95
        let max_capacity = (max_capacity
            .checked_mul(20)
            .ok_or(Error::CapacityTooLarge)?
            / 19)
            .next_power_of_two()
            .max(64);
        let max_qbits = max_capacity.trailing_zeros() as u8;
        let initial_capacity = (initial_capacity * 20 / 19).next_power_of_two().max(64);
        let qbits = initial_capacity.trailing_zeros() as u8;
        let rbits = (-fp_rate.log2()).round().max(1.0) as u8 + (max_qbits - qbits);
        let mut result = Self::with_qr(qbits.try_into().unwrap(), rbits.try_into().unwrap())?;
        if max_qbits > qbits {
            result.max_qbits = Some(max_qbits.try_into().unwrap());
        }
        Ok(result)
    }

    /// Creates a new resizeable filter that can hold at least `initial_capacity` items initially while
    /// utilizing a fingerprint bit size of `fingerprint_bits` (7..=64). Normally this function is only
    /// useful if the filter is being used to manually store fingerprints.
    pub fn with_fingerprint_size(
        initial_capacity: u64,
        fingerprint_bits: u8,
    ) -> Result<Filter, Error> {
        if !(7..=64).contains(&fingerprint_bits) {
            return Err(Error::NotEnoughFingerprintBits);
        }
        let initial_capacity = (initial_capacity
            .checked_mul(20)
            .ok_or(Error::CapacityTooLarge)?
            / 19)
            .next_power_of_two()
            .max(64);
        let qbits = initial_capacity.trailing_zeros() as u8;
        if fingerprint_bits <= qbits {
            return Err(Error::NotEnoughFingerprintBits);
        }
        let rbits = fingerprint_bits - qbits;
        let mut result = Self::with_qr(qbits.try_into().unwrap(), rbits.try_into().unwrap())?;
        if rbits > 1 {
            result.max_qbits = Some((qbits + rbits - 1).try_into().unwrap());
        }
        Ok(result)
    }

    fn with_qr(qbits: NonZeroU8, rbits: NonZeroU8) -> Result<Filter, Error> {
        Self::check_cpu_support();
        if qbits.get() + rbits.get() > 64 {
            return Err(Error::NotEnoughFingerprintBits);
        }
        let num_slots = 1 << qbits.get();
        let num_blocks = num_slots / 64;
        assert_ne!(num_blocks, 0);
        let block_bytes_size = 1 + 16 + 64 * rbits.u64() / 8;
        let buffer_bytes = num_blocks * block_bytes_size;
        let buffer = vec![0u8; buffer_bytes.try_into().unwrap()].into_boxed_slice();
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
    }

    /// The internal fingerprint size in bits.
    #[inline]
    pub fn fingerprint_size(&self) -> u8 {
        self.qbits.get() + self.rbits.get()
    }

    /// Whether the filter is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Current number of fingerprints admitted to the filter.
    #[inline]
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Resets/Clears the filter.
    pub fn clear(&mut self) {
        self.buffer.fill(0);
        self.len = 0;
    }

    /// Maximum filter capacity.
    #[inline]
    pub fn capacity_resizeable(&self) -> u64 {
        // Overflow is not possible here as it'd have overflowed in the constructor.
        (1 << self.max_qbits.unwrap_or(self.qbits).get()) * 19 / 20
    }

    /// Current filter capacity.
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
            self.total_buckets().get() * 19 / 20
        }
    }

    /// Max error ratio when at the resizeable capacity (len == resizeable_capacity).
    pub fn max_error_ratio_resizeable(&self) -> f64 {
        let extra_rbits = self.max_qbits.unwrap_or(self.qbits).get() - self.qbits.get();
        2f64.powi(-((self.rbits.get() - extra_rbits) as i32))
    }

    /// Max error ratio when at full capacity (len == capacity).
    pub fn max_error_ratio(&self) -> f64 {
        2f64.powi(-(self.rbits.get() as i32))
    }

    /// Current error ratio at the current occupancy.
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
        let block_bytes: &mut [u8; 1 + 8 + 8] = (&mut self.buffer[block_start..][..1 + 8 + 8])
            .try_into()
            .unwrap();
        block_bytes[1 + 8..1 + 8 + 8].copy_from_slice(&runends.to_le_bytes());
    }

    #[inline]
    fn raw_block(&self, block_num: u64) -> Block {
        let block_num = block_num % self.total_blocks();
        let block_start = block_num as usize * self.block_byte_size();
        let block_bytes: &[u8; 1 + 8 + 8] =
            &self.buffer[block_start..][..1 + 8 + 8].try_into().unwrap();
        Block {
            offset: block_bytes[0] as u64,
            occupieds: u64::from_le_bytes(block_bytes[1..1 + 8].try_into().unwrap()),
            runends: u64::from_le_bytes(block_bytes[1 + 8..1 + 8 + 8].try_into().unwrap()),
        }
    }

    #[inline]
    fn block(&self, block_num: u64) -> Block {
        let block_num = block_num % self.total_blocks();
        let block_start = block_num as usize * self.block_byte_size();
        let block_bytes: &[u8; 1 + 8 + 8] = &self.buffer[block_start..block_start + 1 + 8 + 8]
            .try_into()
            .unwrap();
        let offset = {
            if block_bytes[0] < u8::MAX {
                block_bytes[0] as u64
            } else {
                self.calc_offset(block_num)
            }
        };
        Block {
            offset,
            occupieds: u64::from_le_bytes(block_bytes[1..1 + 8].try_into().unwrap()),
            runends: u64::from_le_bytes(block_bytes[1 + 8..1 + 8 + 8].try_into().unwrap()),
        }
    }

    #[inline]
    fn adjust_block_offset(&mut self, block_num: u64, inc: bool) {
        let block_num = block_num % self.total_blocks();
        let block_start = block_num as usize * self.block_byte_size();
        let offset = &mut self.buffer[block_start];
        if inc {
            *offset = offset.saturating_add(1);
        } else if *offset != u8::MAX {
            *offset -= 1;
        } else {
            self.buffer[block_start] = self.calc_offset(block_num).try_into().unwrap_or(u8::MAX);
        }
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
        let occupieds = u64::from_le_bytes(
            self.buffer[block_start + 1..block_start + 1 + 8]
                .try_into()
                .unwrap(),
        );
        occupieds.is_bit_set((hash_bucket_idx % 64) as usize)
    }

    #[inline(always)]
    fn set_occupied(&mut self, hash_bucket_idx: u64, value: bool) {
        let hash_bucket_idx = hash_bucket_idx % self.total_buckets();
        let block_start = (hash_bucket_idx / 64) as usize * self.block_byte_size();
        let mut occupieds = u64::from_le_bytes(
            self.buffer[block_start + 1..block_start + 1 + 8]
                .try_into()
                .unwrap(),
        );
        occupieds.update_bit((hash_bucket_idx % 64) as usize, value);
        self.buffer[block_start + 1..block_start + 1 + 8].copy_from_slice(&occupieds.to_le_bytes());
    }

    #[inline(always)]
    fn is_runend(&self, hash_bucket_idx: u64) -> bool {
        let hash_bucket_idx = hash_bucket_idx % self.total_buckets();
        let block_start = (hash_bucket_idx / 64) as usize * self.block_byte_size();
        let runends = u64::from_le_bytes(
            self.buffer[block_start + 1 + 8..block_start + 1 + 8 + 8]
                .try_into()
                .unwrap(),
        );
        runends.is_bit_set((hash_bucket_idx % 64) as usize)
    }

    #[inline(always)]
    fn set_runend(&mut self, hash_bucket_idx: u64, value: bool) {
        let hash_bucket_idx = hash_bucket_idx % self.total_buckets();
        let block_start = (hash_bucket_idx / 64) as usize * self.block_byte_size();
        let mut runends = u64::from_le_bytes(
            self.buffer[block_start + 1 + 8..block_start + 1 + 8 + 8]
                .try_into()
                .unwrap(),
        );
        runends.update_bit((hash_bucket_idx % 64) as usize, value);
        self.buffer[block_start + 1 + 8..block_start + 1 + 8 + 8]
            .copy_from_slice(&runends.to_le_bytes());
    }

    #[inline(always)]
    fn get_remainder(&self, hash_bucket_idx: u64) -> u64 {
        debug_assert!(self.rbits.get() > 0 && self.rbits.get() < 64);
        let hash_bucket_idx = hash_bucket_idx % self.total_buckets();
        let remainders_start = (hash_bucket_idx / 64) as usize * self.block_byte_size() + 1 + 8 + 8;
        let start_bit_idx = self.rbits.usize() * (hash_bucket_idx % 64) as usize;
        let end_bit_idx = start_bit_idx + self.rbits.usize();
        let start_u64 = start_bit_idx / 64;
        let num_rem_parts = 1 + (end_bit_idx > (start_u64 + 1) * 64) as usize;
        let rem_parts_bytes = &self.buffer[remainders_start + start_u64 * 8..][..num_rem_parts * 8];
        let extra_low = start_bit_idx - start_u64 * 64;
        let extra_high = ((start_u64 + 1) * 64).saturating_sub(end_bit_idx);
        let rem_part = u64::from_le_bytes(rem_parts_bytes[..8].try_into().unwrap());
        // zero high bits & truncate low bits
        let mut remainder = (rem_part << extra_high) >> (extra_high + extra_low);
        if let Some(rem_part) = rem_parts_bytes.get(8..16) {
            let remaining_bits = end_bit_idx - (start_u64 + 1) * 64;
            let rem_part = u64::from_le_bytes(rem_part.try_into().unwrap());
            remainder |=
                (rem_part & !(u64::MAX << remaining_bits)) << (self.rbits.usize() - remaining_bits);
        }
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
        let end_bit_idx = start_bit_idx + self.rbits.usize();
        let start_u64 = start_bit_idx / 64;
        let num_rem_parts = 1 + (end_bit_idx > (start_u64 + 1) * 64) as usize;
        let rem_parts_bytes =
            &mut self.buffer[remainders_start + start_u64 * 8..][..num_rem_parts * 8];
        let mut rem_part = u64::from_le_bytes(rem_parts_bytes[..8].try_into().unwrap());
        let extra_low = start_bit_idx - start_u64 * 64;
        let extra_high = ((start_u64 + 1) * 64).saturating_sub(end_bit_idx);
        // zero region we'll copy remainder bits in
        rem_part &= !((u64::MAX << extra_low) & (u64::MAX >> extra_high));
        let low_bits_to_copy = 64 - extra_high - extra_low;
        rem_part |= (remainder & !(u64::MAX << low_bits_to_copy)) << extra_low;
        rem_parts_bytes[..8].copy_from_slice(&rem_part.to_le_bytes());
        if rem_parts_bytes.len() < 16 {
            return;
        }

        let remaining_bits = end_bit_idx - (start_u64 + 1) * 64;
        rem_part = u64::from_le_bytes(rem_parts_bytes[8..16].try_into().unwrap());
        // zero region we'll copy remainder bits in
        rem_part &= u64::MAX << remaining_bits;
        rem_part |= remainder >> (self.rbits.usize() - remaining_bits);
        rem_parts_bytes[8..16].copy_from_slice(&rem_part.to_le_bytes());
    }

    #[inline]
    fn get_rem_u64(&self, rem_u64: u64) -> u64 {
        let rbits = NonZeroU64::from(self.rbits);
        let bucket_block_idx = (rem_u64 / rbits) % self.total_blocks();
        let bucket_rem_u64 = (rem_u64 % rbits) as usize;
        let bucket_rem_start = (bucket_block_idx as usize * self.block_byte_size()) + 1 + 8 + 8;
        u64::from_le_bytes(
            self.buffer[bucket_rem_start + bucket_rem_u64 * 8..][..8]
                .try_into()
                .unwrap(),
        )
    }

    #[inline]
    fn set_rem_u64(&mut self, rem_u64: u64, rem: u64) {
        let rbits = NonZeroU64::from(self.rbits);
        let bucket_block_idx = (rem_u64 / rbits) % self.total_blocks();
        let bucket_rem_u64 = (rem_u64 % rbits) as usize;
        let bucket_rem_start = (bucket_block_idx as usize * self.block_byte_size()) + 1 + 8 + 8;
        self.buffer[bucket_rem_start + bucket_rem_u64 * 8..][..8]
            .copy_from_slice(&rem.to_le_bytes());
    }

    fn shift_remainders_by_1(&mut self, start: u64, end_inc: u64) {
        let end = if end_inc < start {
            end_inc + self.total_buckets().get() + 1
        } else {
            end_inc + 1
        };
        let mut end_u64 = end * self.rbits.u64() / 64;
        let mut bend = (end * self.rbits.u64() % 64) as usize;
        let start_u64 = start * self.rbits.u64() / 64;
        let bstart = (start * self.rbits.u64() % 64) as usize;
        while end_u64 != start_u64 {
            let prev_rem_u64 = self.get_rem_u64(end_u64 - 1);
            let mut rem_u64 = self.get_rem_u64(end_u64);
            rem_u64 = prev_rem_u64.shift_right(self.rbits.usize(), &rem_u64, 0, bend);
            self.set_rem_u64(end_u64, rem_u64);
            end_u64 -= 1;
            bend = 64;
        }
        let mut rem_u64 = self.get_rem_u64(start_u64);
        rem_u64 = 0u64.shift_right(self.rbits.usize(), &rem_u64, bstart, bend);
        self.set_rem_u64(start_u64, rem_u64);
    }

    fn shift_remainders_back_by_1(&mut self, start: u64, end_inc: u64) {
        let end = if end_inc < start {
            end_inc + self.total_buckets().get() + 1
        } else {
            end_inc + 1
        };
        let end_u64 = end * self.rbits.u64() / 64;
        let bend = (end * self.rbits.u64() % 64) as usize;
        let mut start_u64 = start * self.rbits.u64() / 64;
        let mut bstart = (start * self.rbits.u64() % 64) as usize;
        while end_u64 != start_u64 {
            let next_rem_u64 = self.get_rem_u64(start_u64 + 1);
            let mut rem_u64 = self.get_rem_u64(start_u64);
            rem_u64 = next_rem_u64.shift_left(self.rbits.usize(), &rem_u64, bstart, 64);
            self.set_rem_u64(start_u64, rem_u64);
            start_u64 += 1;
            bstart = 0;
        }
        let mut rem_u64 = self.get_rem_u64(end_u64);
        rem_u64 = 0u64.shift_left(self.rbits.usize(), &rem_u64, bstart, bend);
        self.set_rem_u64(end_u64, rem_u64);
    }

    fn shift_runends_by_1(&mut self, start: u64, end_inc: u64) {
        let end = if end_inc < start {
            end_inc + self.total_buckets().get() + 1
        } else {
            end_inc + 1
        };
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
        let end = if end_inc < start {
            end_inc + self.total_buckets().get() + 1
        } else {
            end_inc + 1
        };
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

    /// End idx of the end of the run (inclusive).
    fn run_end(&self, hash_bucket_idx: u64) -> u64 {
        let hash_bucket_idx = hash_bucket_idx % self.total_buckets();
        let bucket_block_idx = hash_bucket_idx / 64;
        let bucket_intrablock_offset = hash_bucket_idx % 64;
        let bucket_block = self.block(bucket_block_idx);
        let bucket_intrablock_rank = bucket_block.occupieds.popcnt(..=bucket_intrablock_offset);
        // No occupied buckets all the way to bucket_intrablock_offset
        // which also means hash_bucket_idx isn't occupied
        if bucket_intrablock_rank == 0 {
            return if bucket_block.offset <= bucket_intrablock_offset {
                // hash_bucket_idx points to an empty bucket unaffected by block offset,
                // thus end == start
                hash_bucket_idx
            } else {
                // hash_bucket_idx fall within the section occupied by the offset,
                // thus end == last bucket of offset section
                (bucket_block_idx * 64 + bucket_block.offset - 1) % self.total_buckets()
            };
        }

        // Must search runends to figure out the end of the run
        let mut runend_block_idx = bucket_block_idx + bucket_block.offset / 64;
        let mut runend_ignore_bits = bucket_block.offset % 64;
        let mut runend_block = self.raw_block(runend_block_idx);
        // Try to find the runend for the bucket in this block.
        // We're looking for the runend_rank'th bit set (0 based)
        let mut runend_rank = bucket_intrablock_rank - 1;
        let mut runend_block_offset = runend_block
            .runends
            .select(runend_ignore_bits.., runend_rank);

        if let Some(runend_block_offset) = runend_block_offset {
            let runend_idx = runend_block_idx * 64 + runend_block_offset;
            return runend_idx.max(hash_bucket_idx) % self.total_buckets();
        }
        // There were not enough runend bits set, keep looking...
        loop {
            // subtract any runend bits found
            runend_rank -= runend_block.runends.popcnt(runend_ignore_bits..);
            // move to the next block
            runend_block_idx += 1;
            runend_ignore_bits = 0;
            runend_block = self.raw_block(runend_block_idx);
            runend_block_offset = runend_block
                .runends
                .select(runend_ignore_bits.., runend_rank);

            if let Some(runend_block_offset) = runend_block_offset {
                let runend_idx = runend_block_idx * 64 + runend_block_offset;
                return runend_idx.max(hash_bucket_idx) % self.total_buckets();
            }
        }
    }

    /// Returns whether item is present (probabilistically) in the filter.
    pub fn contains<T: Hash>(&self, item: T) -> bool {
        self.contains_fingerprint(self.hash(item))
    }

    /// Returns whether the fingerprint is present (probabilistically) in the filter.
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

    /// Returns the number of times the item appears (probabilistically) in the filter.
    pub fn count<T: Hash>(&mut self, item: T) -> u64 {
        self.count_fingerprint(self.hash(item))
    }

    /// Returns the amount of times the fingerprint appears (probabilistically) in the filter.
    pub fn count_fingerprint(&mut self, hash: u64) -> u64 {
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

    #[inline]
    fn offset_lower_bound(&self, hash_bucket_idx: u64) -> u64 {
        let bucket_block_idx = hash_bucket_idx / 64;
        let bucket_intrablock_offset = hash_bucket_idx % 64;
        let bucket_block = self.raw_block(bucket_block_idx);
        let num_occupied = bucket_block.occupieds.popcnt(..=bucket_intrablock_offset);
        if bucket_block.offset <= bucket_intrablock_offset {
            num_occupied
                - bucket_block
                    .runends
                    .popcnt(bucket_block.offset..bucket_intrablock_offset)
        } else {
            bucket_block.offset + num_occupied - bucket_intrablock_offset
        }
    }

    fn find_first_empty_slot(&self, mut hash_bucket_idx: u64) -> u64 {
        loop {
            let olb = self.offset_lower_bound(hash_bucket_idx);
            if olb == 0 {
                return hash_bucket_idx % self.total_buckets();
            }
            hash_bucket_idx += olb;
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

    /// Removes `item` from the filter.
    /// Returns whether item was actually found and removed.
    ///
    /// Note that removing an item who wasn't previously added to the filter
    /// may introduce **false negatives**. This is because it could be removing
    /// fingerprints from a colliding item!
    pub fn remove<T: Hash>(&mut self, item: T) -> bool {
        self.remove_fingerprint(self.hash(item))
    }

    /// Removes the fingerprint specified by `hash` was from the filter.
    /// Returns whether a fingerprint was actually found and removed.
    ///
    /// Note that removing a fingerprint that wasn't previously added to the filter
    /// may introduce false negatives. This is because it could be removing
    /// fingerprints from a colliding hash!
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

    /// Inserts `item` in the filter, even if already appears to be in the filter.
    /// This works by inserting a possibly duplicated fingerprint in the filter.
    ///
    /// This function should be used when the filter is also subject to removals
    /// and the item is known to not have been added to the filter before (or was removed).
    ///
    /// Returns `Err(Error::CapacityExceeded)` if the filter cannot admit the new item.
    pub fn insert_duplicated<T: Hash>(&mut self, item: T) -> Result<(), Error> {
        let hash = self.hash(item);
        match self.insert_impl(true, hash) {
            Ok(_added) => Ok(()),
            Err(_) => {
                self.grow_if_possible()?;
                self.insert_impl(true, hash).map(|_| ())
            }
        }
    }

    /// Inserts `item` in the filter if it's not already present (probabilistically).
    /// Note that membership is probabilistic, so this function may return false positives
    /// but never false negatives.
    ///
    /// Returns `Ok(true)` if the item was successfully added to the filter.
    /// Returns `Ok(false)` if the item is already contained (probabilistically) in the filter.
    /// Returns `Err(Error::CapacityExceeded)` if the filter cannot admit the new item.
    pub fn insert<T: Hash>(&mut self, item: T) -> Result<bool, Error> {
        let hash = self.hash(item);
        match self.insert_impl(false, hash) {
            Ok(added) => Ok(added),
            Err(_) => {
                self.grow_if_possible()?;
                self.insert_impl(false, hash)
            }
        }
    }

    /// Inserts the fingerprint specified by `hash` in the filter.
    /// `duplicate` specifies if the fingerprint should be added even if it's already in the filter.
    ///
    /// Note that this function will automatically grow the filter if needed.
    /// The implementation uses the first [`Self::fingerprint_size`] bits of `hash` to place the fingerprint in the appropriate slot.
    /// The remaining bits are ignored and will be returned as 0 if the fingerprint is retrieved via [`Self::fingerprints`].
    ///
    /// Returns `Ok(true)` if the item was successfully added to the filter.
    /// Returns `Ok(false)` if the item is already contained (probabilistically) in the filter.
    /// Returns `Err(Error::CapacityExceeded)` if the filter cannot admit the new item.
    pub fn insert_fingerprint(&mut self, duplicate: bool, hash: u64) -> Result<bool, Error> {
        match self.insert_impl(duplicate, hash) {
            Ok(added) => Ok(added),
            Err(_) => {
                self.grow_if_possible()?;
                self.insert_impl(duplicate, hash)
            }
        }
    }

    /// Inserts the fingerprint specified by `hash` in the filter.
    /// `duplicate` specifies if the fingerprint should be added even if it's already in the filter.
    /// It's up to the caller to grow the filter if needed and retry the insert.
    fn insert_impl(&mut self, duplicate: bool, hash: u64) -> Result<bool, Error> {
        enum Operation {
            NewRun,
            BeforeRunend,
            NewRunend,
        }

        let (hash_bucket_idx, hash_remainder) = self.calc_qr(hash);
        if self.offset_lower_bound(hash_bucket_idx) == 0 {
            if self.len >= self.capacity() {
                return Err(Error::CapacityExceeded);
            }
            debug_assert!(!self.is_occupied(hash_bucket_idx));
            debug_assert!(!self.is_runend(hash_bucket_idx));
            self.set_occupied(hash_bucket_idx, true);
            self.set_runend(hash_bucket_idx, true);
            self.set_remainder(hash_bucket_idx, hash_remainder);
            self.len += 1;
            return Ok(true);
        }

        let mut runstart_idx = self.run_start(hash_bucket_idx);
        let mut runend_idx = self.run_end(hash_bucket_idx);
        let insert_idx;
        let operation;
        if self.is_occupied(hash_bucket_idx) {
            // adjust runend so its >= runstart even if it wrapped around
            if runend_idx < runstart_idx {
                runend_idx += self.total_buckets().get();
            }
            while runstart_idx <= runend_idx {
                match self.get_remainder(runstart_idx).cmp(&hash_remainder) {
                    Ordering::Less => (), // TODO: sorted hashes appears to have no positive impact
                    Ordering::Equal if duplicate => (),
                    Ordering::Equal => return Ok(false),
                    Ordering::Greater => break,
                }

                runstart_idx += 1;
            }

            if runstart_idx > runend_idx {
                /* new remainder is >= than any remainder in the run. */
                operation = Operation::NewRunend;
                insert_idx = runstart_idx % self.total_buckets();
            } else {
                /* there are larger remainders already in the run. */
                operation = Operation::BeforeRunend; /* Inserting */
                insert_idx = runstart_idx % self.total_buckets();
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
        Ok(true)
    }

    /// Returns an iterator over the fingerprints stored in the filter.
    ///
    /// Fingerprints will be returned in ascending order.
    pub fn fingerprints(&self) -> FingerprintIter {
        FingerprintIter::new(self)
    }

    /// Shrinks the capacity of the filter as much as possible while preserving
    /// the false positive ratios and fingerprint size.
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

    /// Merges `other` filter into `self`.
    ///
    /// `keep_duplicates` specifies whether duplicated fingerprints should be store,
    /// this is normally only useful is the filter is being used for counting.
    ///
    /// Note that the `other` filter must have a fingerprint >= `self` fingerprint size,
    /// otherwise the function will fail with `Err(Error::IncompatibleFingerprintSize)`.
    /// This is the case for filters created with the same parameters or if the `other`
    /// filter has a lower target false positive ratio.
    ///
    /// Returns `Err(Error::CapacityExceeded)` if the filter cannot merge all items.
    /// Note that in this case items could have already been added and the filter is left
    /// full but in an otherwise valid state.
    pub fn merge(&mut self, keep_duplicates: bool, other: &Self) -> Result<(), Error> {
        if other.fingerprint_size() < self.fingerprint_size() {
            return Err(Error::IncompatibleFingerprintSize);
        }
        for hash in other.fingerprints() {
            self.insert_impl(keep_duplicates, hash)?;
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
        #[cfg(any(debug_assertions, fuzzing))]
        {
            NonZeroU64::new((1u64 << self.qbits.get()) / 64).unwrap()
        }
        #[cfg(not(any(debug_assertions, fuzzing)))]
        {
            // Safety: All filter have at least 1 block (which have 64 slots each)
            unsafe { NonZeroU64::new_unchecked((1u64 << self.qbits.get()) / 64) }
        }
    }

    #[inline]
    fn total_buckets(&self) -> NonZeroU64 {
        NonZeroU64::new(1 << self.qbits.get()).unwrap()
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
                    assert_eq!(f.count(-1), i as u64 + 1);
                    assert_eq!(f.count(s), i as u64);
                    f.insert_duplicated(s).unwrap();
                    assert_eq!(f.count(-1), i as u64 + 1);
                    assert_eq!(f.count(s), i as u64 + 1);
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
                    assert!(f.count(s) >= i + 1);
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
                assert_eq!(f.len() as u64, c);
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
                assert_eq!(f.len() as u64, c);
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
        assert_eq!(f.capacity_resizeable(), 128 * 19 / 20);
        assert_eq!(f.capacity(), 64 * 19 / 20);
        for i in 0..f.capacity_resizeable() {
            f.insert_fingerprint(false, i).unwrap();
        }
        assert_eq!(f.len() as u64, f.capacity_resizeable());
        assert!(f
            .insert_fingerprint(false, f.capacity_resizeable())
            .is_err());
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
                f1.insert_impl(true, 1),
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
                let mut f01 = qfilter01::Filter::new(capacity, fp_ratio);
                for i in 0..f.capacity() {
                    f.insert(i).unwrap();
                    f01.insert(i).unwrap();
                }

                let ser = serde_cbor::to_vec(&f).unwrap();
                let ser01 = serde_cbor::to_vec(&f01).unwrap();
                // ensure serde output is the same
                assert_eq!(ser, ser01);

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
        let mut f = Filter::new(400, 0.1).unwrap();
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
}
