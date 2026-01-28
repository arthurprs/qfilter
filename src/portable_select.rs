//! Portable bit selection algorithm.
//!
//! Implements Vigna's broadword select algorithm - an O(1), branchless
//! method to find the position of the n-th set bit in a u64.

#![cfg_attr(
    all(
        target_arch = "x86_64",
        not(feature = "legacy_x86_64_support"),
        not(target_feature = "bmi2")
    ),
    allow(dead_code)
)]

const L8: u64 = 0x0101_0101_0101_0101;

/// High bit set in each byte where x <= y
#[inline(always)]
fn le8(x: u64, y: u64) -> u64 {
    ((y | 0x8080_8080_8080_8080).wrapping_sub(x & !0x8080_8080_8080_8080)) & 0x8080_8080_8080_8080
}

/// High bit set in each non-zero byte
#[inline(always)]
fn u_nz8(x: u64) -> u64 {
    ((x | 0x8080_8080_8080_8080).wrapping_sub(L8) | x) & 0x8080_8080_8080_8080
}

/// Select the position of the n-th set bit (0-indexed) in `x`.
///
/// Returns `None` if there are fewer than `n + 1` set bits.
///
/// This implements Vigna's broadword select algorithm, which is O(1)
/// and branchless, requiring no lookup tables.
#[inline]
pub fn select(x: u64, n: u64) -> Option<u64> {
    // Parallel popcount per byte, then prefix sum
    let mut s = x - ((x & 0xAAAA_AAAA_AAAA_AAAA) >> 1);
    s = (s & 0x3333_3333_3333_3333) + ((s >> 2) & 0x3333_3333_3333_3333);
    s = ((s + (s >> 4)) & 0x0F0F_0F0F_0F0F_0F0F).wrapping_mul(L8);

    // Find which byte contains the n-th bit
    let b = (le8(s, n.wrapping_mul(L8)) >> 7).wrapping_mul(L8) >> 53 & !7;

    // Rank within that byte
    let l = n - ((s << 8).wrapping_shr(b as u32) & 0xFF);

    // Select within the byte
    s = (u_nz8((x.wrapping_shr(b as u32) & 0xFF).wrapping_mul(L8) & 0x8040_2010_0804_0201) >> 7)
        .wrapping_mul(L8);

    let result = b + ((le8(s, l.wrapping_mul(L8)) >> 7).wrapping_mul(L8) >> 56);
    if result == 72 {
        None
    } else {
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select() {
        fn naive_select(x: u64, n: u64) -> Option<u64> {
            let mut count = 0u64;
            for i in 0..64 {
                if (x >> i) & 1 == 1 {
                    if count == n {
                        return Some(i);
                    }
                    count += 1;
                }
            }
            None
        }

        // Test patterns: zero, all bits, alternating, sparse, single bits at each position
        let mut test_values = vec![
            0u64,
            u64::MAX,
            0x5555_5555_5555_5555,
            0xAAAA_AAAA_AAAA_AAAA,
            0xFF00_FF00_FF00_FF00,
            0x8000_0000_0000_0001,
            0x0123_4567_89AB_CDEF,
        ];
        for i in 0..64 {
            test_values.push(1u64 << i);
        }

        for x in test_values {
            for n in 0..=64 {
                assert_eq!(select(x, n), naive_select(x, n), "select({x:#x}, {n})");
            }
        }
    }
}
