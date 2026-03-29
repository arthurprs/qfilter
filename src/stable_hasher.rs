use std::hash::{BuildHasher, Hasher};

/// Wrapper over a hasher that provides stable output across platforms.
///
/// The architecture dependent `isize` and `usize` types are extended to
/// 64 bits if needed. The `portable` feature of foldhash-portable handles
/// endianness normalization internally.
pub struct StableHasher {
    state: foldhash_portable::quality::FoldHasher<'static>,
}

impl Default for StableHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl StableHasher {
    #[inline]
    pub fn new() -> Self {
        Self {
            state: foldhash_portable::quality::FixedState::with_seed(0).build_hasher(),
        }
    }
}

impl Hasher for StableHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.state.finish()
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.state.write(bytes);
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.state.write_u8(i);
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.state.write_u16(i);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.state.write_u32(i);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.state.write_u64(i);
    }

    #[inline]
    fn write_u128(&mut self, i: u128) {
        self.state.write_u128(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        // Always treat usize as u64 so we get the same results on 32 and 64 bit
        // platforms.
        self.state.write_u64(i as u64);
    }

    #[inline]
    fn write_i8(&mut self, i: i8) {
        self.state.write_i8(i);
    }

    #[inline]
    fn write_i16(&mut self, i: i16) {
        self.state.write_i16(i);
    }

    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.state.write_i32(i);
    }

    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.state.write_i64(i);
    }

    #[inline]
    fn write_i128(&mut self, i: i128) {
        self.state.write_i128(i);
    }

    #[inline]
    fn write_isize(&mut self, i: isize) {
        // Always treat isize as i64 so we get the same results on 32 and 64 bit
        // platforms.
        self.state.write_i64(i as i64);
    }
}

/// A [`BuildHasher`] that produces `StableHasher` instances with a fixed seed.
///
/// This is a zero-sized type, so storing it in a struct adds no overhead.
/// Uses foldhash-portable for high performance and cross-platform stability.
///
/// This is the default hasher for [`crate::Filter`]. It uses a fixed seed,
/// so it does **not** provide DoS resistance. Use a custom [`BuildHasher`]
/// (e.g. `foldhash_portable::quality::RandomState`) via [`crate::Filter::new_with_hasher()`]
/// or [`crate::Filter::with_hasher()`] for DoS resistance.
#[derive(Clone, Copy, Default)]
pub struct StableBuildHasher;

impl BuildHasher for StableBuildHasher {
    type Hasher = StableHasher;

    #[inline]
    fn build_hasher(&self) -> StableHasher {
        StableHasher::new()
    }
}
