//! Lossy perfect hash table for the FSST12 codec.
//!
//! Mirrors [`crate::lossy_pht`]: fixed-size, open-addressed, keyed by `fsst_hash` of the
//! symbol's first 3 bytes. Stores 3+ byte symbols only.

use crate::Symbol;
use crate::builder::fsst_hash;

/// 2048 × 16-byte entries = 32 KB. Matches the classic FSST PHT size. The full working
/// set during compression (this table plus the 128 KB two-byte index plus up to 36 KB of
/// symbol and length tables) does not fit in 32 KB L1d on common hardware; this size is
/// inherited from the classic codec for parity rather than chosen by analysis here.
pub(crate) const FSST12_PHT_SIZE: usize = 1 << 11;

/// Code 0 is the identity code for byte `0x00`, which is length 1 and therefore never
/// stored in the PHT, so it is safe to use as the empty-slot sentinel.
const PHT_UNUSED: u16 = 0;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(crate) struct TableEntry12 {
    pub(crate) symbol: Symbol,
    /// [`PHT_UNUSED`] marks an empty slot.
    pub(crate) code: u16,
    /// `64 - 8 * len`, pre-computed for the compression loop's mask.
    pub(crate) ignored_bits: u16,
}

assert_sizeof!(TableEntry12 => 16);

impl TableEntry12 {
    #[inline]
    pub(crate) fn is_unused(&self) -> bool {
        self.code == PHT_UNUSED
    }
}

/// Insertions silently drop on collision, so callers must insert in decreasing-gain
/// order.
#[derive(Clone, Debug)]
pub(crate) struct LossyPht12 {
    slots: Vec<TableEntry12>,
}

impl LossyPht12 {
    pub(crate) fn new() -> Self {
        Self {
            slots: vec![
                TableEntry12 {
                    symbol: Symbol::ZERO,
                    code: PHT_UNUSED,
                    ignored_bits: 64,
                };
                FSST12_PHT_SIZE
            ],
        }
    }

    /// Returns `false` if the slot was already occupied (existing entry kept).
    pub(crate) fn insert(&mut self, symbol: Symbol, len: u8, code: u16) -> bool {
        let prefix_3bytes = symbol.to_u64() & 0xFF_FF_FF;
        let slot = fsst_hash(prefix_3bytes) as usize & (FSST12_PHT_SIZE - 1);
        let entry = &mut self.slots[slot];
        if !entry.is_unused() {
            return false;
        }
        entry.symbol = symbol;
        entry.code = code;
        entry.ignored_bits = 64 - 8 * len as u16;
        true
    }

    /// Returns the slot keyed by the first 3 bytes of `word`. Callers must verify the
    /// result via [`TableEntry12::is_unused`] and a masked comparison against
    /// `entry.symbol`.
    #[inline]
    pub(crate) fn lookup(&self, word: u64) -> TableEntry12 {
        let prefix_3bytes = word & 0xFF_FF_FF;
        let slot = fsst_hash(prefix_3bytes) as usize & (FSST12_PHT_SIZE - 1);
        // SAFETY: slot is masked into [0, FSST12_PHT_SIZE).
        unsafe { *self.slots.get_unchecked(slot) }
    }

    pub(crate) fn remove(&mut self, symbol: Symbol) {
        let prefix_3bytes = symbol.to_u64() & 0xFF_FF_FF;
        let slot = fsst_hash(prefix_3bytes) as usize & (FSST12_PHT_SIZE - 1);
        self.slots[slot].code = PHT_UNUSED;
    }
}
