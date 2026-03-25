use std::fmt::Debug;

use super::Code12;
use crate::Symbol;
use crate::builder::fsst_hash;

/// Size of the perfect hash table for 12-bit FSST.
///
/// Larger than the 8-bit variant to accommodate up to 3840 symbols.
pub const HASH_TABLE_SIZE: usize = 1 << 14;

/// A single entry in the [`LossyPHT`].
#[derive(Clone, Debug)]
#[repr(C)]
pub(crate) struct TableEntry {
    /// Symbol, piece of a string, 8 bytes or fewer.
    pub(crate) symbol: Symbol,

    /// Code and associated metadata for the symbol.
    pub(crate) code: Code12,

    /// Number of ignored bits in `symbol`.
    ///
    /// This is equivalent to `64 - 8 * code.len()` but is pre-computed to save a few instructions in
    /// the compression loop.
    pub(crate) ignored_bits: u16,
}

assert_sizeof!(TableEntry => 16);

impl TableEntry {
    pub(crate) fn is_unused(&self) -> bool {
        self.code == Code12::UNUSED
    }
}

/// Lossy Perfect Hash Table for 12-bit FSST compression.
///
/// Same concept as the 8-bit variant: insertion may fail if a slot is already occupied.
/// Higher-gain symbols should be inserted first.
#[derive(Clone, Debug)]
pub(crate) struct LossyPHT {
    slots: Vec<TableEntry>,
}

impl LossyPHT {
    /// Construct a new empty lossy perfect hash table.
    pub(crate) fn new() -> Self {
        let slots = vec![
            TableEntry {
                symbol: Symbol::ZERO,
                code: Code12::UNUSED,
                ignored_bits: 64,
            };
            HASH_TABLE_SIZE
        ];

        Self { slots }
    }

    /// Try and insert the (symbol, code) pair into the table.
    ///
    /// Returns true if inserted, false if rejected due to collision.
    pub(crate) fn insert(&mut self, symbol: Symbol, len: usize, code: u16) -> bool {
        let prefix_3bytes = symbol.to_u64() & 0xFF_FF_FF;
        let slot = fsst_hash(prefix_3bytes) as usize & (HASH_TABLE_SIZE - 1);
        let entry = &mut self.slots[slot];
        if !entry.is_unused() {
            false
        } else {
            entry.symbol = symbol;
            entry.code = Code12::new_symbol_building(code, len);
            entry.ignored_bits = (64 - 8 * symbol.len()) as u16;
            true
        }
    }

    /// Given a new code mapping, rewrite the codes into the new code range.
    pub(crate) fn renumber(&mut self, new_codes: &[u16]) {
        for slot in self.slots.iter_mut() {
            if slot.code != Code12::UNUSED {
                let old_code = slot.code.builder_index();
                let new_code = new_codes[old_code as usize];
                let len = slot.code.len();
                slot.code = Code12::new_symbol(new_code, len as usize);
            }
        }
    }

    /// Remove the symbol from the hashtable, if it exists.
    pub(crate) fn remove(&mut self, symbol: Symbol) {
        let prefix_3bytes = symbol.to_u64() & 0xFF_FF_FF;
        let slot = fsst_hash(prefix_3bytes) as usize & (HASH_TABLE_SIZE - 1);
        self.slots[slot].code = Code12::UNUSED;
    }

    #[inline]
    pub(crate) fn lookup(&self, word: u64) -> &TableEntry {
        let prefix_3bytes = word & 0xFF_FF_FF;
        let slot = fsst_hash(prefix_3bytes) as usize & (HASH_TABLE_SIZE - 1);

        // SAFETY: the slot is guaranteed to be between [0, HASH_TABLE_SIZE).
        unsafe { self.slots.get_unchecked(slot) }
    }
}

impl Default for LossyPHT {
    fn default() -> Self {
        Self::new()
    }
}
