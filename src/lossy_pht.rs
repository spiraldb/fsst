use std::fmt::Debug;

use crate::CodeMeta;
use crate::Symbol;
use crate::MAX_CODE;

/// Size of the perfect hash table.
///
/// NOTE: this differs from the paper, which recommends a 64KB total
/// table size. The paper does not account for the fact that most
/// vendors split the L1 cache into 32KB of instruction and 32KB of data.
pub const HASH_TABLE_SIZE: usize = 1 << 11;

/// A single entry in the [Lossy Perfect Hash Table][`LossyPHT`].
///
/// `TableEntry` is based on the `Symbol` class outlined in Algorithm 4 of the FSST paper. See
/// the module documentation for a link to the paper.
#[derive(Clone, Debug)]
#[repr(C)]
pub(crate) struct TableEntry {
    /// Symbol, piece of a string, 8 bytes or fewer.
    pub(crate) symbol: Symbol,

    /// Code and associated metadata for the symbol
    pub(crate) code: CodeMeta,

    /// Number of ignored bits in `symbol`.
    ///
    /// This is equivalent to `64 - 8 * code.len()` but is pre-computed to save a few instructions in
    /// the compression loop.
    pub(crate) ignored_bits: u16,
}

assert_sizeof!(TableEntry => 16);

impl TableEntry {
    pub(crate) fn is_unused(&self) -> bool {
        // 511 should never come up for real, so use as the sentinel for an unused slot
        self.code.extended_code() == MAX_CODE
    }
}

/// Lossy Perfect Hash Table implementation for compression.
///
/// This implements the "Lossy Perfect Hash Table" described in Section 5 of the paper.
///
/// It is so-called because the `insert` operation for a symbol may fail, if another symbol is
/// already occupying the slot.
///
/// If insertions are made from highest-gain to lowest and from longest-symbol to shortest, then
/// we can say that any failed insert is not a big loss, because its slot is being held by a higher-gain
/// symbol. Note that because other code in this crate calls `insert` in the pop-order of a max heap,
/// this holds.
#[derive(Clone, Debug)]
pub(crate) struct LossyPHT {
    /// Hash table slots. Used for strings that are 3 bytes or more.
    slots: Vec<TableEntry>,
}

impl LossyPHT {
    /// Construct a new empty lossy perfect hash table
    pub(crate) fn new() -> Self {
        let slots = vec![
            TableEntry {
                symbol: Symbol::ZERO,
                code: CodeMeta::EMPTY,
                ignored_bits: 64,
            };
            HASH_TABLE_SIZE
        ];

        Self { slots }
    }

    /// Try and insert the (symbol, code) pair into the table.
    ///
    /// If there is a collision, we keep the current thing and reject the write.
    ///
    /// # Returns
    ///
    /// True if the symbol was inserted into the table, false if it was rejected due to collision.
    pub(crate) fn insert(&mut self, symbol: Symbol, code: u8) -> bool {
        let prefix_3bytes = symbol.as_u64() & 0xFF_FF_FF;
        let slot = self.hash(prefix_3bytes) as usize & (HASH_TABLE_SIZE - 1);

        unsafe {
            let entry = self.slots.as_mut_ptr().add(slot);
            if (*entry).code.extended_code() != MAX_CODE {
                // in-use
                false
            } else {
                // unused
                (*entry).symbol = symbol;
                (*entry).code = CodeMeta::new_symbol(code, symbol);
                (*entry).ignored_bits = (64 - 8 * symbol.len()) as u16;
                true
            }
        }

        // let entry = &mut self.slots[slot];
        // if !entry.is_unused() {
        //     false
        // } else {
        //     entry.symbol = symbol;
        //     entry.code = CodeMeta::new_symbol(code, symbol);
        //     entry.ignored_bits = (64 - 8 * symbol.len()) as u16;
        //     true
        // }
    }

    /// Remove the symbol from the hashtable, if it exists.
    pub(crate) fn remove(&mut self, symbol: Symbol) {
        let prefix_3bytes = symbol.as_u64() & 0xFF_FF_FF;
        let slot = self.hash(prefix_3bytes) as usize & (HASH_TABLE_SIZE - 1);
        self.slots[slot].code = CodeMeta::EMPTY;
    }

    #[inline]
    pub(crate) fn lookup(&self, word: u64) -> &TableEntry {
        let prefix_3bytes = word & 0xFF_FF_FF;
        let slot = self.hash(prefix_3bytes) as usize & (HASH_TABLE_SIZE - 1);

        // SAFETY: the slot is guaranteed to between 0...(HASH_TABLE_SIZE - 1).
        unsafe { self.slots.get_unchecked(slot) }
    }

    /// Hash a value to find the bucket it belongs in.
    ///
    /// The particular hash function comes from the code listing of Algorithm 4 of the FSST paper.
    #[inline]
    fn hash(&self, value: u64) -> u64 {
        (value * 2971215073) ^ (value >> 15)
    }
}

impl Default for LossyPHT {
    fn default() -> Self {
        Self::new()
    }
}
