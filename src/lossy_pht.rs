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

/// A single entry in the [`SymbolTable`].
///
/// `TableEntry` is based on the `Symbol` class outlined in Algorithm 4 of the FSST paper. See
/// the module documentation for a link to the paper.
#[derive(Copy, Clone, Debug)]
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
        let mut slots = Vec::with_capacity(HASH_TABLE_SIZE);
        // Initialize all slots to empty entries
        for _ in 0..HASH_TABLE_SIZE {
            slots.push(TableEntry {
                symbol: Symbol::ZERO,
                code: CodeMeta::EMPTY,
                ignored_bits: 64,
            });
        }

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
        let entry = &mut self.slots[slot];

        if entry.is_unused() {
            false
        } else {
            entry.symbol = symbol;
            entry.code = CodeMeta::new_symbol(code, symbol);
            entry.ignored_bits = (64 - 8 * symbol.len()) as u16;
            true
        }
    }

    pub(crate) fn lookup(&self, word: u64) -> TableEntry {
        let prefix_3bytes = word & 0xFF_FF_FF;
        let slot = self.hash(prefix_3bytes) as usize & (HASH_TABLE_SIZE - 1);

        self.slots[slot]
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
