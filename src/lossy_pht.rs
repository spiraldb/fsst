use std::fmt::Debug;
use std::fmt::Formatter;
use std::u16;

use crate::Symbol;

/// Size of the perfect hash table.
///
/// NOTE: this differs from the paper, which recommends a 64KB total
/// table size. The paper does not account for the fact that most
/// vendors split the L1 cache into 32KB of instruction and 32KB of data.
pub const HASH_TABLE_SIZE: usize = 1 << 11;

/// Bit-packed metadata for a [`TableEntry`]
///
/// Bitpacked layout:
///
/// bits 9-15: ignored bits in the symbol. Equivalent to 64 - symbol.len()*8
/// bit 8: the "unused" flag
/// bits 0-7: code value (0-254)
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct PackedMeta(u16);

assert_sizeof!(PackedMeta => 2);

impl PackedMeta {
    /// Constant unused instance.
    ///
    /// All bits are set, corresponding to
    ///
    /// 6 bits set for `ignored bits`
    /// 1 unused bit
    /// 1 bit to indicate the `unused` flag
    /// 8 bits of `code` data
    pub const UNUSED: Self = Self(0b10000001_11111111);

    /// The 8th bit toggles if the slot is unused or not.
    const UNUSED_FLAG: u16 = 1 << 8;

    /// Create a new `PackedSymbolMeta` from raw parts.
    ///
    /// # Panics
    /// If `len` > 8 or `code` > [`Code::CODE_MAX`]
    pub fn new(len: u16, code: u8) -> Self {
        assert!(len <= 8, "cannot construct PackedCode with len > 8");

        let ignored_bits = 64 - 8 * len;

        let packed = (ignored_bits << 9) | (code as u16);
        Self(packed)
    }

    /// Import a `PackedSymbolMeta` from a raw `u16`.
    pub fn from_u16(value: u16) -> Self {
        assert!(
            (value >> 9) <= 64,
            "cannot construct PackedCode with len > 8"
        );

        Self(value)
    }

    /// Get the number of ignored bits in the corresponding symbol's `u64` representation.
    ///
    /// Always <= 64
    #[inline]
    pub(crate) fn ignored_bits(&self) -> u16 {
        (self.0 >> 9) as u16
    }

    /// Get the code value.
    #[inline]
    pub(crate) fn code(&self) -> u8 {
        self.0 as u8
    }

    /// Check if the unused flag is set
    #[inline]
    pub(crate) fn is_unused(&self) -> bool {
        (self.0 & Self::UNUSED_FLAG) != 0
    }
}

impl Default for PackedMeta {
    fn default() -> Self {
        // The default implementation of a `PackedMeta` is one where only the `UNUSED_FLAG` is set,
        // representing an unused slot in the table.
        Self::UNUSED
    }
}

impl Debug for PackedMeta {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedCode")
            .field("ignored_bits", &self.ignored_bits())
            .field("code", &self.code())
            .finish()
    }
}

/// A single entry in the [`SymbolTable`].
///
/// `TableEntry` is based on the `Symbol` class outlined in Algorithm 4 of the FSST paper. See
/// the module documentation for a link to the paper.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub(crate) struct TableEntry {
    /// Symbol, piece of a string, 8 bytes or fewer.
    pub(crate) symbol: Symbol,

    /// Bit-packed metadata for the entry.
    ///
    /// [`PackedMeta`] provides compact, efficient access to metadata about the `symbol`, including
    /// its code and length.
    pub(crate) packed_meta: PackedMeta,
}

assert_sizeof!(TableEntry => 16);

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
                packed_meta: PackedMeta::UNUSED,
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
        println!("\t\tinserting to slot {slot}");
        let entry = &mut self.slots[slot];

        if !entry.packed_meta.is_unused() {
            return false;
        } else {
            entry.symbol = symbol;
            entry.packed_meta = PackedMeta::new(symbol.len() as u16, code);
            return true;
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

#[cfg(test)]
mod test {
    use crate::lossy_pht::PackedMeta;

    #[test]
    fn test_packedmeta() {
        assert!(PackedMeta::UNUSED.is_unused());
        assert_eq!(PackedMeta::UNUSED.ignored_bits(), 64);
    }
}
