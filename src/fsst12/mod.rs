//! 12-bit FSST compression.
//!
//! An alternative version of FSST that uses 12-bit codes, allowing up to 4096 symbols
//! (of max 8 bytes long). The first 256 codes are identity codes for each byte value,
//! eliminating the need for an escape mechanism. Codes 256-4095 represent multi-byte
//! symbols discovered during training.
//!
//! Two 12-bit codes are packed into 3 bytes of output. This means each code costs 1.5
//! bytes on average. FSST12 needs longer symbols than 8-bit FSST to achieve the same
//! compression ratio, but works better on diverse data like JSON and URLs.

use crate::{Symbol, advance_8byte_word, compare_masked};
use lossy_pht::LossyPHT;
use std::fmt::{Debug, Formatter};
use std::mem::MaybeUninit;

mod builder;
mod lossy_pht;

pub use builder::*;

/// Number of bits per code in 12-bit FSST.
pub const FSST12_CODE_BITS: usize = 12;

/// Bit position where the length field starts in [`Code12`].
const FSST12_LEN_BITS: usize = 12;

/// Maximum number of codes (2^12 = 4096).
pub const FSST12_CODE_MAX: u16 = 1 << FSST12_CODE_BITS;

/// Bitmask for extracting a 12-bit code value.
pub const FSST12_CODE_MASK: u16 = FSST12_CODE_MAX - 1;

/// First code assigned to multi-byte symbols. Codes 0-255 are identity (single-byte).
pub const FSST12_CODE_BASE: u16 = 256;

/// Maximum number of multi-byte symbols in a 12-bit symbol table.
pub const FSST12_MAX_SYMBOLS: u16 = FSST12_CODE_MAX - FSST12_CODE_BASE;

/// A packed type containing a 12-bit code value and a 4-bit symbol length.
///
/// Bits 0-11: code (0-4095).
/// Bits 12-15: symbol length (1-8).
///
/// Codes 0-255 are identity codes (byte X maps to code X).
/// Codes 256-4095 are multi-byte symbols discovered during training.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct Code12(u16);

#[allow(clippy::len_without_is_empty)]
impl Code12 {
    /// Sentinel for an unused slot. Uses length=15 which is never valid (max valid is 8).
    const UNUSED: Self = Code12(0xFFFF);

    /// Create a code for a finalized multi-byte symbol.
    fn new_symbol(code: u16, len: usize) -> Self {
        debug_assert!(code < FSST12_CODE_MAX);
        debug_assert!((1..=8).contains(&len));
        Self(code | ((len as u16) << FSST12_LEN_BITS))
    }

    /// Create a code during the building phase. The builder-local index is remapped
    /// to the range [256, 4095] by adding [`FSST12_CODE_BASE`].
    fn new_symbol_building(code: u16, len: usize) -> Self {
        debug_assert!(code < FSST12_MAX_SYMBOLS);
        debug_assert!((1..=8).contains(&len));
        Self((code + FSST12_CODE_BASE) | ((len as u16) << FSST12_LEN_BITS))
    }

    /// Create an identity code for a raw byte.
    fn new_identity(byte: u8) -> Self {
        Self(byte as u16 | (1 << FSST12_LEN_BITS))
    }

    /// Extract the 12-bit code value.
    #[inline]
    fn code(self) -> u16 {
        self.0 & FSST12_CODE_MASK
    }

    /// Extract the builder-local index (code - CODE_BASE).
    /// Only valid for codes >= CODE_BASE.
    #[inline]
    fn builder_index(self) -> u16 {
        debug_assert!(self.code() >= FSST12_CODE_BASE);
        self.code() - FSST12_CODE_BASE
    }

    /// Extract the symbol length (1-8).
    #[inline]
    fn len(self) -> u16 {
        self.0 >> FSST12_LEN_BITS
    }
}

impl Debug for Code12 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Code12")
            .field("code", &self.code())
            .field("is_identity", &(self.code() < FSST12_CODE_BASE))
            .field("len", &self.len())
            .finish()
    }
}

/// Decompressor for 12-bit FSST compressed data.
///
/// Reads pairs of 12-bit codes packed into 3 bytes and expands them using
/// the symbol table.
#[derive(Clone)]
pub struct Decompressor12<'a> {
    /// Slice mapping codes to symbols. Includes both identity (0-255) and multi-byte codes.
    symbols: &'a [Symbol],

    /// Slice containing the length of each symbol.
    lengths: &'a [u8],
}

impl<'a> Decompressor12<'a> {
    /// Returns a new decompressor that uses the provided symbol table.
    ///
    /// The symbol and length slices must include identity codes at indices 0-255,
    /// followed by multi-byte symbol codes at indices 256+.
    ///
    /// # Panics
    ///
    /// If the provided symbol table has length greater than 4096.
    pub fn new(symbols: &'a [Symbol], lengths: &'a [u8]) -> Self {
        assert!(
            symbols.len() <= FSST12_CODE_MAX as usize,
            "12-bit symbol table cannot have size exceeding 4096"
        );
        assert_eq!(
            symbols.len(),
            lengths.len(),
            "symbols and lengths must have the same length"
        );

        Self { symbols, lengths }
    }

    /// Returns an upper bound on the size of the decompressed data.
    pub fn max_decompression_capacity(&self, compressed: &[u8]) -> usize {
        // Each 3 bytes → 2 codes → up to 16 bytes of output.
        // Each 2 bytes → 1 code → up to 8 bytes of output.
        let n_pairs = compressed.len() / 3;
        let has_tail = (compressed.len() % 3 == 2) as usize;
        let max_codes = n_pairs * 2 + has_tail;
        max_codes * size_of::<Symbol>()
    }

    /// Decompress a slice of 12-bit packed codes into a provided buffer.
    ///
    /// ## Panics
    ///
    /// If the compressed input has an invalid length (length % 3 == 1 and length > 0),
    /// or if the caller fails to provide sufficient capacity in the decoded buffer.
    pub fn decompress_into(&self, compressed: &[u8], decoded: &mut [MaybeUninit<u8>]) -> usize {
        assert!(
            compressed.is_empty() || compressed.len() % 3 != 1,
            "invalid 12-bit compressed length: {} (length % 3 must not be 1)",
            compressed.len()
        );

        if compressed.is_empty() {
            return 0;
        }

        unsafe {
            let mut in_ptr = compressed.as_ptr();
            let in_end = in_ptr.add(compressed.len());

            let mut out_ptr: *mut u8 = decoded.as_mut_ptr().cast();
            let out_begin = out_ptr.cast_const();
            let out_end = decoded.as_ptr().add(decoded.len()).cast::<u8>();

            macro_rules! store_symbol {
                ($code:expr) => {{
                    out_ptr
                        .cast::<u64>()
                        .write_unaligned(self.symbols.get_unchecked($code as usize).to_u64());
                    out_ptr = out_ptr.add(*self.lengths.get_unchecked($code as usize) as usize);
                }};
            }

            // Fast path: read u32, extract 2 codes from 3 bytes, advance by 3.
            // We need 4 bytes readable for the u32 load, and 2*8 bytes of output headroom.
            if decoded.len() >= 2 * size_of::<Symbol>() && compressed.len() >= 4 {
                let block_out_end = out_end.sub(2 * size_of::<Symbol>());
                let block_in_end = in_end.sub(4);

                while out_ptr.cast_const() <= block_out_end && in_ptr <= block_in_end {
                    let val = in_ptr.cast::<u32>().read_unaligned();
                    let code0 = val & 0xFFF;
                    let code1 = (val >> 12) & 0xFFF;

                    store_symbol!(code0);
                    store_symbol!(code1);

                    in_ptr = in_ptr.add(3);
                }
            }

            // Handle tail: 0, 2, or 3 remaining bytes.
            let bytes_left = in_end.offset_from(in_ptr) as usize;
            if bytes_left == 2 {
                // One trailing code in 2 bytes.
                let val = in_ptr.cast::<u16>().read_unaligned() as u32;
                let code = val & 0xFFF;
                let len = *self.lengths.get_unchecked(code as usize) as usize;
                assert!(
                    out_end.offset_from(out_ptr) >= len as isize,
                    "output buffer sized too small"
                );
                let sym = self.symbols.get_unchecked(code as usize).to_u64();
                let sym_bytes = sym.to_le_bytes();
                std::ptr::copy_nonoverlapping(sym_bytes.as_ptr(), out_ptr, len);
                out_ptr = out_ptr.add(len);
            } else if bytes_left == 3 {
                // Two trailing codes in 3 bytes. Read as u16 + u8 to avoid out-of-bounds.
                let lo = in_ptr.cast::<u16>().read_unaligned() as u32;
                let hi = in_ptr.add(2).read() as u32;
                let val = lo | (hi << 16);
                let code0 = val & 0xFFF;
                let code1 = (val >> 12) & 0xFFF;

                let len0 = *self.lengths.get_unchecked(code0 as usize) as usize;
                let len1 = *self.lengths.get_unchecked(code1 as usize) as usize;
                assert!(
                    out_end.offset_from(out_ptr) >= (len0 + len1) as isize,
                    "output buffer sized too small"
                );

                let sym0 = self
                    .symbols
                    .get_unchecked(code0 as usize)
                    .to_u64()
                    .to_le_bytes();
                std::ptr::copy_nonoverlapping(sym0.as_ptr(), out_ptr, len0);
                out_ptr = out_ptr.add(len0);

                let sym1 = self
                    .symbols
                    .get_unchecked(code1 as usize)
                    .to_u64()
                    .to_le_bytes();
                std::ptr::copy_nonoverlapping(sym1.as_ptr(), out_ptr, len1);
                out_ptr = out_ptr.add(len1);
            } else {
                assert_eq!(bytes_left, 0, "decompression should exhaust input");
            }

            out_ptr.offset_from(out_begin) as usize
        }
    }

    /// Decompress 12-bit FSST compressed data into a new vector.
    pub fn decompress(&self, compressed: &[u8]) -> Vec<u8> {
        let cap = self.max_decompression_capacity(compressed) + 7;
        let mut decoded = Vec::with_capacity(cap);

        let len = self.decompress_into(compressed, decoded.spare_capacity_mut());
        // SAFETY: len bytes have been initialized by the decompressor.
        unsafe { decoded.set_len(len) };
        decoded
    }
}

/// A compressor that uses a 12-bit symbol table to greedily compress strings.
///
/// FSST12 uses 4096 codes (12-bit) packed in pairs of 3 bytes. The first 256 codes
/// are identity mappings for raw bytes, so no escape mechanism is needed.
///
/// Create a compressor by [training][`Self::train`] on a corpus, or by
/// [rebuilding][`Self::rebuild_from`] from an existing symbol table.
///
/// ```
/// use fsst::fsst12::Compressor12;
///
/// let sample = "the quick brown fox jumped over the lazy dog";
/// let trained = Compressor12::train(&vec![sample.as_bytes()]);
/// let compressed = trained.compress(sample.as_bytes());
/// let decompressed = trained.decompressor().decompress(&compressed);
/// assert_eq!(decompressed, sample.as_bytes());
/// ```
#[derive(Clone)]
pub struct Compressor12 {
    /// Table mapping codes to symbols.
    /// Indices 0-255 are identity, 256+ are multi-byte.
    pub(crate) symbols: Vec<Symbol>,

    /// Length of each symbol (1 for identity, 2-8 for multi-byte).
    pub(crate) lengths: Vec<u8>,

    /// The number of multi-byte symbols (codes 256..256+n_symbols).
    pub(crate) n_symbols: u16,

    /// Inverted index mapping 2-byte inputs to codes.
    codes_two_byte: Vec<Code12>,

    /// Code threshold for the suffix optimization.
    /// Two-byte codes below this value have no longer suffix in the table.
    has_suffix_code: u16,

    /// Lossy perfect hash table for symbols of 3+ bytes.
    lossy_pht: LossyPHT,
}

impl Compressor12 {
    /// Compress one 8-byte word, returning (code, advance_in).
    ///
    /// # Safety
    ///
    /// The `word` must contain at least the remaining bytes of input (zero-padded).
    #[inline]
    unsafe fn compress_word(&self, word: u64) -> (u16, usize) {
        // Check the two-byte table first.
        // SAFETY: codes_two_byte has exactly 65536 entries and `word as u16` is always in [0, 65535].
        let code_twobyte = unsafe { *self.codes_two_byte.get_unchecked(word as u16 as usize) };

        // Fast path: 2-byte symbol without a longer suffix.
        // The extra >= CODE_BASE check prevents identity codes from short-circuiting
        // the hash table probe (identity codes are always < 256).
        if code_twobyte.code() >= FSST12_CODE_BASE && code_twobyte.code() < self.has_suffix_code {
            return (code_twobyte.code(), 2);
        }

        // Probe the hash table for a 3+ byte match.
        let entry = self.lossy_pht.lookup(word);
        let ignored_bits = entry.ignored_bits;
        if entry.code != Code12::UNUSED && compare_masked(word, entry.symbol.to_u64(), ignored_bits)
        {
            return (entry.code.code(), entry.code.len() as usize);
        }

        // Fall back to the two-byte table result (2-byte symbol or identity).
        (code_twobyte.code(), code_twobyte.len() as usize)
    }

    /// Compress a string, writing its result into a target buffer.
    ///
    /// Two 12-bit codes are packed into every 3 output bytes.
    /// If there is an odd number of codes, the final code occupies 2 bytes.
    ///
    /// # Safety
    ///
    /// The caller must ensure `values` has sufficient capacity. An upper bound is
    /// `plaintext.len() * 3 / 2 + 8`.
    pub unsafe fn compress_into(&self, plaintext: &[u8], values: &mut Vec<u8>) {
        if plaintext.is_empty() {
            return;
        }

        let mut in_ptr = plaintext.as_ptr();
        // SAFETY: `in_end` points just past the end of the plaintext slice.
        let in_end = unsafe { in_ptr.byte_add(plaintext.len()) };
        let in_end_sub8 = in_end as usize - 8;

        let mut out_ptr = values.as_mut_ptr();
        let out_end = unsafe { out_ptr.byte_add(values.capacity()) };

        // We accumulate codes in pairs: pending holds the first of a pair.
        let mut pending: u32 = 0;
        let mut have_pending = false;

        macro_rules! emit_code {
            ($code:expr) => {
                if have_pending {
                    // Pack two 12-bit codes into 3 bytes (little-endian).
                    let packed = pending | (($code as u32) << 12);
                    // SAFETY: caller ensures capacity; writes 4 bytes but only advances 3.
                    // The 4th byte is overwritten by the next pair or is within capacity slack.
                    #[allow(unused_unsafe)]
                    unsafe {
                        out_ptr.cast::<u32>().write_unaligned(packed);
                        out_ptr = out_ptr.add(3);
                    }
                    have_pending = false;
                } else {
                    pending = $code as u32;
                    have_pending = true;
                }
            };
        }

        // Main loop: process 8-byte words. Need 4 bytes of output space for the u32 write.
        while (in_ptr as usize) <= in_end_sub8 && unsafe { out_end.offset_from(out_ptr) } >= 4 {
            unsafe {
                let word: u64 = std::ptr::read_unaligned(in_ptr as *const u64);
                let (code, advance_in) = self.compress_word(word);
                in_ptr = in_ptr.byte_add(advance_in);
                emit_code!(code);
            };
        }

        // Handle the final <8 bytes.
        let remaining_bytes = unsafe { in_end.byte_offset_from(in_ptr) };
        assert!(
            out_ptr < out_end || remaining_bytes == 0,
            "output buffer sized too small"
        );
        let remaining_bytes = remaining_bytes as usize;

        let mut bytes = [0u8; 8];
        unsafe { std::ptr::copy_nonoverlapping(in_ptr, bytes.as_mut_ptr(), remaining_bytes) };
        let mut last_word = u64::from_le_bytes(bytes);

        while in_ptr < in_end && unsafe { out_end.offset_from(out_ptr) } >= 4 {
            let (code, advance_in) = unsafe { self.compress_word(last_word) };
            unsafe {
                in_ptr = in_ptr.add(advance_in);
                out_ptr = out_ptr.add(0); // no-op for clarity
            }
            last_word = advance_8byte_word(last_word, advance_in);
            emit_code!(code);
        }

        // Flush the pending code (odd number of total codes).
        if have_pending {
            assert!(
                unsafe { out_end.offset_from(out_ptr) } >= 2,
                "output buffer sized too small for trailing code"
            );
            unsafe {
                out_ptr.cast::<u16>().write_unaligned(pending as u16);
                out_ptr = out_ptr.add(2);
            }
        }

        assert!(
            in_ptr >= in_end,
            "exhausted output buffer before exhausting input"
        );
        assert!(out_ptr <= out_end, "output buffer sized too small");

        let bytes_written = unsafe { out_ptr.offset_from(values.as_ptr()) };
        assert!(bytes_written >= 0, "out_ptr ended before it started");
        unsafe { values.set_len(bytes_written as usize) };
    }

    /// Compress a plaintext string using the 12-bit symbol table.
    pub fn compress(&self, plaintext: &[u8]) -> Vec<u8> {
        if plaintext.is_empty() {
            return Vec::new();
        }

        // Worst case: every byte is an identity code → n codes → ceil(3n/2) bytes.
        let mut buffer = Vec::with_capacity(plaintext.len() * 3 / 2 + 8);

        // SAFETY: we over-allocate the buffer.
        unsafe { self.compress_into(plaintext, &mut buffer) };

        buffer
    }

    /// Compress many lines in bulk.
    pub fn compress_bulk(&self, lines: &Vec<&[u8]>) -> Vec<Vec<u8>> {
        let mut res = Vec::new();
        for line in lines {
            res.push(self.compress(line));
        }
        res
    }

    /// Access the decompressor for data compressed with this symbol table.
    pub fn decompressor(&self) -> Decompressor12<'_> {
        Decompressor12::new(&self.symbols, &self.lengths)
    }

    /// Returns a readonly slice of the multi-byte symbols (codes 256+).
    ///
    /// This does NOT include the 256 identity symbols. The returned slice
    /// has length [`n_symbols`][Self::n_symbols].
    pub fn symbol_table(&self) -> &[Symbol] {
        let base = FSST12_CODE_BASE as usize;
        &self.symbols[base..base + self.n_symbols as usize]
    }

    /// Returns a readonly slice of lengths for the multi-byte symbols.
    ///
    /// Values range from 2-8.
    pub fn symbol_lengths(&self) -> &[u8] {
        let base = FSST12_CODE_BASE as usize;
        &self.lengths[base..base + self.n_symbols as usize]
    }

    /// The number of multi-byte symbols in this compressor.
    pub fn n_symbols(&self) -> u16 {
        self.n_symbols
    }

    /// Rebuild a compressor from an existing multi-byte symbol table.
    ///
    /// The provided symbols and lengths should be only the multi-byte symbols (not
    /// including the 256 identity codes). This will not re-optimize the table.
    pub fn rebuild_from(symbols: impl AsRef<[Symbol]>, symbol_lens: impl AsRef<[u8]>) -> Self {
        let symbols = symbols.as_ref();
        let symbol_lens = symbol_lens.as_ref();

        assert_eq!(
            symbols.len(),
            symbol_lens.len(),
            "symbols and lengths differ"
        );
        assert!(
            symbols.len() <= FSST12_MAX_SYMBOLS as usize,
            "symbol table len must be <= {}, was {}",
            FSST12_MAX_SYMBOLS,
            symbols.len()
        );
        validate_symbol_order_12(symbol_lens);

        let n_symbols = symbols.len() as u16;

        // Build the full symbol/length arrays: 256 identity + multi-byte.
        let total_codes = FSST12_CODE_BASE as usize + symbols.len();
        let mut full_symbols = Vec::with_capacity(total_codes);
        let mut full_lengths = Vec::with_capacity(total_codes);
        for byte in 0..=255u8 {
            full_symbols.push(Symbol::from_u8(byte));
            full_lengths.push(1u8);
        }
        full_symbols.extend_from_slice(symbols);
        full_lengths.extend_from_slice(symbol_lens);

        // Build the lossy PHT for 3+ byte symbols.
        // Pass the builder-local index (not the full code) because insert()
        // adds CODE_BASE internally via new_symbol_building().
        let mut lossy_pht = LossyPHT::new();
        for (idx, (&symbol, &len)) in symbols.iter().zip(symbol_lens.iter()).enumerate() {
            if len >= 3 {
                assert!(
                    lossy_pht.insert(symbol, len as usize, idx as u16),
                    "rebuild symbol insertion into PHT must succeed"
                );
            }
        }

        // Build codes_two_byte: identity fallback for all entries, then overwrite
        // with 2-byte symbol matches.
        let mut codes_two_byte = Vec::with_capacity(65_536);
        for idx in 0..65_536u32 {
            codes_two_byte.push(Code12::new_identity(idx as u8));
        }
        for (idx, (&symbol, &len)) in symbols.iter().zip(symbol_lens.iter()).enumerate() {
            if len == 2 {
                let code = FSST12_CODE_BASE + idx as u16;
                codes_two_byte[symbol.first2() as usize] = Code12::new_symbol(code, 2);
            }
        }

        // Compute has_suffix_code: first 2-byte code that has a 3+ byte suffix.
        let mut has_suffix_code = FSST12_CODE_BASE;
        for (idx, (&symbol, &len)) in symbols.iter().zip(symbol_lens.iter()).enumerate() {
            if len != 2 {
                break;
            }
            let code = FSST12_CODE_BASE + idx as u16;
            let rest = &symbols[idx..];
            if rest
                .iter()
                .any(|&other| other.len() > 2 && symbol.first2() == other.first2())
            {
                has_suffix_code = code;
                break;
            }
            has_suffix_code = code + 1;
        }

        Compressor12 {
            n_symbols,
            symbols: full_symbols,
            lengths: full_lengths,
            codes_two_byte,
            lossy_pht,
            has_suffix_code,
        }
    }
}

/// Validate that multi-byte symbol lengths are ordered correctly (non-decreasing, 2-8).
fn validate_symbol_order_12(symbol_lens: &[u8]) {
    let mut expected = 2u8;
    for (idx, &len) in symbol_lens.iter().enumerate() {
        assert!(
            (2..=8).contains(&len),
            "12-bit symbol code={idx} must be 2-8 bytes, was {len}"
        );
        assert!(
            len >= expected,
            "12-bit symbol code={idx} violates ordering (expected >= {expected}, got {len})"
        );
        expected = len;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_identity_roundtrip() {
        // Empty compressor: all codes are identity. Each byte → 1 code → 1.5 bytes.
        let compressor = CompressorBuilder12::new().build();
        let decompressor = compressor.decompressor();

        let input = b"hello world!";
        let compressed = compressor.compress(input);
        let decompressed = decompressor.decompress(&compressed);
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_empty_input() {
        let compressor = CompressorBuilder12::new().build();
        let compressed = compressor.compress(b"");
        assert!(compressed.is_empty());
        let decompressed = compressor.decompressor().decompress(&compressed);
        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_single_byte() {
        let compressor = CompressorBuilder12::new().build();
        let compressed = compressor.compress(&[42]);
        // 1 code → 2 bytes (trailing odd code)
        assert_eq!(compressed.len(), 2);
        let decompressed = compressor.decompressor().decompress(&compressed);
        assert_eq!(decompressed, &[42]);
    }

    #[test]
    fn test_two_bytes() {
        let compressor = CompressorBuilder12::new().build();
        let compressed = compressor.compress(&[1, 2]);
        // 2 codes → 3 bytes
        assert_eq!(compressed.len(), 3);
        let decompressed = compressor.decompressor().decompress(&compressed);
        assert_eq!(decompressed, &[1, 2]);
    }

    #[test]
    fn test_code12_basics() {
        let identity = Code12::new_identity(42);
        assert_eq!(identity.code(), 42);
        assert_eq!(identity.len(), 1);

        let symbol = Code12::new_symbol(300, 4);
        assert_eq!(symbol.code(), 300);
        assert_eq!(symbol.len(), 4);

        let building = Code12::new_symbol_building(10, 3);
        assert_eq!(building.code(), 10 + FSST12_CODE_BASE);
        assert_eq!(building.len(), 3);
        assert_eq!(building.builder_index(), 10);

        assert_eq!(Code12::UNUSED.len(), 15);
    }
}
