//! FSST12: 12-bit-code variant of FSST.
//!
//! FSST12 is described in the [FastLanes File Format paper][fastlanes] and implemented in
//! reference form by [cwida/fsst][cwida]. It differs from classic 8-bit FSST in three ways:
//!
//! - Codes are 12 bits wide, so the symbol table can hold up to 4096 entries.
//! - The first 256 codes are reserved: code `i` always decodes to the single byte `i`.
//!   These slots are not chosen by training and are present in every FSST12 table.
//! - There is no escape mechanism. Because every byte already has a code, the compressor
//!   always finds a match, and the decoder never branches on an escape sentinel.
//!
//! Each emitted code occupies 12 bits, so the encoded stream is bit-packed at 1.5 bytes
//! per code. Single-byte fallbacks still cost more than the byte they encode (1.5× vs.
//! 1×), but the penalty is lighter than FSST8's 2× escape cost.
//!
//! [fastlanes]: https://www.vldb.org/pvldb/vol18/p4629-afroozeh.pdf
//! [cwida]: https://github.com/cwida/fsst

use crate::{Symbol, advance_8byte_word, compare_masked};
use std::mem::MaybeUninit;

mod builder;
mod lossy_pht;

pub use builder::*;

use lossy_pht::LossyPht12;

/// Number of bits per code in FSST12.
pub const FSST12_CODE_BITS: usize = 12;

/// Maximum number of entries in an FSST12 symbol table.
///
/// Equal to `2^12 = 4096`, including the [`FSST12_RESERVED_CODES`] reserved single-byte
/// slots.
pub const FSST12_MAX_SYMBOLS: usize = 1 << FSST12_CODE_BITS;

/// Number of code slots reserved for single-byte literal symbols.
///
/// Codes `0..256` always decode to the byte equal to their code value, regardless of any
/// training that took place. Training only contributes symbols at codes `256..4096`.
pub const FSST12_RESERVED_CODES: usize = 256;

/// Maximum number of learned (length 2-8) symbols a training run may add.
pub const FSST12_MAX_LEARNED: usize = FSST12_MAX_SYMBOLS - FSST12_RESERVED_CODES;

pub(crate) const FSST12_TWO_BYTE_TABLE_SIZE: usize = 1 << 16;

pub(crate) const CODE12_LEN_SHIFT: u32 = 12;
pub(crate) const CODE12_MASK: u16 = (1 << CODE12_LEN_SHIFT) - 1;

/// Bits `0..12` hold the code, bits `12..16` hold the length (1..=8).
#[inline]
pub(crate) fn pack_code12(code: u16, len: u8) -> u16 {
    debug_assert!(code <= CODE12_MASK);
    debug_assert!((1..=8).contains(&len));
    ((len as u16) << CODE12_LEN_SHIFT) | code
}

/// An FSST12 compressor.
///
/// Compresses arbitrary byte sequences into a bit-packed stream of 12-bit codes. The
/// first 256 codes always decode as the matching single byte, so every input has a code
/// and no escape mechanism is needed.
#[derive(Clone)]
pub struct Compressor12 {
    /// Entries 0..256 are reserved identity codes; entries 256.. are learned symbols of
    /// length 2..=8.
    pub(crate) symbols: Vec<Symbol>,
    pub(crate) lengths: Vec<u8>,
    /// Indexed by `u16` read of the input; each entry is `pack_code12(code, len)`. Every
    /// slot is at least the identity code for the low input byte; 2-byte learned symbols
    /// overwrite their respective slot.
    pub(crate) codes_two_byte: Vec<u16>,
    pub(crate) lossy_pht: LossyPht12,
}

impl Compressor12 {
    /// Compress a byte slice into a bit-packed 12-bit code stream.
    pub fn compress(&self, plaintext: &[u8]) -> Vec<u8> {
        if plaintext.is_empty() {
            return Vec::new();
        }
        // Worst case: every input byte resolves to a length-1 code, and the bit-packed
        // output is 1.5 bytes per code. Round up and add 1 for the odd-code 2-byte tail.
        let cap = (plaintext.len() * 3).div_ceil(2) + 1;
        let mut out: Vec<u8> = Vec::with_capacity(cap);
        // SAFETY: `cap` is at least the worst-case FSST12 output size.
        unsafe { self.compress_into(plaintext, &mut out) };
        out
    }

    /// Greedy compression into `out`. On return, `out` is truncated to the actual encoded
    /// length. FSST12 analogue of [`crate::Compressor::compress_into`].
    ///
    /// # Safety
    ///
    /// `out.capacity()` must be at least `ceil(plaintext.len() * 3 / 2) + 1`. The function
    /// writes raw bytes into the spare capacity and then calls `set_len`, so calling with
    /// undersized capacity is undefined behaviour.
    ///
    /// ```
    /// use fsst::fsst12::{Compressor12, CompressorBuilder12};
    /// use fsst::Symbol;
    ///
    /// let mut builder = CompressorBuilder12::new();
    /// assert!(builder.insert(Symbol::from_slice(b"abcdefgh"), 8));
    /// let compressor = builder.build();
    ///
    /// let plaintext: &[u8] = b"abcdefghabcdefgh";
    /// let mut buf = Vec::with_capacity(plaintext.len() * 3 / 2 + 2);
    /// // SAFETY: `buf` capacity exceeds the worst-case FSST12 output size.
    /// unsafe { compressor.compress_into(plaintext, &mut buf) };
    /// assert_eq!(compressor.decompressor().decompress(&buf), plaintext);
    /// ```
    pub unsafe fn compress_into(&self, plaintext: &[u8], out: &mut Vec<u8>) {
        // SAFETY: pointer arithmetic is guarded by either the fast-loop precondition
        // (`in_ptr + 8 <= in_end`) or by reading from a zero-padded `last_word` in the
        // tail. Output writes are bounded by the caller-asserted capacity.
        let written = unsafe {
            let in_begin = plaintext.as_ptr();
            let in_end = in_begin.add(plaintext.len());
            let mut in_ptr = in_begin;
            let out_begin = out.as_mut_ptr();
            let mut out_ptr = out_begin;

            // When `have_pending` is true, `pending` holds the previous code's 12 bits in
            // its low 12. The next code OR's into bits 12..24 to form a 3-byte triple.
            let mut pending: u32 = 0;
            let mut have_pending = false;

            macro_rules! emit_code {
                ($code:expr) => {{
                    let code = $code as u32;
                    if have_pending {
                        let packed = pending | (code << CODE12_LEN_SHIFT);
                        out_ptr.write(packed as u8);
                        out_ptr.add(1).write((packed >> 8) as u8);
                        out_ptr.add(2).write((packed >> 16) as u8);
                        out_ptr = out_ptr.add(3);
                        have_pending = false;
                    } else {
                        pending = code;
                        have_pending = true;
                    }
                }};
            }

            // ≥ 8 input bytes: unaligned u64 reads are safe, so the match-finder can
            // skip the `remaining` check.
            if plaintext.len() >= 8 {
                let in_end_sub8 = in_end.sub(8);
                while (in_ptr as usize) <= (in_end_sub8 as usize) {
                    let word = in_ptr.cast::<u64>().read_unaligned();
                    let (code, advance) = self.compress_word_full(word);
                    emit_code!(code);
                    in_ptr = in_ptr.add(advance);
                }
            }

            // < 8 bytes left: read into a zero-padded word so we never dereference past
            // `in_end`.
            let mut remaining = in_end.offset_from(in_ptr) as usize;
            if remaining > 0 {
                let mut bytes = [0u8; 8];
                std::ptr::copy_nonoverlapping(in_ptr, bytes.as_mut_ptr(), remaining);
                let mut last_word = u64::from_le_bytes(bytes);
                while remaining > 0 {
                    let (code, advance) = self.compress_word(last_word, remaining);
                    emit_code!(code);
                    remaining -= advance;
                    last_word = advance_8byte_word(last_word, advance);
                }
            }

            // Odd trailing code: 2 bytes with the high nibble zero-padded.
            if have_pending {
                out_ptr.write(pending as u8);
                out_ptr.add(1).write(((pending >> 8) & 0x0F) as u8);
                out_ptr = out_ptr.add(2);
            }

            out_ptr.offset_from(out_begin) as usize
        };

        // SAFETY: the unsafe block above wrote `written` bytes into `out`'s spare capacity.
        unsafe { out.set_len(written) };
    }

    /// Assumes ≥ 8 input bytes are valid past `word`'s first byte. PHT entry lengths
    /// are bounded by the PHT invariant (3..=8), so no `remaining` check is needed.
    #[inline]
    fn compress_word_full(&self, word: u64) -> (u16, usize) {
        // SAFETY: codes_two_byte has FSST12_TWO_BYTE_TABLE_SIZE = 65536 entries, and
        // `word as u16 as usize` is always in [0, 65535].
        let two_byte = unsafe {
            *self
                .codes_two_byte
                .get_unchecked((word as u16) as usize)
        };
        let two_byte_code = two_byte & CODE12_MASK;
        let two_byte_len = (two_byte >> CODE12_LEN_SHIFT) as usize;

        let entry = self.lossy_pht.lookup(word);
        if !entry.is_unused() && compare_masked(word, entry.symbol.to_u64(), entry.ignored_bits)
        {
            let entry_len = ((64 - entry.ignored_bits) >> 3) as usize;
            return (entry.code, entry_len);
        }
        (two_byte_code, two_byte_len)
    }

    /// `remaining` is the number of real bytes in `word` (1..=7); padding must not match.
    #[inline]
    fn compress_word(&self, word: u64, remaining: usize) -> (u16, usize) {
        if remaining >= 2 {
            // SAFETY: `word as u16 as usize` is always in [0, 65535].
            let two_byte = unsafe {
                *self
                    .codes_two_byte
                    .get_unchecked((word as u16) as usize)
            };
            let two_byte_code = two_byte & CODE12_MASK;
            let two_byte_len = (two_byte >> CODE12_LEN_SHIFT) as usize;

            if remaining >= 3 {
                let entry = self.lossy_pht.lookup(word);
                if !entry.is_unused() {
                    let entry_len = ((64 - entry.ignored_bits) >> 3) as usize;
                    if entry_len <= remaining
                        && compare_masked(word, entry.symbol.to_u64(), entry.ignored_bits)
                    {
                        return (entry.code, entry_len);
                    }
                }
            }

            (two_byte_code, two_byte_len)
        } else {
            ((word & 0xFF) as u16, 1)
        }
    }

    /// Compress many lines in bulk. FSST12 analogue of [`crate::Compressor::compress_bulk`].
    pub fn compress_bulk(&self, lines: &[&[u8]]) -> Vec<Vec<u8>> {
        lines.iter().map(|line| self.compress(line)).collect()
    }

    /// Borrow a [`Decompressor12`] view over this compressor's symbol table.
    pub fn decompressor(&self) -> Decompressor12<'_> {
        Decompressor12::new(&self.symbols, &self.lengths)
    }

    /// The first 256 entries are the reserved single-byte literals.
    pub fn symbol_table(&self) -> &[Symbol] {
        &self.symbols
    }

    /// Length of each symbol in [`symbol_table`][Self::symbol_table]. Values in 1..=8.
    pub fn symbol_lengths(&self) -> &[u8] {
        &self.lengths
    }
}

/// Decompressor for FSST12-compressed byte streams.
#[derive(Clone)]
pub struct Decompressor12<'a> {
    pub(crate) symbols: &'a [Symbol],
    pub(crate) lengths: &'a [u8],
}

impl<'a> Decompressor12<'a> {
    /// Construct a decompressor from a symbol table and matching length slice.
    ///
    /// # Panics
    ///
    /// Panics if `symbols` and `lengths` have different lengths, if `symbols.len()`
    /// is below [`FSST12_RESERVED_CODES`] or exceeds [`FSST12_MAX_SYMBOLS`], or if any
    /// of the first 256 entries are not the identity single-byte codes required by FSST12.
    pub fn new(symbols: &'a [Symbol], lengths: &'a [u8]) -> Self {
        assert_eq!(
            symbols.len(),
            lengths.len(),
            "symbol table and length table must be the same size"
        );
        assert!(
            symbols.len() >= FSST12_RESERVED_CODES,
            "FSST12 table must contain the {FSST12_RESERVED_CODES} reserved identity codes"
        );
        assert!(
            symbols.len() <= FSST12_MAX_SYMBOLS,
            "FSST12 table cannot exceed {FSST12_MAX_SYMBOLS} entries"
        );
        // Full u64 must equal code so the SWAR fast loop's 8-byte unaligned writes
        // don't smear garbage into the output buffer's slack region.
        for (code, (symbol, &length)) in symbols
            .iter()
            .zip(lengths.iter())
            .enumerate()
            .take(FSST12_RESERVED_CODES)
        {
            assert!(
                length == 1 && symbol.to_u64() == code as u64,
                "FSST12 code {code} must be the identity single-byte symbol"
            );
        }
        Self { symbols, lengths }
    }

    /// The symbol table this decompressor was constructed with.
    pub fn symbol_table(&self) -> &[Symbol] {
        self.symbols
    }

    /// The length table this decompressor was constructed with.
    pub fn symbol_lengths(&self) -> &[u8] {
        self.lengths
    }

    /// Upper bound on the decompressed size: `n_codes <= ceil(compressed.len() * 8 / 12)`
    /// and each code is at most 8 bytes.
    pub fn max_decompression_capacity(&self, compressed: &[u8]) -> usize {
        compressed.len() * 16 / 3 + 8
    }

    /// Decompress a bit-packed 12-bit code stream into a caller-provided buffer, and
    /// return the number of bytes written. The buffer must be at least
    /// [`Self::max_decompression_capacity`] long for the SWAR fast path; smaller buffers
    /// fall back to a byte-by-byte tail.
    ///
    /// # Panics
    ///
    /// Panics if `compressed.len() % 3 == 1` (not a valid FSST12 encoded length), or if
    /// `decoded` is too small to hold the decoded output.
    ///
    /// ```
    /// use fsst::fsst12::{Compressor12, CompressorBuilder12};
    /// use fsst::Symbol;
    ///
    /// let mut builder = CompressorBuilder12::new();
    /// assert!(builder.insert(Symbol::from_slice(b"abcdefgh"), 8));
    /// let compressor = builder.build();
    /// let decompressor = compressor.decompressor();
    ///
    /// let compressed = compressor.compress(b"abcdefgh");
    /// let mut out = Vec::with_capacity(decompressor.max_decompression_capacity(&compressed));
    /// let len = decompressor.decompress_into(&compressed, out.spare_capacity_mut());
    /// // SAFETY: decompress_into initializes the first `len` bytes.
    /// unsafe { out.set_len(len) };
    /// assert_eq!(&out, b"abcdefgh");
    /// ```
    pub fn decompress_into(
        &self,
        compressed: &[u8],
        decoded: &mut [MaybeUninit<u8>],
    ) -> usize {
        if compressed.is_empty() {
            return 0;
        }

        // SAFETY: the unsafe block uses raw pointer arithmetic but checks every write
        // either by reserving 16 bytes of slack in the fast loop or by guarded byte-wise
        // copies in the tail.
        unsafe {
            let in_begin = compressed.as_ptr();
            let in_end = in_begin.add(compressed.len());
            let mut in_ptr = in_begin;

            let out_begin: *mut u8 = decoded.as_mut_ptr().cast();
            let out_end = out_begin.add(decoded.len());
            let mut out_ptr = out_begin;

            // Read 4 input bytes via u32 (consuming 3, peeking 1 for the next iter), emit
            // 2 codes via unaligned u64 writes. The 16-byte output slack lets back-to-back
            // length-1 symbols over-write safely.
            while in_ptr.add(4) <= in_end && out_ptr.add(16) <= out_end {
                let raw = in_ptr.cast::<u32>().read_unaligned();
                in_ptr = in_ptr.add(3);

                let code_a = (raw & CODE12_MASK as u32) as usize;
                let code_b = ((raw >> CODE12_LEN_SHIFT) & CODE12_MASK as u32) as usize;

                let sym_a = self.symbols.get_unchecked(code_a).to_u64();
                let len_a = *self.lengths.get_unchecked(code_a) as usize;
                out_ptr.cast::<u64>().write_unaligned(sym_a);
                out_ptr = out_ptr.add(len_a);

                let sym_b = self.symbols.get_unchecked(code_b).to_u64();
                let len_b = *self.lengths.get_unchecked(code_b) as usize;
                out_ptr.cast::<u64>().write_unaligned(sym_b);
                out_ptr = out_ptr.add(len_b);
            }

            // Byte-wise tail: used both when input runs short (≤ 3 bytes left) and when
            // output runs out of fast-loop slack.
            while in_ptr.add(3) <= in_end {
                let raw = (*in_ptr as u32)
                    | ((*in_ptr.add(1) as u32) << 8)
                    | ((*in_ptr.add(2) as u32) << 16);
                in_ptr = in_ptr.add(3);
                let code_a = (raw & CODE12_MASK as u32) as usize;
                let code_b = ((raw >> CODE12_LEN_SHIFT) & CODE12_MASK as u32) as usize;

                let len_a = *self.lengths.get_unchecked(code_a) as usize;
                assert!(
                    out_end.offset_from(out_ptr) >= len_a as isize,
                    "decoded buffer too small for FSST12 decompression"
                );
                let sym_a = self.symbols.get_unchecked(code_a).to_u64().to_le_bytes();
                std::ptr::copy_nonoverlapping(sym_a.as_ptr(), out_ptr, len_a);
                out_ptr = out_ptr.add(len_a);

                let len_b = *self.lengths.get_unchecked(code_b) as usize;
                assert!(
                    out_end.offset_from(out_ptr) >= len_b as isize,
                    "decoded buffer too small for FSST12 decompression"
                );
                let sym_b = self.symbols.get_unchecked(code_b).to_u64().to_le_bytes();
                std::ptr::copy_nonoverlapping(sym_b.as_ptr(), out_ptr, len_b);
                out_ptr = out_ptr.add(len_b);
            }

            // 0 bytes = clean end; 2 bytes = trailing odd code; 1 byte = invalid.
            let remaining = in_end.offset_from(in_ptr) as usize;
            match remaining {
                0 => {}
                2 => {
                    let raw = (*in_ptr as u16) | ((*in_ptr.add(1) as u16) << 8);
                    let code = (raw & CODE12_MASK) as usize;
                    let len = *self.lengths.get_unchecked(code) as usize;
                    assert!(
                        out_end.offset_from(out_ptr) >= len as isize,
                        "decoded buffer too small for FSST12 decompression"
                    );
                    let sym = self.symbols.get_unchecked(code).to_u64().to_le_bytes();
                    std::ptr::copy_nonoverlapping(sym.as_ptr(), out_ptr, len);
                    out_ptr = out_ptr.add(len);
                }
                _ => panic!("invalid FSST12 packed length"),
            }

            out_ptr.offset_from(out_begin) as usize
        }
    }

    /// Decompress a bit-packed 12-bit code stream into a fresh `Vec`.
    ///
    /// # Panics
    ///
    /// Panics if `compressed.len() % 3 == 1`, which cannot be produced by a valid
    /// [`Compressor12::compress`] output (valid encoded lengths are `0 mod 3` for pairs
    /// of codes, or `2 mod 3` for an odd code count).
    pub fn decompress(&self, compressed: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.max_decompression_capacity(compressed));
        let len = self.decompress_into(compressed, out.spare_capacity_mut());
        // SAFETY: decompress_into initialized the first `len` bytes of `out`'s spare capacity.
        unsafe { out.set_len(len) };
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_round_trip() {
        let compressor = CompressorBuilder12::new().build();
        let compressed = compressor.compress(b"");
        assert!(compressed.is_empty());
        let decompressed = compressor.decompressor().decompress(&compressed);
        assert!(decompressed.is_empty());
    }

    #[test]
    fn single_byte_packs_to_two_bytes_with_padding() {
        // With no learned symbols, byte 0x12 maps to identity code 0x012.
        // One code occupies 12 bits, padded to 2 bytes; high nibble of byte 1 is zero.
        let compressor = CompressorBuilder12::new().build();
        let compressed = compressor.compress(&[0x12]);
        assert_eq!(compressed, vec![0x12, 0x00]);
    }

    #[test]
    fn two_bytes_pack_into_three_bytes_le() {
        // Codes [0x012, 0x034] pack as u24 = 0x012 | (0x034 << 12) = 0x034012,
        // little-endian: [0x12, 0x40, 0x03].
        let compressor = CompressorBuilder12::new().build();
        let compressed = compressor.compress(&[0x12, 0x34]);
        assert_eq!(compressed, vec![0x12, 0x40, 0x03]);
    }

    #[test]
    fn identity_round_trip_even_length() {
        let compressor = CompressorBuilder12::new().build();
        let plaintext: &[u8] = b"abcdef";
        let compressed = compressor.compress(plaintext);
        let decompressed = compressor.decompressor().decompress(&compressed);
        assert_eq!(decompressed, plaintext);
    }

    #[test]
    fn identity_round_trip_odd_length() {
        // Odd byte count exercises the trailing-padded-code path.
        let compressor = CompressorBuilder12::new().build();
        let plaintext: &[u8] = b"abcdefg";
        let compressed = compressor.compress(plaintext);
        let decompressed = compressor.decompressor().decompress(&compressed);
        assert_eq!(decompressed, plaintext);
    }

    #[test]
    fn two_byte_learned_symbol_emits_single_code() {
        // "ab" gets code 256 = 0x100, packed into 2 bytes: low byte 0x00, high nibble 0x01.
        let mut builder = CompressorBuilder12::new();
        assert!(builder.insert(Symbol::from_slice(&[b'a', b'b', 0, 0, 0, 0, 0, 0]), 2));
        let compressor = builder.build();
        let compressed = compressor.compress(b"ab");
        assert_eq!(compressed, vec![0x00, 0x01]);

        let decompressed = compressor.decompressor().decompress(&compressed);
        assert_eq!(decompressed, b"ab");
    }

    #[test]
    fn learned_symbol_prefers_longer_match() {
        // With both 2-byte "ab" and 3-byte "abc" in the table, "abc" must emit 1 code.
        let mut builder = CompressorBuilder12::new();
        assert!(builder.insert(Symbol::from_slice(&[b'a', b'b', 0, 0, 0, 0, 0, 0]), 2));
        assert!(builder.insert(Symbol::from_slice(&[b'a', b'b', b'c', 0, 0, 0, 0, 0]), 3));
        let compressor = builder.build();
        let compressed = compressor.compress(b"abc");
        // Single code 257 = 0x101: low 8 bits = 0x01, high nibble = 0x01.
        assert_eq!(compressed, vec![0x01, 0x01]);

        let decompressed = compressor.decompressor().decompress(&compressed);
        assert_eq!(decompressed, b"abc");
    }

    #[test]
    fn learned_symbol_eight_byte_round_trip() {
        let mut builder = CompressorBuilder12::new();
        assert!(builder.insert(Symbol::from_slice(b"abcdefgh"), 8));
        let compressor = builder.build();
        let plaintext: &[u8] = b"abcdefghZabcdefgh";
        let compressed = compressor.compress(plaintext);
        let decompressed = compressor.decompressor().decompress(&compressed);
        assert_eq!(decompressed, plaintext);
        // 3 codes (8-byte sym, 'Z', 8-byte sym) packed into 5 bytes.
        assert_eq!(compressed.len(), 5);
    }

    #[test]
    fn longer_pht_match_wins_over_two_byte_match() {
        // "ab" (2-byte) and "abcdef" (6-byte) share the same prefix; "abcdef" wins.
        let mut builder = CompressorBuilder12::new();
        assert!(builder.insert(Symbol::from_slice(b"ab\0\0\0\0\0\0"), 2));
        assert!(builder.insert(Symbol::from_slice(b"abcdef\0\0"), 6));
        let compressor = builder.build();
        let compressed = compressor.compress(b"abcdef");
        // Code 257 (= 0x101). Single code, 2 bytes.
        assert_eq!(compressed, vec![0x01, 0x01]);
        let decompressed = compressor.decompressor().decompress(&compressed);
        assert_eq!(decompressed, b"abcdef");
    }

    #[test]
    fn symbol_match_at_end_of_input_no_over_read() {
        // An 8-byte symbol matches the last 8 bytes of a 9-byte input, with no padding
        // for the encoder to falsely match into.
        let mut builder = CompressorBuilder12::new();
        assert!(builder.insert(Symbol::from_slice(b"abcdefgh"), 8));
        let compressor = builder.build();
        let plaintext: &[u8] = b"Xabcdefgh";
        let compressed = compressor.compress(plaintext);
        let decompressed = compressor.decompressor().decompress(&compressed);
        assert_eq!(decompressed, plaintext);
    }

    #[test]
    fn long_identity_round_trip() {
        // 257 bytes: enough for the SWAR body plus an odd-length tail.
        let compressor = CompressorBuilder12::new().build();
        let plaintext: Vec<u8> = (0..257u32).map(|i| (i & 0xFF) as u8).collect();
        let compressed = compressor.compress(&plaintext);
        let decompressed = compressor.decompressor().decompress(&compressed);
        assert_eq!(decompressed, plaintext);
    }

    #[test]
    fn long_mixed_round_trip() {
        let mut builder = CompressorBuilder12::new();
        assert!(builder.insert(Symbol::from_slice(b"ab\0\0\0\0\0\0"), 2));
        assert!(builder.insert(Symbol::from_slice(b"abc\0\0\0\0\0"), 3));
        assert!(builder.insert(Symbol::from_slice(b"the quic"), 8));
        let compressor = builder.build();
        let mut plaintext = Vec::with_capacity(256);
        for _ in 0..20 {
            plaintext.extend_from_slice(b"the quick abc ab xyz ");
        }
        let compressed = compressor.compress(&plaintext);
        let decompressed = compressor.decompressor().decompress(&compressed);
        assert_eq!(decompressed, plaintext);
    }

    #[test]
    fn compress_fast_loop_boundary_sizes() {
        // Spans the 8-byte fast/tail-loop threshold; includes a PHT and a 2-byte symbol.
        let mut builder = CompressorBuilder12::new();
        assert!(builder.insert(Symbol::from_slice(b"abcdefgh"), 8));
        assert!(builder.insert(Symbol::from_slice(b"xy\0\0\0\0\0\0"), 2));
        let compressor = builder.build();

        let base = b"abcdefgh_xy_abcdefgh_xy_";
        for n in [1usize, 2, 3, 7, 8, 9, 15, 16, 17, 24, 25, 31, 32, 33] {
            let plaintext: Vec<u8> = base.iter().copied().cycle().take(n).collect();
            let compressed = compressor.compress(&plaintext);
            let decompressed = compressor.decompressor().decompress(&compressed);
            assert_eq!(decompressed, plaintext, "round-trip failed at n={}", n);
        }
    }

    #[test]
    fn compress_no_learned_symbols_round_trips_at_8_byte_boundary() {
        // Every byte is a length-1 identity code: the fast loop emits one code per iter.
        let compressor = CompressorBuilder12::new().build();
        for n in [7usize, 8, 9, 15, 16, 17, 64, 65] {
            let plaintext: Vec<u8> = (0..n as u32).map(|i| (i & 0xFF) as u8).collect();
            let compressed = compressor.compress(&plaintext);
            // Each code is 12 bits → ceil(n * 12 / 8) bytes.
            assert_eq!(
                compressed.len(),
                n.div_ceil(2) * 3 - if n % 2 == 1 { 1 } else { 0 },
                "compressed length mismatch at n={}",
                n
            );
            let decompressed = compressor.decompressor().decompress(&compressed);
            assert_eq!(decompressed, plaintext, "round-trip failed at n={}", n);
        }
    }

    #[test]
    fn decompress_into_exact_capacity() {
        // With exactly the uncompressed length, decompress_into must finish via the safe
        // byte-by-byte tail rather than the SWAR over-writing past the buffer end.
        let mut builder = CompressorBuilder12::new();
        assert!(builder.insert(Symbol::from_slice(b"aaaaaaaa"), 8));
        let compressor = builder.build();
        let plaintext = vec![b'a'; 10_000];
        let compressed = compressor.compress(&plaintext);

        let decompressor = compressor.decompressor();
        let mut decompressed: Vec<u8> = Vec::with_capacity(plaintext.len());
        let len =
            decompressor.decompress_into(&compressed, decompressed.spare_capacity_mut());
        // SAFETY: decompress_into initialized the first `len` bytes.
        unsafe { decompressed.set_len(len) };
        assert_eq!(decompressed, plaintext);
    }

    #[test]
    #[should_panic(expected = "invalid FSST12 packed length")]
    fn decompress_into_invalid_packed_length_panics() {
        // Valid FSST12 lengths are `0 mod 3` or `2 mod 3`; a 1-byte buffer is neither.
        let compressor = CompressorBuilder12::new().build();
        let decompressor = compressor.decompressor();
        let mut out: Vec<u8> = Vec::with_capacity(8);
        decompressor.decompress_into(&[0x12], out.spare_capacity_mut());
    }

    #[test]
    fn decompressor_getters_expose_underlying_tables() {
        let mut builder = CompressorBuilder12::new();
        assert!(builder.insert(Symbol::from_slice(b"abcdefgh"), 8));
        let compressor = builder.build();
        let decompressor = compressor.decompressor();
        assert_eq!(decompressor.symbol_table(), compressor.symbol_table());
        assert_eq!(decompressor.symbol_lengths(), compressor.symbol_lengths());
    }

    #[test]
    fn compress_bulk_round_trips_each_line() {
        let mut builder = CompressorBuilder12::new();
        assert!(builder.insert(Symbol::from_slice(b"abcdefgh"), 8));
        let compressor = builder.build();
        let lines: [&[u8]; 3] = [b"abcdefgh".as_slice(), b"hello", b""];
        let bulk = compressor.compress_bulk(&lines);
        assert_eq!(bulk.len(), 3);
        let decompressor = compressor.decompressor();
        for (line, encoded) in lines.iter().zip(bulk.iter()) {
            assert_eq!(decompressor.decompress(encoded), *line);
        }
    }

    #[test]
    #[should_panic(expected = "must be the identity single-byte symbol")]
    fn decompressor_new_rejects_non_identity_reserved_code() {
        let mut symbols: Vec<Symbol> = (0..FSST12_RESERVED_CODES)
            .map(|code| Symbol::from_u8(code as u8))
            .collect();
        let mut lengths = vec![1u8; FSST12_RESERVED_CODES];
        // Corrupt one reserved slot.
        symbols[42] = Symbol::from_slice(b"corrupt!");
        lengths[42] = 8;
        Decompressor12::new(&symbols, &lengths);
    }

    #[test]
    #[should_panic(expected = "reserved identity codes")]
    fn decompressor_new_rejects_table_below_reserved_size() {
        let symbols = vec![Symbol::ZERO; FSST12_RESERVED_CODES - 1];
        let lengths = vec![1u8; FSST12_RESERVED_CODES - 1];
        Decompressor12::new(&symbols, &lengths);
    }

    #[test]
    #[should_panic(expected = "decoded buffer too small")]
    fn decompress_into_buffer_too_small_panics() {
        let mut builder = CompressorBuilder12::new();
        assert!(builder.insert(Symbol::from_slice(b"aaaaaaaa"), 8));
        let compressor = builder.build();
        let plaintext = vec![b'a'; 1_000];
        let compressed = compressor.compress(&plaintext);
        let decompressor = compressor.decompressor();

        let mut undersized: Vec<u8> = Vec::with_capacity(plaintext.len() / 2);
        decompressor.decompress_into(&compressed, undersized.spare_capacity_mut());
    }
}
