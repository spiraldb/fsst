//! Builds a [`Compressor12`] from a corpus of text.
//!
//! Same generational loop as [`crate::builder`], with two adjustments for FSST12:
//! length-1 candidates are skipped (they are pre-reserved as identity codes), and no
//! escape penalty is charged.

use crate::{
    Symbol,
    builder::{FSST_SAMPLETARGET, fsst_hash, make_sample},
    compare_masked,
};
use rustc_hash::{FxBuildHasher, FxHashMap};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::{
    CODE12_LEN_SHIFT, CODE12_MASK, Compressor12, FSST12_MAX_LEARNED, FSST12_MAX_SYMBOLS,
    FSST12_RESERVED_CODES, FSST12_TWO_BYTE_TABLE_SIZE, lossy_pht::LossyPht12, pack_code12,
};

/// Sample fractions used per training generation.
///
/// Deliberately differs from cwida/fsst's FSST12-specific schedule (`[14, 52, 90, 128]`
/// in `libfsst12.cpp`, four rounds), which is tuned for chaotic input like URLs / JSON.
/// We reuse the five-round FSST8 schedule here because it compresses text materially
/// better in our sweep (declaration / wikipedia / l_comment all degrade ~5-10% on the
/// cwida schedule); the cwida schedule only wins on the `urls` corpus.
#[cfg(not(miri))]
const GENERATIONS12: [usize; 5] = [8usize, 38, 68, 98, 128];
#[cfg(miri)]
const GENERATIONS12: [usize; 3] = [8usize, 38, 128];

/// Sentinel for `prev_code` before the first emission. Stored as the highest row of
/// [`Counter12::pair_index`]; never appears in `code1_index`.
const FSST12_NO_PREV: u16 = FSST12_MAX_SYMBOLS as u16;

const COUNTS1_SIZE: usize = FSST12_MAX_SYMBOLS + 1;
const COUNTS2_SIZE: usize = COUNTS1_SIZE * COUNTS1_SIZE;

#[derive(Clone, Copy, Debug)]
struct CodesBitmap12 {
    words: [u64; 65],
}

assert_sizeof!(CodesBitmap12 => 520);

impl Default for CodesBitmap12 {
    fn default() -> Self {
        Self { words: [0; 65] }
    }
}

impl CodesBitmap12 {
    fn set(&mut self, index: usize) {
        debug_assert!(index <= FSST12_NO_PREV as usize);
        self.words[index >> 6] |= 1u64 << (index & 63);
    }

    fn is_set(&self, index: usize) -> bool {
        debug_assert!(index <= FSST12_NO_PREV as usize);
        self.words[index >> 6] & (1u64 << (index & 63)) != 0
    }

    fn codes(&self) -> CodesIter12<'_> {
        CodesIter12 {
            inner: self,
            index: 0,
            block: self.words[0],
            reference: 0,
        }
    }

    fn clear(&mut self) {
        for w in &mut self.words {
            *w = 0;
        }
    }
}

struct CodesIter12<'a> {
    inner: &'a CodesBitmap12,
    index: usize,
    block: u64,
    reference: usize,
}

impl Iterator for CodesIter12<'_> {
    type Item = u16;

    fn next(&mut self) -> Option<u16> {
        while self.block == 0 {
            self.index += 1;
            if self.index >= self.inner.words.len() {
                return None;
            }
            self.block = self.inner.words[self.index];
            self.reference = self.index * 64;
        }
        let position = self.block.trailing_zeros() as usize;
        let code = self.reference + position;
        self.reference = code + 1;
        self.block = if position == 63 {
            0
        } else {
            self.block >> (1 + position)
        };
        Some(code as u16)
    }
}

/// Single-code and code-pair counts for one training generation.
///
/// `counts2` is left uninitialized; reads only happen for slots flagged in `pair_index`,
/// which is set on each write.
///
/// `counts2` is a dense `4097 × 4097 × usize` table, ~128 MB. This is a virtual
/// reservation: on Linux/macOS the allocation is large enough to land on `mmap` zero-fill,
/// so physical pages are only committed when written. The bitmap-guarded access pattern
/// means the touched footprint tracks the number of distinct code pairs seen in a
/// generation — typically a few MB even for large corpora. A sparse `FxHashMap` would
/// shrink the virtual reservation but slow down `record_count2` (the training hot path)
/// by an order of magnitude, so the dense layout wins on every target with adequate
/// address space. Note: this is unsuitable for 32-bit targets.
#[derive(Debug, Clone)]
struct Counter12 {
    counts1: Vec<usize>,
    /// Row-major over (prev_code, code).
    counts2: Vec<usize>,
    code1_index: CodesBitmap12,
    /// Per-row bitmap into `counts2`. The extra row at [`FSST12_NO_PREV`] records
    /// first-emission codes.
    pair_index: Vec<CodesBitmap12>,
}

impl Counter12 {
    fn new() -> Self {
        let mut counts1 = Vec::with_capacity(COUNTS1_SIZE);
        let mut counts2 = Vec::with_capacity(COUNTS2_SIZE);
        // SAFETY: reads of these vectors are gated on the bitmap indexes; an unset slot
        // is never read.
        unsafe {
            counts1.set_len(COUNTS1_SIZE);
            counts2.set_len(COUNTS2_SIZE);
        }
        Self {
            counts1,
            counts2,
            code1_index: CodesBitmap12::default(),
            pair_index: vec![CodesBitmap12::default(); COUNTS1_SIZE],
        }
    }

    #[inline]
    fn record_count1(&mut self, code: u16) {
        let idx = code as usize;
        let base = if self.code1_index.is_set(idx) {
            self.counts1[idx]
        } else {
            0
        };
        self.counts1[idx] = base + 1;
        self.code1_index.set(idx);
    }

    #[inline]
    fn record_count2(&mut self, code1: u16, code2: u16) {
        let i1 = code1 as usize;
        let i2 = code2 as usize;
        let pair_idx = i1 * COUNTS1_SIZE + i2;
        if self.pair_index[i1].is_set(i2) {
            self.counts2[pair_idx] += 1;
        } else {
            self.counts2[pair_idx] = 1;
        }
        self.pair_index[i1].set(i2);
    }

    #[inline]
    fn count1(&self, code: u16) -> usize {
        debug_assert!(self.code1_index.is_set(code as usize));
        self.counts1[code as usize]
    }

    #[inline]
    fn count2(&self, code1: u16, code2: u16) -> usize {
        debug_assert!(self.pair_index[code1 as usize].is_set(code2 as usize));
        self.counts2[(code1 as usize) * COUNTS1_SIZE + (code2 as usize)]
    }

    fn first_codes(&self) -> CodesIter12<'_> {
        self.code1_index.codes()
    }

    fn second_codes(&self, code1: u16) -> CodesIter12<'_> {
        self.pair_index[code1 as usize].codes()
    }

    fn clear(&mut self) {
        self.code1_index.clear();
        for row in &mut self.pair_index {
            row.clear();
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Candidate12 {
    gain: usize,
    symbol: Symbol,
}

impl Eq for Candidate12 {}

impl PartialEq<Self> for Candidate12 {
    fn eq(&self, other: &Self) -> bool {
        (self.gain, self.symbol.len()) == (other.gain, other.symbol.len())
    }
}

impl PartialOrd<Self> for Candidate12 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate12 {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.gain, self.symbol.len()).cmp(&(other.gain, other.symbol.len()))
    }
}

/// Builder for a [`Compressor12`].
///
/// A fresh builder starts with the 256 reserved single-byte literal codes populated.
/// Callers add learned symbols of length 2-8 via [`insert`][Self::insert], up to
/// [`FSST12_MAX_SYMBOLS`].
pub struct CompressorBuilder12 {
    symbols: Vec<Symbol>,
    lengths: Vec<u8>,
    /// Indexed by the `u16` of the next two input bytes; each slot is `pack_code12(code,
    /// len)`.
    codes_two_byte: Vec<u16>,
    lossy_pht: LossyPht12,
}

impl CompressorBuilder12 {
    /// Create a new builder with the 256 reserved single-byte literal codes pre-populated.
    pub fn new() -> Self {
        let mut symbols = Vec::with_capacity(FSST12_MAX_SYMBOLS);
        let mut lengths = Vec::with_capacity(FSST12_MAX_SYMBOLS);
        for byte in 0u16..256 {
            symbols.push(Symbol::from_u8(byte as u8));
            lengths.push(1);
        }
        let mut codes_two_byte = Vec::with_capacity(FSST12_TWO_BYTE_TABLE_SIZE);
        for i in 0..FSST12_TWO_BYTE_TABLE_SIZE {
            let low_byte = (i & 0xFF) as u16;
            codes_two_byte.push(pack_code12(low_byte, 1));
        }
        Self {
            symbols,
            lengths,
            codes_two_byte,
            lossy_pht: LossyPht12::new(),
        }
    }

    /// Returns `false` if the table is full or a 3+ byte symbol collides in the PHT.
    ///
    /// # Panics
    ///
    /// Panics if `len` is not in `2..=8`.
    pub fn insert(&mut self, symbol: Symbol, len: usize) -> bool {
        assert!(
            (2..=8).contains(&len),
            "FSST12 learned symbols must be 2-8 bytes; single-byte symbols are reserved"
        );
        if self.symbols.len() >= FSST12_MAX_SYMBOLS {
            return false;
        }
        let code = self.symbols.len() as u16;
        if len == 2 {
            let key = (symbol.to_u64() & 0xFFFF) as usize;
            self.codes_two_byte[key] = pack_code12(code, 2);
        } else if !self.lossy_pht.insert(symbol, len as u8, code) {
            return false;
        }
        self.symbols.push(symbol);
        self.lengths.push(len as u8);
        true
    }

    /// Reset to a fresh builder. Reverts lookup structures slot-by-slot to avoid
    /// rezeroing the 128 KB two-byte table between training generations.
    fn clear(&mut self) {
        for i in FSST12_RESERVED_CODES..self.symbols.len() {
            let symbol = self.symbols[i];
            let len = self.lengths[i];
            if len == 2 {
                let key = (symbol.to_u64() & 0xFFFF) as usize;
                let low_byte = (key & 0xFF) as u16;
                self.codes_two_byte[key] = pack_code12(low_byte, 1);
            } else {
                self.lossy_pht.remove(symbol);
            }
        }
        self.symbols.truncate(FSST12_RESERVED_CODES);
        self.lengths.truncate(FSST12_RESERVED_CODES);
    }

    /// Finalize the builder into a [`Compressor12`].
    pub fn build(self) -> Compressor12 {
        Compressor12 {
            symbols: self.symbols,
            lengths: self.lengths,
            codes_two_byte: self.codes_two_byte,
            lossy_pht: self.lossy_pht,
        }
    }

    /// Builder-side analogue of [`Compressor12::compress_word`], used during training.
    #[inline]
    fn find_longest_symbol(&self, word: u64, remaining: usize) -> (u16, u8) {
        if remaining >= 3 {
            let entry = self.lossy_pht.lookup(word);
            if !entry.is_unused() {
                let len = ((64 - entry.ignored_bits) >> 3) as u8;
                if (len as usize) <= remaining
                    && compare_masked(word, entry.symbol.to_u64(), entry.ignored_bits)
                {
                    return (entry.code, len);
                }
            }
        }
        if remaining >= 2 {
            let packed = self.codes_two_byte[(word as u16) as usize];
            return (packed & CODE12_MASK, (packed >> CODE12_LEN_SHIFT) as u8);
        }
        ((word & 0xFF) as u16, 1)
    }

    /// Greedy-compress `sample`, recording emitted codes and code-pairs into `counter`.
    /// Returns a coarse savings tally (sum of `len - 1`) used only as a metric.
    fn compress_count(&self, sample: &[u8], counter: &mut Counter12) -> usize {
        if sample.is_empty() {
            return 0;
        }
        let mut pos = 0;
        let mut prev_code: u16 = FSST12_NO_PREV;
        let mut gain = 0usize;
        while pos < sample.len() {
            let (word, remaining) = read_word_padded(sample, pos);
            let (code, len) = self.find_longest_symbol(word, remaining);
            let len_usz = len as usize;
            gain += len_usz.saturating_sub(1);

            counter.record_count1(code);
            counter.record_count2(prev_code, code);

            // Also record the first byte alone, so the optimizer can rediscover a
            // single-byte's pair statistics even when it was absorbed into a longer match.
            if len_usz > 1 {
                let first_byte_code = self.symbols[code as usize].first_byte() as u16;
                counter.record_count1(first_byte_code);
                counter.record_count2(prev_code, first_byte_code);
            }

            pos += len_usz;
            prev_code = code;
        }
        gain
    }

    /// Reset and repopulate with the highest-gain candidates from `counters`.
    fn optimize(
        &mut self,
        counters: &Counter12,
        sample_frac: usize,
        pqueue: &mut BinaryHeap<Candidate12>,
        prune: bool,
    ) {
        // Deduplicate candidates by symbol content.
        let mut candidates: FxHashMap<Symbol, usize> =
            FxHashMap::with_capacity_and_hasher(4096, FxBuildHasher);

        for code1 in counters.first_codes() {
            let symbol1 = self.symbols[code1 as usize];
            let symbol1_len = symbol1.len();
            let count = counters.count1(code1);

            let min_count = if prune { 1 } else { 5 * sample_frac / 128 };
            if count < min_count {
                continue;
            }

            // Length-1 symbols are pre-reserved as identity codes; they cannot become
            // learned candidates on their own. They remain valid as the left half of a
            // merge, which the second loop covers.
            if symbol1_len >= 2 {
                let gain = count * symbol1_len;
                *candidates.entry(symbol1).or_insert(0) += gain;
            }

            if sample_frac >= 128 || symbol1_len == 8 {
                continue;
            }

            for code2 in counters.second_codes(code1) {
                let symbol2 = self.symbols[code2 as usize];
                let symbol2_len = symbol2.len();
                if symbol1_len + symbol2_len > 8 {
                    continue;
                }
                let new_symbol = symbol1.concat(symbol2);
                let new_len = new_symbol.len();
                if new_len < 2 {
                    // Happens when `symbol2 == Symbol::ZERO` is appended; the trailing
                    // zero is indistinguishable from padding.
                    continue;
                }
                let gain = counters.count2(code1, code2) * new_len;
                *candidates.entry(new_symbol).or_insert(0) += gain;
            }
        }

        for (symbol, gain) in candidates {
            pqueue.push(Candidate12 { symbol, gain });
        }

        self.clear();

        let mut n_learned = 0usize;
        while let Some(candidate) = pqueue.pop() {
            if n_learned >= FSST12_MAX_LEARNED {
                break;
            }
            let len = candidate.symbol.len();
            if prune && candidate.gain <= len + 1 {
                continue;
            }
            if self.insert(candidate.symbol, len) {
                n_learned += 1;
            }
        }
    }
}

impl Default for CompressorBuilder12 {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor12 {
    /// Train an FSST12 compressor on a corpus of byte slices. The table grows up to
    /// [`FSST12_MAX_LEARNED`] learned symbols.
    pub fn train(values: &[&[u8]]) -> Self {
        let mut builder = CompressorBuilder12::new();
        if values.is_empty() {
            return builder.build();
        }

        let mut counters = Counter12::new();
        let mut pqueue: BinaryHeap<Candidate12> = BinaryHeap::with_capacity(1 << 16);

        let tot_size: usize = values.iter().map(|s| s.len()).sum();
        let sampled = tot_size >= FSST_SAMPLETARGET;
        let sample = make_sample(values, tot_size);
        for sample_frac in GENERATIONS12 {
            for (i, line) in sample.iter().enumerate() {
                if sample_frac < 128 && ((fsst_hash(i as u64) & 127) as usize) > sample_frac {
                    continue;
                }
                builder.compress_count(line, &mut counters);
            }
            pqueue.clear();
            let prune = sample_frac >= 128 && !sampled;
            builder.optimize(&counters, sample_frac, &mut pqueue, prune);
            counters.clear();
        }

        builder.build()
    }
}

/// Reads up to 8 bytes from `pos` into a little-endian `u64`, zero-padding if fewer
/// remain. Returns `(word, remaining)` with `remaining` capped at 8.
#[inline]
fn read_word_padded(input: &[u8], pos: usize) -> (u64, usize) {
    let remaining = input.len() - pos;
    if remaining >= 8 {
        // SAFETY: at least 8 bytes are available starting at `pos`.
        let word = unsafe { input.as_ptr().add(pos).cast::<u64>().read_unaligned() };
        (word, 8)
    } else {
        let mut bytes = [0u8; 8];
        bytes[..remaining].copy_from_slice(&input[pos..]);
        (u64::from_le_bytes(bytes), remaining)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fsst12::FSST12_RESERVED_CODES;

    #[test]
    fn builder_prepopulates_reserved_codes() {
        let compressor = CompressorBuilder12::new().build();
        let table = compressor.symbol_table();
        let lengths = compressor.symbol_lengths();
        assert_eq!(table.len(), FSST12_RESERVED_CODES);
        for (code, sym) in table.iter().enumerate() {
            assert_eq!(sym.first_byte(), code as u8);
            assert_eq!(lengths[code], 1);
        }
    }

    #[test]
    fn builder_inserts_learned_symbol() {
        let mut builder = CompressorBuilder12::new();
        assert!(builder.insert(Symbol::from_slice(b"abcdefgh"), 8));
        let compressor = builder.build();
        assert_eq!(compressor.symbol_table().len(), FSST12_RESERVED_CODES + 1);
        assert_eq!(compressor.symbol_lengths()[FSST12_RESERVED_CODES], 8);
    }

    #[test]
    fn builder_caps_at_max_symbols() {
        let mut builder = CompressorBuilder12::new();
        let two_byte = Symbol::from_slice(&[b'a', b'b', 0, 0, 0, 0, 0, 0]);
        for _ in 0..FSST12_MAX_LEARNED {
            assert!(builder.insert(two_byte, 2));
        }
        assert!(!builder.insert(two_byte, 2));
    }

    #[test]
    #[should_panic(expected = "single-byte symbols are reserved")]
    fn builder_rejects_single_byte_inserts() {
        let mut builder = CompressorBuilder12::new();
        builder.insert(Symbol::from_u8(b'a'), 1);
    }

    #[test]
    fn builder_clear_resets_to_reserved_codes_only() {
        let mut builder = CompressorBuilder12::new();
        assert!(builder.insert(Symbol::from_slice(b"ab\0\0\0\0\0\0"), 2));
        assert!(builder.insert(Symbol::from_slice(b"abcdefgh"), 8));
        builder.clear();

        // The codes_two_byte entry for "ab" must be back to the identity for byte 'a'.
        let compressor = builder.build();
        assert_eq!(compressor.symbol_table().len(), FSST12_RESERVED_CODES);
        let compressed = compressor.compress(b"ab");
        // Two identity codes, packed: 0x61 | (0x62 << 12) = 0x062061.
        assert_eq!(compressed, vec![0x61, 0x20, 0x06]);
    }

    #[test]
    fn train_empty_corpus_yields_identity_only_table() {
        let compressor = Compressor12::train(&[]);
        assert_eq!(compressor.symbol_table().len(), FSST12_RESERVED_CODES);
    }

    #[test]
    fn train_round_trip_with_unseen_bytes() {
        let corpus: &[&[u8]] = &[b"hello world".as_slice(), b"hello rust"];
        let compressor = Compressor12::train(corpus);
        let plaintext: &[u8] = b"\x00\x01\x02!@# unseen ~~~";
        let compressed = compressor.compress(plaintext);
        let decompressed = compressor.decompressor().decompress(&compressed);
        assert_eq!(decompressed, plaintext);
    }

    #[test]
    fn train_compresses_repetitive_corpus_below_raw_size() {
        let phrase: &[u8] = b"the quick brown fox jumps over the lazy dog. ";
        let mut text = Vec::with_capacity(phrase.len() * 64);
        for _ in 0..64 {
            text.extend_from_slice(phrase);
        }
        let corpus: Vec<&[u8]> = vec![text.as_slice()];
        let compressor = Compressor12::train(&corpus);
        assert!(
            compressor.symbol_table().len() > FSST12_RESERVED_CODES,
            "training should learn at least one symbol; got {} entries",
            compressor.symbol_table().len()
        );

        let compressed = compressor.compress(&text);
        let decompressed = compressor.decompressor().decompress(&compressed);
        assert_eq!(decompressed, text);
        assert!(
            compressed.len() < text.len(),
            "compressed {} should be smaller than raw {}",
            compressed.len(),
            text.len(),
        );
    }

    #[test]
    fn bitmap_round_trips_set_codes() {
        let mut map = CodesBitmap12::default();
        map.set(10);
        map.set(100);
        map.set(500);
        map.set(4000);
        let codes: Vec<u16> = map.codes().collect();
        assert_eq!(codes, vec![10u16, 100, 500, 4000]);

        let map = CodesBitmap12::default();
        assert!(map.codes().collect::<Vec<_>>().is_empty());

        // First bit in each 64-bit word.
        let mut map = CodesBitmap12::default();
        for w in 0..64 {
            map.set(64 * w);
        }
        assert_eq!(
            map.codes().collect::<Vec<_>>(),
            (0u16..64).map(|w| 64 * w).collect::<Vec<_>>(),
        );

        // Fully saturated: 0..=FSST12_NO_PREV.
        let mut map = CodesBitmap12::default();
        for i in 0..=(FSST12_NO_PREV as usize) {
            map.set(i);
        }
        let collected: Vec<u16> = map.codes().collect();
        assert_eq!(collected.len(), FSST12_NO_PREV as usize + 1);
        assert_eq!(collected.first(), Some(&0));
        assert_eq!(collected.last(), Some(&FSST12_NO_PREV));
    }

    #[test]
    #[should_panic]
    fn bitmap_set_out_of_range_panics() {
        // Relies on debug_assert! firing under `cargo test`'s default debug build.
        let mut map = CodesBitmap12::default();
        map.set(FSST12_NO_PREV as usize + 1);
    }

    #[test]
    fn builder_rejects_pht_collision() {
        // "abcd" and "abce" share the same 3-byte prefix and hash to the same PHT slot.
        let mut builder = CompressorBuilder12::new();
        assert!(builder.insert(Symbol::from_slice(b"abcd\0\0\0\0"), 4));
        assert!(!builder.insert(Symbol::from_slice(b"abce\0\0\0\0"), 4));
        let compressor = builder.build();
        let compressed = compressor.compress(b"abcd");
        assert_eq!(compressed.len(), 2, "first inserted symbol should still match");
    }

    #[test]
    fn train_produces_no_duplicate_symbols() {
        let text = b"aababcabcdabcde";
        let corpus: Vec<&[u8]> = std::iter::repeat_n(text.as_slice(), 100).collect();
        let compressor = Compressor12::train(&corpus);
        let symbols = compressor.symbol_table();
        let lengths = compressor.symbol_lengths();

        let one_byte: Vec<u8> = symbols
            .iter()
            .zip(lengths.iter())
            .filter(|&(_, &len)| len == 1)
            .map(|(sym, _)| sym.first_byte())
            .collect();
        assert_eq!(one_byte.len(), FSST12_RESERVED_CODES);
        let mut one_byte_sorted = one_byte.clone();
        one_byte_sorted.sort_unstable();
        one_byte_sorted.dedup();
        assert_eq!(
            one_byte.len(),
            one_byte_sorted.len(),
            "duplicate 1-byte symbols in trained table"
        );

        let two_byte: Vec<u16> = symbols
            .iter()
            .zip(lengths.iter())
            .filter(|&(_, &len)| len == 2)
            .map(|(sym, _)| sym.first2())
            .collect();
        let mut two_byte_sorted = two_byte.clone();
        two_byte_sorted.sort_unstable();
        two_byte_sorted.dedup();
        assert_eq!(
            two_byte.len(),
            two_byte_sorted.len(),
            "duplicate 2-byte symbols in trained table"
        );
    }

    #[test]
    fn train_round_trip_on_multi_line_corpus() {
        let lines: &[&[u8]] = &[
            b"select * from users where email = 'alice@example.com'".as_slice(),
            b"select * from users where email = 'bob@example.com'",
            b"select * from orders where user_id = 42",
            b"select * from orders where status = 'shipped'",
            b"update users set last_login = now() where id = 7",
        ];
        let compressor = Compressor12::train(lines);
        for line in lines {
            let compressed = compressor.compress(line);
            let decompressed = compressor.decompressor().decompress(&compressed);
            assert_eq!(decompressed, *line);
        }
    }
}
