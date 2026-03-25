//! Functions and types for building a [`Compressor12`] from a corpus of text.
//!
//! Same generational training algorithm as the 8-bit variant, but adapted for
//! 12-bit codes with identity mappings for all 256 byte values.

use super::{
    Code12, Compressor12, FSST12_CODE_BASE, FSST12_CODE_MASK, FSST12_CODE_MAX, FSST12_MAX_SYMBOLS,
    lossy_pht::LossyPHT,
};
use crate::{Symbol, advance_8byte_word, builder::fsst_hash, compare_masked};
use rustc_hash::{FxBuildHasher, FxHashMap};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// ---- CodesBitmap12: 4096-bit bitmap ----

/// Bitmap supporting values 0..4095 for tracking observed codes.
#[derive(Clone, Copy, Debug)]
struct CodesBitmap12 {
    codes: [u64; 64],
}

assert_sizeof!(CodesBitmap12 => 512);

impl Default for CodesBitmap12 {
    fn default() -> Self {
        Self { codes: [0; 64] }
    }
}

impl CodesBitmap12 {
    /// Set the indicated bit.
    pub(crate) fn set(&mut self, index: usize) {
        debug_assert!(
            index <= FSST12_CODE_MASK as usize,
            "code cannot exceed {FSST12_CODE_MASK}"
        );
        let map = index >> 6;
        self.codes[map] |= 1 << (index % 64);
    }

    /// Check if `index` is present in the bitmap.
    pub(crate) fn is_set(&self, index: usize) -> bool {
        debug_assert!(
            index <= FSST12_CODE_MASK as usize,
            "code cannot exceed {FSST12_CODE_MASK}"
        );
        let map = index >> 6;
        self.codes[map] & (1 << (index % 64)) != 0
    }

    /// Iterate over all set bits.
    pub(crate) fn codes(&self) -> CodesIterator12<'_> {
        CodesIterator12 {
            inner: self,
            index: 0,
            block: self.codes[0],
            reference: 0,
        }
    }

    /// Clear all bits.
    pub(crate) fn clear(&mut self) {
        self.codes = [0; 64];
    }
}

struct CodesIterator12<'a> {
    inner: &'a CodesBitmap12,
    index: usize,
    block: u64,
    reference: usize,
}

impl Iterator for CodesIterator12<'_> {
    type Item = u16;

    fn next(&mut self) -> Option<Self::Item> {
        while self.block == 0 {
            self.index += 1;
            if self.index >= 64 {
                return None;
            }
            self.block = self.inner.codes[self.index];
            self.reference = self.index * 64;
        }

        let position = self.block.trailing_zeros() as usize;
        let code = self.reference + position;

        if code >= FSST12_CODE_MASK as usize {
            return None;
        }

        self.reference = code + 1;
        self.block = if position == 63 {
            0
        } else {
            self.block >> (1 + position)
        };

        Some(code as u16)
    }
}

// ---- Counter12 ----

const COUNTS1_SIZE: usize = FSST12_CODE_MAX as usize;

// NOTE: 4096 * 4096 * 8 = 128MB. This is a known cost of 12-bit FSST.
// The bitmap-guarded pattern avoids touching most pages (demand-paged by OS).
const COUNTS2_SIZE: usize = COUNTS1_SIZE * COUNTS1_SIZE;

#[derive(Debug, Clone)]
struct Counter12 {
    /// Frequency count for each code (0-4095).
    counts1: Vec<usize>,

    /// Frequency count for each code-pair.
    counts2: Vec<usize>,

    /// Bitmap index for codes that appear in counts1.
    code1_index: CodesBitmap12,

    /// Bitmap index of pairs that have been set.
    pair_index: Vec<CodesBitmap12>,
}

impl Counter12 {
    fn new() -> Self {
        let mut counts1 = Vec::with_capacity(COUNTS1_SIZE);
        let mut counts2 = Vec::with_capacity(COUNTS2_SIZE);
        // SAFETY: all accesses to the vector go through the bitmap to ensure no uninitialized
        //  data is ever read from these vectors.
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
    fn record_count1(&mut self, code1: u16) {
        let base = if self.code1_index.is_set(code1 as usize) {
            self.counts1[code1 as usize]
        } else {
            0
        };
        self.counts1[code1 as usize] = base + 1;
        self.code1_index.set(code1 as usize);
    }

    #[inline]
    fn record_count2(&mut self, code1: u16, code2: u16) {
        debug_assert!(code1 == FSST12_CODE_MASK || self.code1_index.is_set(code1 as usize));
        debug_assert!(self.code1_index.is_set(code2 as usize));

        let idx = (code1 as usize) * COUNTS1_SIZE + (code2 as usize);
        if self.pair_index[code1 as usize].is_set(code2 as usize) {
            self.counts2[idx] += 1;
        } else {
            self.counts2[idx] = 1;
        }
        self.pair_index[code1 as usize].set(code2 as usize);
    }

    #[inline]
    fn count1(&self, code1: u16) -> usize {
        debug_assert!(self.code1_index.is_set(code1 as usize));
        self.counts1[code1 as usize]
    }

    #[inline]
    fn count2(&self, code1: u16, code2: u16) -> usize {
        debug_assert!(self.code1_index.is_set(code1 as usize));
        debug_assert!(self.code1_index.is_set(code2 as usize));
        debug_assert!(self.pair_index[code1 as usize].is_set(code2 as usize));

        let idx = (code1 as usize) * COUNTS1_SIZE + (code2 as usize);
        self.counts2[idx]
    }

    fn first_codes(&self) -> CodesIterator12<'_> {
        self.code1_index.codes()
    }

    fn second_codes(&self, code1: u16) -> CodesIterator12<'_> {
        self.pair_index[code1 as usize].codes()
    }

    fn clear(&mut self) {
        self.code1_index.clear();
        for index in &mut self.pair_index {
            index.clear();
        }
    }
}

// ---- CompressorBuilder12 ----

/// Entrypoint for building a new 12-bit [`Compressor12`].
pub struct CompressorBuilder12 {
    /// Table mapping codes to symbols.
    /// Indices 0-255: identity symbols. Indices 256+: multi-byte symbols.
    symbols: Vec<Symbol>,

    /// Number of multi-byte symbols inserted (max [`FSST12_MAX_SYMBOLS`]).
    n_symbols: u16,

    /// Histogram of symbol counts by length. `len_histogram[len-1]` = count at `len`.
    len_histogram: [u16; 8],

    /// Inverted index mapping 2-byte inputs to codes.
    codes_two_byte: Vec<Code12>,

    /// Lossy perfect hash table for 3+ byte symbols.
    lossy_pht: LossyPHT,
}

impl CompressorBuilder12 {
    /// Create a new builder with identity mappings for all 256 byte values.
    pub fn new() -> Self {
        // Allocate as u64 for the fast vec![0; N] specialization, then transmute.
        let symbols = vec![0u64; FSST12_CODE_MAX as usize];
        // SAFETY: Symbol is repr-transparent over u64.
        let mut symbols: Vec<Symbol> = unsafe { std::mem::transmute(symbols) };

        // Populate identity symbols at positions 0-255.
        for byte in 0..=255u8 {
            symbols[byte as usize] = Symbol::from_u8(byte);
        }

        // Identity fallback codes for the two-byte table.
        let mut codes_two_byte = Vec::with_capacity(65_536);
        for idx in 0..65_536u32 {
            codes_two_byte.push(Code12::new_identity(idx as u8));
        }

        Self {
            symbols,
            n_symbols: 0,
            len_histogram: [0; 8],
            codes_two_byte,
            lossy_pht: LossyPHT::new(),
        }
    }

    /// Attempt to insert a new multi-byte symbol.
    ///
    /// # Panics
    ///
    /// If the symbol table is full, or if a 1-byte symbol is provided.
    ///
    /// # Returns
    ///
    /// True if inserted successfully, false if rejected by the hash table.
    pub fn insert(&mut self, symbol: Symbol, len: usize) -> bool {
        assert!(
            self.n_symbols < FSST12_MAX_SYMBOLS,
            "cannot insert into full 12-bit symbol table"
        );
        assert_eq!(len, symbol.len(), "provided len must equal symbol.len()");
        assert!(
            len >= 2,
            "12-bit FSST does not need 1-byte symbols (they are identity codes)"
        );

        if len == 2 {
            self.codes_two_byte[symbol.first2() as usize] =
                Code12::new_symbol_building(self.n_symbols, 2);
        } else {
            // 3+ byte symbols go into the hash table.
            if !self.lossy_pht.insert(symbol, len, self.n_symbols) {
                return false;
            }
        }

        self.len_histogram[len - 1] += 1;
        self.symbols[(FSST12_CODE_BASE + self.n_symbols) as usize] = symbol;
        self.n_symbols += 1;
        true
    }

    /// Clear all multi-byte symbols, keeping identity codes intact.
    fn clear(&mut self) {
        for i in 0..self.n_symbols as usize {
            let symbol = self.symbols[FSST12_CODE_BASE as usize + i];
            if symbol.len() == 2 {
                self.codes_two_byte[symbol.first2() as usize] =
                    Code12::new_identity(symbol.first_byte());
            } else {
                self.lossy_pht.remove(symbol);
            }
        }
        self.len_histogram = [0; 8];
        self.n_symbols = 0;
    }

    /// Finalize the table: reorder codes by length (2 no-suffix | 2 suffix | 3..8),
    /// rebuild lookup structures.
    ///
    /// Returns (has_suffix_code, lengths_for_all_codes).
    #[allow(clippy::needless_range_loop)]
    fn finalize(&mut self) -> (u16, Vec<u8>) {
        // Cumulative code assignment starting from CODE_BASE.
        // Order: 2 (no suffix) | 2 (suffix) | 3 | 4 | 5 | 6 | 7 | 8
        let mut codes_by_length = [0u16; 8];
        // Length-1 (1-byte): not used in 12-bit builder.
        codes_by_length[0] = 0;
        // Length-2: starts at CODE_BASE.
        codes_by_length[1] = FSST12_CODE_BASE;
        // Lengths 3..8: cumulative.
        for i in 1..7 {
            codes_by_length[i + 1] = codes_by_length[i] + self.len_histogram[i];
        }

        // Split 2-byte codes into no-suffix and has-suffix groups.
        let mut no_suffix_code = FSST12_CODE_BASE;
        let mut has_suffix_code = codes_by_length[2]; // start of 3-byte range

        let mut new_codes = vec![0u16; self.n_symbols as usize];

        for i in 0..self.n_symbols as usize {
            let symbol = self.symbols[FSST12_CODE_BASE as usize + i];
            let len = symbol.len();

            if len == 2 {
                let has_suffix = self.symbols[FSST12_CODE_BASE as usize
                    ..FSST12_CODE_BASE as usize + self.n_symbols as usize]
                    .iter()
                    .enumerate()
                    .any(|(k, other)| i != k && symbol.first2() == other.first2());

                if has_suffix {
                    has_suffix_code -= 1;
                    new_codes[i] = has_suffix_code;
                } else {
                    new_codes[i] = no_suffix_code;
                    no_suffix_code += 1;
                }
            } else {
                new_codes[i] = codes_by_length[len - 1];
                codes_by_length[len - 1] += 1;
            }
        }

        // Reorder symbols into their final positions using a temp buffer to avoid
        // overwriting unprocessed entries.
        let mut new_symbols = vec![Symbol::ZERO; self.n_symbols as usize];
        for i in 0..self.n_symbols as usize {
            let new_idx = (new_codes[i] - FSST12_CODE_BASE) as usize;
            new_symbols[new_idx] = self.symbols[FSST12_CODE_BASE as usize + i];
        }
        for i in 0..self.n_symbols as usize {
            self.symbols[FSST12_CODE_BASE as usize + i] = new_symbols[i];
        }

        // Truncate to only the codes we use.
        self.symbols
            .truncate(FSST12_CODE_BASE as usize + self.n_symbols as usize);

        // Rebuild codes_two_byte with finalized codes.
        for two_bytes in 0..65_536usize {
            let entry = self.codes_two_byte[two_bytes];
            if entry.code() >= FSST12_CODE_BASE {
                let builder_idx = entry.builder_index();
                let new_code = new_codes[builder_idx as usize];
                self.codes_two_byte[two_bytes] = Code12::new_symbol(new_code, 2);
            } else {
                // Reset to identity for the first byte.
                self.codes_two_byte[two_bytes] = Code12::new_identity(two_bytes as u8);
            }
        }

        // Renumber the hash table.
        self.lossy_pht.renumber(&new_codes);

        // Build lengths array for all codes.
        let total = self.symbols.len();
        let mut lengths = Vec::with_capacity(total);
        for i in 0..FSST12_CODE_BASE as usize {
            let _ = i;
            lengths.push(1u8);
        }
        for sym in &self.symbols[FSST12_CODE_BASE as usize..] {
            lengths.push(sym.len() as u8);
        }

        (has_suffix_code, lengths)
    }

    /// Build the final [`Compressor12`].
    pub fn build(mut self) -> Compressor12 {
        let (has_suffix_code, lengths) = self.finalize();

        Compressor12 {
            symbols: self.symbols,
            lengths,
            n_symbols: self.n_symbols,
            has_suffix_code,
            codes_two_byte: self.codes_two_byte,
            lossy_pht: self.lossy_pht,
        }
    }
}

impl Default for CompressorBuilder12 {
    fn default() -> Self {
        Self::new()
    }
}

// ---- Training ----

/// Generations for training, same as 8-bit.
#[cfg(not(miri))]
const GENERATIONS: [usize; 5] = [8usize, 38, 68, 98, 128];
#[cfg(miri)]
const GENERATIONS: [usize; 3] = [8usize, 38, 128];

const FSST_SAMPLETARGET: usize = 1 << 14;
const FSST_SAMPLEMAX: usize = 1 << 15;
const FSST_SAMPLELINE: usize = 512;

fn make_sample<'a, 'b: 'a>(sample_buf: &'a mut Vec<u8>, str_in: &Vec<&'b [u8]>) -> Vec<&'a [u8]> {
    assert!(
        sample_buf.capacity() >= FSST_SAMPLEMAX,
        "sample_buf.len() < FSST_SAMPLEMAX"
    );

    let mut sample: Vec<&[u8]> = Vec::new();

    let tot_size: usize = str_in.iter().map(|s| s.len()).sum();
    if tot_size < FSST_SAMPLETARGET {
        return str_in.clone();
    }

    let mut sample_rnd = fsst_hash(4637947);
    let sample_lim = FSST_SAMPLETARGET;
    let mut sample_buf_offset: usize = 0;

    while sample_buf_offset < sample_lim {
        sample_rnd = fsst_hash(sample_rnd);
        let line_nr = (sample_rnd as usize) % str_in.len();

        let Some(line) = (line_nr..str_in.len())
            .chain(0..line_nr)
            .map(|line_nr| str_in[line_nr])
            .find(|line| !line.is_empty())
        else {
            return sample;
        };

        let chunks = 1 + ((line.len() - 1) / FSST_SAMPLELINE);
        sample_rnd = fsst_hash(sample_rnd);
        let chunk = FSST_SAMPLELINE * ((sample_rnd as usize) % chunks);

        let len = FSST_SAMPLELINE.min(line.len() - chunk);

        sample_buf.extend_from_slice(&line[chunk..chunk + len]);

        let slice =
            unsafe { std::slice::from_raw_parts(sample_buf.as_ptr().add(sample_buf_offset), len) };

        sample.push(slice);
        sample_buf_offset += len;
    }

    sample
}

impl Compressor12 {
    /// Build and train a 12-bit `Compressor12` from a sample corpus of text.
    ///
    /// Uses the same generational algorithm as 8-bit FSST but with up to
    /// [`FSST12_MAX_SYMBOLS`] multi-byte symbols and no escape mechanism.
    ///
    /// ```
    /// use fsst::fsst12::Compressor12;
    ///
    /// let text = b"the quick brown fox jumped over the lazy dog";
    /// let compressor = Compressor12::train(&vec![text.as_slice()]);
    /// let compressed = compressor.compress(text);
    /// assert_eq!(compressor.decompressor().decompress(&compressed), text);
    /// ```
    pub fn train(values: &Vec<&[u8]>) -> Self {
        let mut builder = CompressorBuilder12::new();

        if values.is_empty() {
            return builder.build();
        }

        let mut counters = Counter12::new();
        let mut sample_memory = Vec::with_capacity(FSST_SAMPLEMAX);
        let mut pqueue = BinaryHeap::with_capacity(65_536);

        let sample = make_sample(&mut sample_memory, values);
        for sample_frac in GENERATIONS {
            for (i, line) in sample.iter().enumerate() {
                if sample_frac < 128 && ((fsst_hash(i as u64) & 127) as usize) > sample_frac {
                    continue;
                }
                builder.compress_count(line, &mut counters);
            }

            pqueue.clear();
            builder.optimize(&counters, sample_frac, &mut pqueue);
            counters.clear();
        }

        builder.build()
    }
}

impl CompressorBuilder12 {
    /// Find the longest symbol using the hash table and the codes_two_byte index.
    fn find_longest_symbol(&self, word: u64) -> Code12 {
        // Probe the hash table first for a long match.
        let entry = self.lossy_pht.lookup(word);
        let ignored_bits = entry.ignored_bits;

        if !entry.is_unused() && compare_masked(word, entry.symbol.to_u64(), ignored_bits) {
            return entry.code;
        }

        // Try matching first two bytes.
        let twobyte = self.codes_two_byte[word as u16 as usize];
        if twobyte.code() >= FSST12_CODE_BASE {
            return twobyte;
        }

        // Fall back to identity code for the first byte.
        Code12::new_identity(word as u8)
    }

    /// Compress the text using the current symbol table, counting code frequencies.
    fn compress_count(&self, sample: &[u8], counter: &mut Counter12) -> usize {
        let mut gain = 0;
        if sample.is_empty() {
            return gain;
        }

        let mut in_ptr = sample.as_ptr();
        let in_end = unsafe { in_ptr.byte_add(sample.len()) };
        let in_end_sub8 = in_end as usize - 8;

        let mut prev_code: u16 = FSST12_CODE_MASK;

        while (in_ptr as usize) < in_end_sub8 {
            let word: u64 = unsafe { std::ptr::read_unaligned(in_ptr as *const u64) };
            let code = self.find_longest_symbol(word);
            let code_u16 = code.code();

            // Gain: symbol length minus 1 for multi-byte, 0 for identity.
            if code_u16 >= FSST12_CODE_BASE {
                gain += code.len() as usize - 1;
            }

            counter.record_count1(code_u16);
            counter.record_count2(prev_code, code_u16);

            // Also record the count for just extending by a single byte,
            // if the symbol is not itself a single byte.
            if code.len() > 1 {
                let code_first_byte = self.symbols[code_u16 as usize].first_byte() as u16;
                counter.record_count1(code_first_byte);
                counter.record_count2(prev_code, code_first_byte);
            }

            in_ptr = unsafe { in_ptr.byte_add(code.len() as usize) };
            prev_code = code_u16;
        }

        let remaining_bytes = unsafe { in_end.byte_offset_from(in_ptr) };
        assert!(
            remaining_bytes.is_positive(),
            "in_ptr exceeded in_end, should not be possible"
        );
        let remaining_bytes = remaining_bytes as usize;

        let mut bytes = [0u8; 8];
        unsafe {
            std::ptr::copy_nonoverlapping(in_ptr, bytes.as_mut_ptr(), remaining_bytes);
        }
        let mut last_word = u64::from_le_bytes(bytes);
        let mut remaining_bytes = remaining_bytes;

        while remaining_bytes > 0 {
            let code = self.find_longest_symbol(last_word);
            let code_u16 = code.code();

            if code_u16 >= FSST12_CODE_BASE {
                gain += code.len() as usize - 1;
            }

            counter.record_count1(code_u16);
            counter.record_count2(prev_code, code_u16);

            if code.len() > 1 {
                let code_first_byte = self.symbols[code_u16 as usize].first_byte() as u16;
                counter.record_count1(code_first_byte);
                counter.record_count2(prev_code, code_first_byte);
            }

            let advance = code.len() as usize;
            remaining_bytes -= advance;
            last_word = advance_8byte_word(last_word, advance);
            prev_code = code_u16;
        }

        gain
    }

    /// Using counters and existing symbols, build a new set that optimizes compression gain.
    fn optimize(
        &mut self,
        counters: &Counter12,
        sample_frac: usize,
        pqueue: &mut BinaryHeap<Candidate>,
    ) {
        let mut candidates = FxHashMap::with_capacity_and_hasher(4096, FxBuildHasher);

        for code1 in counters.first_codes() {
            let symbol1 = self.symbols[code1 as usize];
            let symbol1_len = symbol1.len();
            let count = counters.count1(code1);

            if count < (5 * sample_frac / 128) {
                continue;
            }

            // Only multi-byte symbols are candidates (identity codes are always present).
            if symbol1_len >= 2 {
                let gain = count * symbol1_len;
                *candidates.entry(symbol1).or_insert(0) += gain;
            }

            // Consider merging with following codes (skip on last round or max length).
            if sample_frac >= 128 || symbol1_len == 8 {
                continue;
            }

            for code2 in counters.second_codes(code1) {
                let symbol2 = self.symbols[code2 as usize];

                if symbol1_len + symbol2.len() > 8 {
                    continue;
                }

                let new_symbol = symbol1.concat(symbol2);

                // Skip merges that produce 1-byte results (e.g., 0x00 + 0x00
                // collapses to Symbol(0) with len=1 due to the zero-symbol edge case).
                if new_symbol.len() < 2 {
                    continue;
                }

                let gain = counters.count2(code1, code2) * new_symbol.len();

                *candidates.entry(new_symbol).or_insert(0) += gain;
            }
        }

        // Transfer to priority queue.
        for (symbol, gain) in candidates {
            pqueue.push(Candidate { symbol, gain });
        }

        // Clear builder and re-insert the top candidates.
        self.clear();

        let mut n_symbols = 0u16;
        while !pqueue.is_empty() && n_symbols < FSST12_MAX_SYMBOLS {
            let candidate = pqueue.pop().unwrap();
            debug_assert!(candidate.symbol.len() >= 2);
            if self.insert(candidate.symbol, candidate.symbol.len()) {
                n_symbols += 1;
            }
        }
    }
}

// ---- Candidate ----

#[derive(Copy, Clone, Debug)]
struct Candidate {
    gain: usize,
    symbol: Symbol,
}

impl Candidate {
    fn comparable_form(&self) -> (usize, usize) {
        (self.gain, self.symbol.len())
    }
}

impl Eq for Candidate {}

impl PartialEq<Self> for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.comparable_form().eq(&other.comparable_form())
    }
}

impl PartialOrd<Self> for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.comparable_form().cmp(&other.comparable_form())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_bitmap12() {
        let mut map = CodesBitmap12::default();
        map.set(10);
        map.set(100);
        map.set(500);
        map.set(3000);

        let codes: Vec<u16> = map.codes().collect();
        assert_eq!(codes, vec![10u16, 100, 500, 3000]);

        // Empty case.
        let map = CodesBitmap12::default();
        assert!(map.codes().collect::<Vec<_>>().is_empty());
    }

    #[test]
    fn test_builder12() {
        let text = b"hello hello hello hello hello";
        let table = Compressor12::train(&vec![text, text, text, text, text]);

        let compressed = table.compress(text);
        let decompressed = table.decompressor().decompress(&compressed);
        assert_eq!(&decompressed, text);
    }

    #[test]
    fn test_builder12_diverse() {
        // JSON-like data that FSST12 should handle well.
        let text = br#"{"name":"alice","age":30,"city":"wonderland","active":true}"#;
        let corpus: Vec<&[u8]> = std::iter::repeat_n(text.as_slice(), 100).collect();
        let table = Compressor12::train(&corpus);

        let compressed = table.compress(text);
        let decompressed = table.decompressor().decompress(&compressed);
        assert_eq!(&decompressed, text);

        // Should achieve some compression.
        assert!(
            compressed.len() < text.len(),
            "compressed={} should be smaller than original={}",
            compressed.len(),
            text.len()
        );
    }
}
