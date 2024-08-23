//! Functions and types used for building a [`Compressor`] from a corpus of text.
//!
//! This module implements the logic from Algorithm 3 of the [FSST Paper].
//!
//! [FSST Paper]: https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::{advance_8byte_word, extract_u64, CodeMeta, Compressor, Symbol, MAX_CODE};

/// Bitmap that only works for values up to 512
#[derive(Clone, Copy, Debug, Default)]
struct CodesBitmap {
    codes: [u64; 8],
}

assert_sizeof!(CodesBitmap => 64);

impl CodesBitmap {
    /// Set the indicated bit. Must be between 0 and [`MAX_CODE`][crate::MAX_CODE].
    pub(crate) fn set(&mut self, index: usize) {
        debug_assert!(index <= MAX_CODE as usize, "code cannot exceed {MAX_CODE}");

        let map = index >> 6;
        self.codes[map] |= 1 << (index % 64);
    }

    /// Check if `index` is present in the bitmap
    pub(crate) fn is_set(&self, index: usize) -> bool {
        debug_assert!(index <= MAX_CODE as usize, "code cannot exceed {MAX_CODE}");

        let map = index >> 6;
        self.codes[map] & 1 << (index % 64) != 0
    }

    /// Get all codes set in this bitmap
    pub(crate) fn codes(&self) -> CodesIterator {
        CodesIterator {
            inner: self,
            index: 0,
            block: self.codes[0],
            reference: 0,
        }
    }

    /// Clear the bitmap of all entries.
    pub(crate) fn clear(&mut self) {
        self.codes[0] = 0;
        self.codes[1] = 0;
        self.codes[2] = 0;
        self.codes[3] = 0;
        self.codes[4] = 0;
        self.codes[5] = 0;
        self.codes[6] = 0;
        self.codes[7] = 0;
    }
}

struct CodesIterator<'a> {
    inner: &'a CodesBitmap,
    index: usize,
    block: u64,
    reference: usize,
}

impl<'a> Iterator for CodesIterator<'a> {
    type Item = u16;

    fn next(&mut self) -> Option<Self::Item> {
        // If current is zero, advance to next non-zero block
        while self.block == 0 {
            self.index += 1;
            if self.index >= 8 {
                return None;
            }
            self.block = self.inner.codes[self.index];
            self.reference = self.index * 64;
        }

        // Find the next set bit in the current block.
        let position = self.block.trailing_zeros() as usize;
        let code = self.reference + position;

        // The next iteration will calculate with reference to the returned code + 1
        self.reference = code + 1;
        self.block = if position == 63 {
            0
        } else {
            self.block >> (1 + position)
        };

        Some(code as u16)
    }
}

#[derive(Debug, Clone)]
struct Counter {
    /// Frequency count for each code.
    counts1: Vec<usize>,

    /// Frequency count for each code-pair.
    counts2: Vec<usize>,

    /// Bitmap index for codes that appear in counts1
    code1_index: CodesBitmap,

    /// Bitmap index of pairs that have been set.
    ///
    /// `pair_index[code1].codes()` yields an iterator that can
    /// be used to find all possible codes that follow `codes1`.
    pair_index: Vec<CodesBitmap>,
}

const COUNTS1_SIZE: usize = MAX_CODE as usize;
// NOTE: in Rust, creating a 1D vector of length N^2 is ~4x faster than creating a 2-D vector,
//  because `vec!` has a specialization for zero.
const COUNTS2_SIZE: usize = COUNTS1_SIZE * COUNTS1_SIZE;

impl Counter {
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
            code1_index: CodesBitmap::default(),
            pair_index: vec![CodesBitmap::default(); COUNTS1_SIZE],
        }
    }

    #[inline]
    fn record_count1(&mut self, code1: u16) {
        if self.code1_index.is_set(code1 as usize) {
            self.counts1[code1 as usize] += 1;
        } else {
            self.counts1[code1 as usize] = 1;
        }
        self.code1_index.set(code1 as usize);
    }

    #[inline]
    fn record_count2(&mut self, code1: u16, code2: u16) {
        debug_assert!(self.code1_index.is_set(code1 as usize));
        debug_assert!(self.code1_index.is_set(code2 as usize));

        let idx = (code1 as usize) * 511 + (code2 as usize);
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

        let idx = (code1 as usize) * 511 + (code2 as usize);
        self.counts2[idx]
    }

    /// Returns an ordered iterator over the codes that were observed
    /// in a call to [`Self::count1`].
    fn first_codes(&self) -> CodesIterator {
        self.code1_index.codes()
    }

    /// Returns an iterator over the codes that have been observed
    /// to follow `code1`.
    ///
    /// This is the set of all values `code2` where there was
    /// previously a call to `self.record_count2(code1, code2)`.
    fn second_codes(&self, code1: u16) -> CodesIterator {
        self.pair_index[code1 as usize].codes()
    }

    /// Clear the counters.
    /// Note that this just touches the bitmaps and sets them all to invalid.
    fn clear(&mut self) {
        self.code1_index.clear();
        for index in &mut self.pair_index {
            index.clear();
        }
    }
}

/// The number of generations used for training. This is taken from the [FSST paper].
///
/// [FSST paper]: https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf
#[cfg(not(miri))]
const GENERATIONS: [usize; 5] = [8usize, 38, 68, 98, 128];
#[cfg(miri)]
const GENERATIONS: [usize; 3] = [8usize, 38, 128];

const FSST_SAMPLETARGET: usize = 1 << 14;
const FSST_SAMPLEMAX: usize = 1 << 15;
const FSST_SAMPLELINE: usize = 512;

/// Create a sample from a set of strings in the input.
///
/// Sample is constructing by copying "chunks" from the `str_in`s into the `sample_buf`, the
/// returned slices are pointers into the `sample_buf`.
///
/// SAFETY: sample_buf must be >= FSST_SAMPLEMAX bytes long. Providing something less may cause unexpected failures.
#[allow(clippy::ptr_arg)]
fn make_sample<'a, 'b: 'a>(sample_buf: &'a mut Vec<u8>, str_in: &Vec<&'b [u8]>) -> Vec<&'a [u8]> {
    debug_assert!(
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
        let mut line_nr = (sample_rnd as usize) % str_in.len();

        // Find the first non-empty chunk starting at line_nr, wrapping around if
        // necessary.
        //
        // TODO: this will loop infinitely if there are no non-empty lines in the sample
        while str_in[line_nr].is_empty() {
            if line_nr == str_in.len() {
                line_nr = 0;
            }
        }

        let line = str_in[line_nr];
        let chunks = 1 + ((line.len() - 1) / FSST_SAMPLELINE);
        sample_rnd = fsst_hash(sample_rnd);
        let chunk = FSST_SAMPLELINE * ((sample_rnd as usize) % chunks);

        let len = FSST_SAMPLELINE.min(line.len() - chunk);

        sample_buf.extend_from_slice(&str_in[line_nr][chunk..chunk + len]);

        // SAFETY: this is the data we just placed into `sample_buf` in the line above.
        let slice =
            unsafe { std::slice::from_raw_parts(sample_buf.as_ptr().add(sample_buf_offset), len) };

        sample.push(slice);

        sample_buf_offset += len;
    }

    sample
}

/// Hash function used in various components of the library.
///
/// This is equivalent to the FSST_HASH macro from the C++ implementation.
#[inline]
pub(crate) fn fsst_hash(value: u64) -> u64 {
    (value * 2971215073) ^ (value >> 15)
}

impl Compressor {
    /// Clear all set items from the compressor.
    ///
    /// This is considerably faster than building a new Compressor from scratch for each
    /// iteration of the `train` loop.
    fn clear(&mut self) {
        // Eliminate every observed code from the table.
        for code in 0..(256 + self.n_symbols as usize) {
            let symbol = self.symbols[code];
            if symbol.len() <= 2 {
                // Clear the codes_twobyte array
                self.codes_twobyte[symbol.first_two_bytes() as usize] = CodeMeta::EMPTY;
            } else {
                // Clear the hashtable
                self.lossy_pht.remove(symbol);
            }
        }

        self.n_symbols = 0;
    }

    /// Build and train a `Compressor` from a sample corpus of text.
    ///
    /// This function implements the generational algorithm described in the [FSST paper] Section
    /// 4.3. Starting with an empty symbol table, it iteratively compresses the corpus, then attempts
    /// to merge symbols when doing so would yield better compression than leaving them unmerged. The
    /// resulting table will have at most 255 symbols (the 256th symbol is reserved for the escape
    /// code).
    ///
    /// [FSST paper]: https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf
    pub fn train(values: &Vec<&[u8]>) -> Self {
        let mut counters = Counter::new();
        let mut compressor = Compressor::default();

        if values.is_empty() {
            return compressor;
        }

        let mut sample_memory = Vec::with_capacity(FSST_SAMPLEMAX);
        let sample = make_sample(&mut sample_memory, values);
        for sample_frac in GENERATIONS {
            for (i, line) in sample.iter().enumerate() {
                if sample_frac < 128 && ((fsst_hash(i as u64) & 127) as usize) > sample_frac {
                    continue;
                }

                compressor.compress_count(line, &mut counters);
            }

            compressor.optimize(&counters, sample_frac);
            counters.clear();
        }

        compressor
    }
}

impl Compressor {
    /// Compress the text using the current symbol table. Count the code occurrences
    /// and code-pair occurrences to allow us to calculate apparent gain.
    ///
    /// NOTE: this is largely an unfortunate amount of copy-paste from `compress`, just to make sure
    /// we can do all the counting in a single pass.
    fn compress_count(&self, sample: &[u8], counter: &mut Counter) {
        if sample.is_empty() {
            return;
        }

        // Output space.
        let mut out_buf = [0u8, 0u8];

        let mut in_ptr = sample.as_ptr();
        let out_ptr = out_buf.as_mut_ptr();

        // SAFETY: `end` will point just after the end of the `plaintext` slice.
        let in_end = unsafe { in_ptr.byte_add(sample.len()) };
        let in_end_sub8 = in_end as usize - 8;

        let mut prev_code: u16 = MAX_CODE;

        while (in_ptr as usize) < in_end_sub8 {
            // SAFETY: pointer ranges are checked in the loop condition
            unsafe {
                // Load a full 8-byte word of data from in_ptr.
                // SAFETY: caller asserts in_ptr is not null. we may read past end of pointer though.
                let word: u64 = std::ptr::read_unaligned(in_ptr as *const u64);
                let (advance_in, advance_out) = self.compress_word(word, out_ptr);
                match advance_out {
                    1 => {
                        // Record a true symbol
                        let code_u16 = out_ptr.read() as u16 + 256u16;
                        counter.record_count1(code_u16);
                        if prev_code != MAX_CODE {
                            counter.record_count2(prev_code, code_u16);
                            // Also record the first byte of the next code
                            let first_byte_code =
                                self.symbols[code_u16 as usize].first_byte() as u16;
                            counter.record_count1(first_byte_code);
                            counter.record_count2(prev_code, first_byte_code);
                        }
                        prev_code = code_u16;
                    }
                    2 => {
                        // Record an escape.
                        let escape_code = out_ptr.byte_offset(1).read() as u16;
                        counter.record_count1(escape_code);
                        if prev_code != MAX_CODE {
                            counter.record_count2(prev_code, escape_code);
                        }
                        prev_code = escape_code;
                    }
                    _ => unreachable!("advance_out will only be 1 or 2 bytes"),
                }

                in_ptr = in_ptr.byte_add(advance_in);
            };
        }

        let remaining_bytes = unsafe { in_end.byte_offset_from(in_ptr) };
        debug_assert!(
            remaining_bytes.is_positive(),
            "in_ptr exceeded in_end, should not be possible"
        );
        let remaining_bytes = remaining_bytes as usize;

        // Load the last `remaining_byte`s of data into a final world. We then replicate the loop above,
        // but shift data out of this word rather than advancing an input pointer and potentially reading
        // unowned memory.
        let mut last_word = unsafe {
            match remaining_bytes {
                0 => 0,
                1 => extract_u64::<1>(in_ptr),
                2 => extract_u64::<2>(in_ptr),
                3 => extract_u64::<3>(in_ptr),
                4 => extract_u64::<4>(in_ptr),
                5 => extract_u64::<5>(in_ptr),
                6 => extract_u64::<6>(in_ptr),
                7 => extract_u64::<7>(in_ptr),
                8 => extract_u64::<8>(in_ptr),
                _ => unreachable!("remaining bytes must be <= 8"),
            }
        };

        while in_ptr < in_end {
            unsafe {
                // Load a full 8-byte word of data from in_ptr.
                // SAFETY: caller asserts in_ptr is not null. we may read past end of pointer though.
                let (advance_in, advance_out) = self.compress_word(last_word, out_ptr);

                match advance_out {
                    1 => {
                        // Record a true symbol
                        let code_u16 = out_buf[0] as u16 + 256u16;
                        counter.record_count1(code_u16);
                        if prev_code != MAX_CODE {
                            counter.record_count2(prev_code, code_u16);
                        }
                        prev_code = code_u16;
                    }
                    2 => {
                        // Record an escape.
                        let escape_code = out_buf[1] as u16;
                        counter.record_count1(escape_code);
                        if prev_code != MAX_CODE {
                            counter.record_count2(prev_code, escape_code);
                        }
                        prev_code = escape_code;
                    }
                    _ => unreachable!("advance_out will only be 1 or 2 bytes"),
                }

                in_ptr = in_ptr.byte_add(advance_in);

                last_word = advance_8byte_word(last_word, advance_in);
            }
        }
    }

    /// Using a set of counters and the existing set of symbols, build a new
    /// set of symbols/codes that optimizes the gain over the distribution in `counter`.
    fn optimize(&mut self, counters: &Counter, sample_frac: usize) {
        let mut pqueue = BinaryHeap::with_capacity(65_536);

        for code1 in counters.first_codes() {
            let symbol1 = self.symbols[code1 as usize];
            let symbol1_len = symbol1.len();
            let count = counters.count1(code1);

            // From the c++ impl:
            // "improves both compression speed (less candidates), but also quality!!"
            // if count < (5 * sample_frac / 128) {
            //     continue;
            // }

            let mut gain = count * symbol1_len;
            // NOTE: use heuristic from C++ implementation to boost the gain of single-byte symbols.
            // This helps to reduce exception counts.
            if code1 < 256 {
                gain *= 8;
            }

            if gain > 0 {
                pqueue.push(Candidate {
                    symbol: symbol1,
                    gain,
                });
            }

            // Skip merges on last round, or when symbol cannot be extended.
            if sample_frac >= 128 || symbol1_len == 8 {
                continue;
            }

            for code2 in counters.second_codes(code1) {
                let symbol2 = self.symbols[code2 as usize];

                // If merging would yield a symbol of length greater than 8, skip.
                if symbol1_len + symbol2.len() > 8 {
                    continue;
                }
                let new_symbol = symbol1.concat(symbol2);
                let gain = counters.count2(code1, code2) * new_symbol.len();
                if gain > 0 {
                    pqueue.push(Candidate {
                        symbol: new_symbol,
                        gain,
                    })
                }
            }
        }

        // clear self in advance of inserting the symbols.
        self.clear();

        // Pop the 255 best symbols.
        let mut n_symbols = 0;
        while !pqueue.is_empty() && n_symbols < 255 {
            let candidate = pqueue.pop().unwrap();
            if self.insert(candidate.symbol) {
                n_symbols += 1;
            }
        }

        // If there are leftover slots, fill them with ASCII chars.
        // This helps reduce the number of escapes.
        //
        // Note that because of the lossy hash table, we won't accidentally
        // save the same ASCII character twice into the table.
        if sample_frac < 128 {
            for character in
                " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ[](){}:?/<>".bytes()
            {
                if n_symbols == 255 {
                    break;
                }

                if self.insert(Symbol::from_u8(character)) {
                    n_symbols += 1
                }
            }
        }
    }
}

/// A candidate for inclusion in a symbol table.
///
/// This is really only useful for the `optimize` step of training.
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
        let self_ord = (self.gain, self.symbol.len());
        let other_ord = (other.gain, other.symbol.len());

        self_ord.cmp(&other_ord)
    }
}

#[cfg(test)]
mod test {
    use crate::{builder::CodesBitmap, Compressor, ESCAPE_CODE};

    #[test]
    fn test_builder() {
        // Train a Compressor on the toy string
        let text = b"hello hello hello hello";

        // count of 5 is the cutoff for including a symbol in the table.
        let table = Compressor::train(&vec![text, text, text, text, text]);

        // Use the table to compress a string, see the values
        let compressed = table.compress(text);

        // Ensure that the compressed string has no escape bytes
        assert!(compressed.iter().all(|b| *b != ESCAPE_CODE));

        // Ensure that we can compress a string with no values seen at training time, with escape bytes
        let compressed = table.compress("xyz123".as_bytes());
        assert_eq!(
            compressed,
            vec![
                ESCAPE_CODE,
                b'x',
                ESCAPE_CODE,
                b'y',
                ESCAPE_CODE,
                b'z',
                ESCAPE_CODE,
                b'1',
                ESCAPE_CODE,
                b'2',
                ESCAPE_CODE,
                b'3',
            ]
        );
    }

    #[test]
    fn test_bitmap() {
        let mut map = CodesBitmap::default();
        map.set(10);
        map.set(100);
        map.set(500);

        let codes: Vec<u16> = map.codes().collect();
        assert_eq!(codes, vec![10u16, 100, 500]);

        // empty case
        let map = CodesBitmap::default();
        assert!(map.codes().collect::<Vec<_>>().is_empty());

        // edge case: first bit in each block is set
        let mut map = CodesBitmap::default();
        (0..8).for_each(|i| map.set(64 * i));
        assert_eq!(
            map.codes().collect::<Vec<_>>(),
            (0u16..8).map(|i| 64 * i).collect::<Vec<_>>(),
        );

        // Full bitmap case. There are only 512 values, so test them all
        let mut map = CodesBitmap::default();
        for i in 0..512 {
            map.set(i);
        }
        assert_eq!(
            map.codes().collect::<Vec<_>>(),
            (0u16..512u16).collect::<Vec<_>>()
        );
    }

    #[test]
    #[should_panic(expected = "code cannot exceed")]
    fn test_bitmap_invalid() {
        let mut map = CodesBitmap::default();
        map.set(512);
    }
}
