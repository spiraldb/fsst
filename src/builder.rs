//! Functions and types used for building a [`Compressor`] from a corpus of text.
//!
//! This module implements the logic from Algorithm 3 of the [FSST Paper].
//!
//! [FSST Paper]: https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::{Compressor, Symbol, ESCAPE_CODE, MAX_CODE};

#[derive(Debug, Clone)]
struct Counter {
    /// Frequency count for each code.
    counts1: Vec<usize>,

    /// Frequency count for each code-pair.
    counts2: Vec<Vec<usize>>,
}

impl Counter {
    fn new() -> Self {
        Self {
            counts1: vec![0; MAX_CODE as usize],
            counts2: vec![vec![0; MAX_CODE as usize]; MAX_CODE as usize],
        }
    }

    fn reset(&mut self) {
        for code1 in 0..MAX_CODE {
            self.counts1[code1 as usize] = 0;

            for code2 in 0..MAX_CODE {
                self.counts2[code1 as usize][code2 as usize] = 0;
            }
        }
    }

    #[inline]
    fn record_count1(&mut self, code1: u16) {
        self.counts1[code1 as usize] += 1;
    }

    #[inline]
    fn record_count2(&mut self, code1: u16, code2: u16) {
        self.counts2[code1 as usize][code2 as usize] += 1;
    }

    #[inline]
    fn count1(&self, code: u16) -> usize {
        self.counts1[code as usize]
    }

    #[inline]
    fn count2(&self, code1: u16, code2: u16) -> usize {
        self.counts2[code1 as usize][code2 as usize]
    }
}

/// The number of generations used for training. This is taken from the [FSST paper].
///
/// [FSST paper]: https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf
#[cfg(not(miri))]
const MAX_GENERATIONS: usize = 5;

#[cfg(miri)]
const MAX_GENERATIONS: usize = 1;

impl Compressor {
    /// Build and train a `Compressor` from a sample corpus of text.
    ///
    /// This function implements the generational algorithm described in the [FSST paper] Section
    /// 4.3. Starting with an empty symbol table, it iteratively compresses the corpus, then attempts
    /// to merge symbols when doing so would yield better compression than leaving them unmerged. The
    /// resulting table will have at most 255 symbols (the 256th symbol is reserved for the escape
    /// code).
    ///
    /// [FSST paper]: https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf
    pub fn train(corpus: impl AsRef<[u8]>) -> Self {
        let mut compressor = Self::default();
        // TODO(aduffy): handle truncating/sampling if corpus > requires sample size.
        let sample = corpus.as_ref();
        if sample.is_empty() {
            return compressor;
        }

        let mut counter = Counter::new();

        for _generation in 0..(MAX_GENERATIONS - 1) {
            compressor.compress_count(sample, &mut counter);
            compressor = compressor.optimize(&counter, true);
            counter.reset();
        }

        compressor.compress_count(sample, &mut counter);
        compressor.optimize(&counter, true)
    }

    /// Specify the number of generations to train for
    pub fn train_n(corpus: impl AsRef<[u8]>, generations: usize) -> Self {
        let mut compressor = Self::default();
        // TODO(aduffy): handle truncating/sampling if corpus > requires sample size.
        let sample = corpus.as_ref();
        if sample.is_empty() {
            return compressor;
        }

        let mut counter = Counter::new();

        for _generation in 0..(generations - 1) {
            compressor.compress_count(sample, &mut counter);
            compressor = compressor.optimize(&counter, false);
        }

        compressor.compress_count(sample, &mut counter);
        compressor.optimize(&counter, true)
    }
}

impl Compressor {
    /// Compress the text using the current symbol table. Count the code occurrences
    /// and code-pair occurrences to allow us to calculate apparent gain.
    fn compress_count(&self, sample: &[u8], counter: &mut Counter) {
        let compressed = self.compress(sample);
        let len = compressed.len();

        if len == 0 {
            return;
        }

        #[inline(never)]
        fn next_code(pos: usize, compressed: &[u8]) -> (u16, usize) {
            if compressed[pos] == ESCAPE_CODE {
                (compressed[pos + 1] as u16, 2)
            } else {
                (256 + compressed[pos] as u16, 1)
            }
        }

        // Get first code, record count
        let (code, pos) = next_code(0, &compressed);
        counter.record_count1(code);

        let mut pos = pos;
        let mut prev_code = code;

        while pos < len {
            let (code, advance) = next_code(pos, &compressed);
            pos += advance;

            counter.record_count1(code);
            counter.record_count2(prev_code, code);

            prev_code = code;
        }
    }

    /// Using a set of counters and the existing set of symbols, build a new
    /// set of symbols/codes that optimizes the gain over the distribution in `counter`.
    fn optimize(&self, counters: &Counter, include_ascii: bool) -> Self {
        let mut res = Compressor::default();
        let mut pqueue = BinaryHeap::with_capacity(65_536);
        for code1 in 0u16..(256u16 + self.n_symbols as u16) {
            let symbol1 = self.symbols[code1 as usize];
            let mut gain = counters.count1(code1) * symbol1.len();
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

            for code2 in 0u16..(256u16 + self.n_symbols as u16) {
                let symbol2 = &self.symbols[code2 as usize];
                // If either symbol is zero-length, or if merging would yield a symbol of
                // length greater than 8, skip.
                if symbol1.len() + symbol2.len() > 8 {
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

        // Pop the 255 best symbols.
        let mut n_symbols = 0;
        while !pqueue.is_empty() && n_symbols < 255 {
            let candidate = pqueue.pop().unwrap();
            if res.insert(candidate.symbol) {
                n_symbols += 1;
            }
        }

        // If there are leftover slots, fill them with ASCII chars.
        // This helps reduce the number of escapes.
        //
        // Note that because of the lossy hash table, we won't accidentally
        // save the same ASCII character twice into the table.
        if include_ascii {
            for character in
                " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ[](){}:?/<>".bytes()
            {
                if n_symbols == 255 {
                    break;
                }

                if res.insert(Symbol::from_u8(character)) {
                    n_symbols += 1
                }
            }
        }

        res
    }
}

/// A candidate for inclusion in a symbol table.
///
/// This is really only useful for the `optimize` step of training.
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

    use crate::{Compressor, ESCAPE_CODE};

    #[test]
    fn test_builder() {
        // Train a Compressor on the toy string
        let text = "hello world";
        let table = Compressor::train(text.as_bytes());

        // Use the table to compress a string, see the values
        let compressed = table.compress(text.as_bytes());

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
}
