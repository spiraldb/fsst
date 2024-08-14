// Copyright 2024 Spiral, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Functions and types used for building a [`SymbolTable`] from a corpus of text.
//!
//! This module implements the logic from Algorithm 3 of the [FSST Paper].
//!
//! [FSST Paper]: https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::find_longest::FindLongestSymbol;
use crate::{CodeMeta, Symbol, SymbolTable, MAX_CODE};

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
pub const MAX_GENERATIONS: usize = 5;

/// Build and train a `SymbolTable` from a sample corpus of text.
///
/// This function implements the generational algorithm described in the [FSST paper] Section
/// 4.3. Starting with an empty symbol table, it iteratively compresses the corpus, then attempts
/// to merge symbols when doing so would yield better compression than leaving them unmerged. The
/// resulting table will have at most 255 symbols (the 256th symbol is reserved for the escape
/// code).
///
/// [FSST paper]: https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf
pub fn train(corpus: impl AsRef<[u8]>) -> SymbolTable {
    let mut table = SymbolTable::default();
    // TODO(aduffy): handle truncating/sampling if corpus > requires sample size.
    let sample = corpus.as_ref();
    for _generation in 0..MAX_GENERATIONS {
        let counter = table.compress_count(sample);
        table = table.optimize(counter);
    }

    table
}

impl SymbolTable {
    /// Compress the text using the current symbol table. Count the code occurrences
    /// and code-pair occurrences to allow us to calculate apparent gain.
    fn compress_count(&self, sample: &[u8]) -> Counter {
        let mut counter = Counter::new();
        let len = sample.len();
        let mut prev_code = self.find_longest_symbol(sample);
        counter.record_count1(prev_code);
        let mut pos = self.symbols[prev_code as usize].len();

        while pos < len {
            let code = self.find_longest_symbol(&sample[pos..len]);
            counter.record_count1(code);
            counter.record_count2(prev_code, code);
            pos += self.symbols[code as usize].len();
            prev_code = code;
        }

        counter
    }

    /// Using a set of counters and the existing set of symbols, build a new
    /// set of symbols/codes that optimizes the gain over the distribution in `counter`.
    fn optimize(&self, counters: Counter) -> Self {
        let mut res = SymbolTable::default();
        let mut pqueue = BinaryHeap::new();
        for code1 in 0..511 {
            let symbol1 = self.symbols[code1 as usize];
            let gain = counters.count1(code1) * symbol1.len();
            pqueue.push(Candidate {
                symbol: symbol1,
                gain,
            });

            for code2 in 0..511 {
                let symbol2 = &self.symbols[code2 as usize];
                // If either symbol is zero-length, or if merging would yield a symbol of
                // length greater than 8, skip.
                if symbol1.len() + symbol2.len() >= 8 || symbol1.is_empty() || symbol2.is_empty() {
                    continue;
                }
                let new_symbol = symbol1.concat(symbol2);
                // as`sert the symbol is not empty
                assert!(
                    !new_symbol.is_empty(),
                    "symbol made by merging {:?} and {:?} is empty",
                    symbol1,
                    symbol2,
                );
                let gain = counters.count2(code1, code2);
                pqueue.push(Candidate {
                    symbol: new_symbol,
                    gain,
                })
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
    use crate::{train, ESCAPE_CODE};

    #[test]
    fn test_builder() {
        // Train a SymbolTable on the toy string
        let text = "hello world";
        let table = train(text.as_bytes());

        // Use the table to compress a string, see the values
        let compressed = table.compress(text.as_bytes());
        assert_eq!(compressed, vec![0u8, 1u8, 2u8]);

        // Ensure that the compressed string has no escape bytes
        assert!(compressed.iter().all(|b| *b != ESCAPE_CODE));

        // Ensure that we can compress a string with no values seen at training time.
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
