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

use crate::find_longest::FindLongestSymbol;
use crate::SymbolTable;

// Find the code that maps to a symbol with longest-match to a piece of text.
//
// This is the naive algorithm that just scans the whole table and is very slow.

impl FindLongestSymbol for SymbolTable {
    // NOTE(aduffy): if you don't disable inlining, this function won't show up in profiles.
    #[inline(never)]
    fn find_longest_symbol(&self, text: &[u8]) -> u16 {
        debug_assert!(!text.is_empty(), "text must not be empty");

        // Find the code that best maps to the provided text table here.
        // Start with the code corresponding to the escape of the first character in the text
        let mut best_code = text[0] as u16;
        let mut best_overlap = 1;
        for code in 256..511 {
            let symbol = &self.symbols[code as usize];
            if symbol.is_prefix(text) && symbol.len() > best_overlap {
                best_code = code;
                best_overlap = symbol.len();
            }
        }

        best_code
    }
}
