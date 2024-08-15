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
        for code in 256..(256 + self.n_symbols as u16) {
            let symbol = &self.symbols[code as usize];
            if symbol.is_prefix(text) && symbol.len() > best_overlap {
                best_code = code;
                best_overlap = symbol.len();
            }
        }

        best_code
    }
}
