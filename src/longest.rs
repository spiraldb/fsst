use crate::{Code, SymbolTable};

/// Find the longest substring.

impl SymbolTable {
    // NOTE(aduffy): if you don't disable inlining, this function won't show up in profiles.
    #[inline(never)]
    pub(crate) fn find_longest_symbol(&self, text: &[u8]) -> Code {
        debug_assert!(!text.is_empty(), "text must not be empty");

        // Find the code that best maps to the provided text table here.
        let mut best_code = Code::new_escaped(text[0]);
        let mut best_overlap = 1;
        for code in 0..511 {
            let symbol = &self.symbols[code as usize];
            if symbol.is_prefix(text) && symbol.len() > best_overlap {
                best_code = Code::from_u16(code);
                best_overlap = symbol.len();
            }
        }

        best_code
    }
}
