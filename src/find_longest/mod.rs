mod naive;

pub trait FindLongestSymbol {
    fn find_longest_symbol(&self, text: &[u8]) -> u16;
}
