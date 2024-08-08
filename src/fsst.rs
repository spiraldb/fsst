use std::cmp::min;

const FSST_CODE_MAX: u16 = 256;
const FSST_CODE_MASK: u16 = FSST_CODE_MAX - 1;
const FSST_LEN_BITS: u32 = 12;
const FSST_CODE_BITS: u32 = 9;
const FSST_CODE_BASE: u16 = 256;
const FSST_HASH_LOG2SIZE: usize = 10;
const FSST_HASH_PRIME: u64 = 2971215073;
const FSST_SHIFT: u32 = 15;
const FSST_ICL_FREE: u64 = (15 << 28) | ((FSST_CODE_MASK as u64) << 16);
const FSST_MAXHEADER: usize = 8 + 1 + 8 + 2048 + 1;
const FSST_ESC: u8 = 255;

#[inline(always)]
fn fsst_unaligned_load(v: &[u8]) -> u64 {
    let mut ret: u64 = 0;
    unsafe {
        std::ptr::copy_nonoverlapping(v.as_ptr(), &mut ret as *mut u64 as *mut u8, 8);
    }
    ret
}

#[inline(always)]
fn fsst_hash(w: u64) -> u64 {
    ((w * FSST_HASH_PRIME) ^ ((w * FSST_HASH_PRIME) >> FSST_SHIFT))
}

#[derive(Clone, Copy)]
struct Symbol {
    val: [u8; 8],
    icl: u64,
}

impl Symbol {
    const MAX_LENGTH: usize = 8;

    fn new() -> Self {
        Symbol { val: [0; 8], icl: 0 }
    }

    fn from_byte(c: u8, code: u16) -> Self {
        let mut s = Symbol::new();
        s.val[0] = c;
        s.set_code_len(code, 1);
        s
    }

    fn from_slice(input: &[u8]) -> Self {
        let mut s = Symbol::new();
        let len = min(input.len(), Self::MAX_LENGTH);
        s.val[..len].copy_from_slice(&input[..len]);
        s.set_code_len(FSST_CODE_MASK, len as u32);
        s
    }

    fn set_code_len(&mut self, code: u16, len: u32) {
        self.icl = (len << 28) as u64 | (code as u64) << 16 | ((8 - len) * 8) as u64;
    }

    fn length(&self) -> u32 {
        (self.icl >> 28) as u32
    }

    fn code(&self) -> u16 {
        ((self.icl >> 16) & FSST_CODE_MASK as u64) as u16
    }

    fn ignored_bits(&self) -> u32 {
        self.icl as u32
    }

    fn first(&self) -> u8 {
        self.val[0]
    }

    fn first2(&self) -> u16 {
        u16::from_le_bytes([self.val[0], self.val[1]])
    }

    fn hash(&self) -> usize {
        let v = u32::from_le_bytes([self.val[0], self.val[1], self.val[2], self.val[3]]);
        fsst_hash(v as u64) as usize
    }
}

struct SymbolTable {
    short_codes: [u16; 65536],
    byte_codes: [u16; 256],
    symbols: Vec<Symbol>,
    hash_tab: Vec<Symbol>,
    n_symbols: u16,
    suffix_lim: u16,
    terminator: u16,
    zero_terminated: bool,
    len_histo: [u16; FSST_CODE_BITS as usize],
}

impl SymbolTable {
    fn new() -> Self {
        let mut st = SymbolTable {
            short_codes: [0; 65536],
            byte_codes: [0; 256],
            symbols: vec![Symbol::new(); FSST_CODE_MAX as usize],
            hash_tab: vec![Symbol::new(); 1 << FSST_HASH_LOG2SIZE],
            n_symbols: 0,
            suffix_lim: FSST_CODE_MAX,
            terminator: 0,
            zero_terminated: false,
            len_histo: [0; FSST_CODE_BITS as usize],
        };

        for i in 0..256 {
            st.symbols[i] = Symbol::from_byte(i as u8, i as u16 | (1 << FSST_LEN_BITS));
        }

        for i in 256..FSST_CODE_MAX as usize {
            st.symbols[i] = Symbol::from_byte(0, FSST_CODE_MASK);
        }

        for i in 0..256 {
            st.byte_codes[i] = (1 << FSST_LEN_BITS) | i as u16;
        }

        for i in 0..65536 {
            st.short_codes[i] = (1 << FSST_LEN_BITS) | (i & 255) as u16;
        }

        st
    }

    fn clear(&mut self) {
        self.len_histo = [0; FSST_CODE_BITS as usize];
        for i in FSST_CODE_BASE as usize..FSST_CODE_BASE as usize + self.n_symbols as usize {
            let symbol = &self.symbols[i];
            if symbol.length() == 1 {
                let val = symbol.first();
                self.byte_codes[val as usize] = (1 << FSST_LEN_BITS) | val as u16;
            } else if symbol.length() == 2 {
                let val = symbol.first2();
                self.short_codes[val as usize] = (1 << FSST_LEN_BITS) | (val & 255);
            } else {
                let idx = symbol.hash() & ((1 << FSST_HASH_LOG2SIZE) - 1);
                self.hash_tab[idx] = Symbol::new();
                self.hash_tab[idx].icl = FSST_ICL_FREE;
            }
        }
        self.n_symbols = 0;
    }

    fn hash_insert(&mut self, s: Symbol) -> bool {
        let idx = s.hash() & ((1 << FSST_HASH_LOG2SIZE) - 1);
        let taken = self.hash_tab[idx].icl < FSST_ICL_FREE;
        if taken {
            return false;
        }
        self.hash_tab[idx] = s;
        true
    }

    fn add(&mut self, mut s: Symbol) -> bool {
        assert!(FSST_CODE_BASE + self.n_symbols < FSST_CODE_MAX);
        let len = s.length();
        s.set_code_len(FSST_CODE_BASE + self.n_symbols, len);
        if len == 1 {
            self.byte_codes[s.first() as usize] = FSST_CODE_BASE + self.n_symbols + (1 << FSST_LEN_BITS);
        } else if len == 2 {
            self.short_codes[s.first2() as usize] = FSST_CODE_BASE + self.n_symbols + (2 << FSST_LEN_BITS);
        } else if !self.hash_insert(s) {
            return false;
        }
        self.symbols[FSST_CODE_BASE as usize + self.n_symbols as usize] = s;
        self.len_histo[len as usize - 1] += 1;
        self.n_symbols += 1;
        true
    }

    fn find_longest_symbol(&self, s: Symbol) -> u16 {
        let idx = s.hash() & ((1 << FSST_HASH_LOG2SIZE) - 1);
        if self.hash_tab[idx].icl <= s.icl && self.hash_tab[idx].val == s.val {
            return (self.hash_tab[idx].icl >> 16) & FSST_CODE_MASK as u64;
        }
        if s.length() >= 2 {
            let code = self.short_codes[s.first2() as usize] & FSST_CODE_MASK;
            if code >= FSST_CODE_BASE {
                return code;
            }
        }
        self.byte_codes[s.first() as usize] & FSST_CODE_MASK
    }

    fn find_longest_symbol_slice(&self, cur: &[u8], end: &[u8]) -> u16 {
        self.find_longest_symbol(Symbol::from_slice(&cur[..min(cur.len(), end.len())]))
    }
}

struct Counters {
    count1: Vec<u16>,
    count2: Vec<Vec<u16>>,
}

impl Counters {
    fn new() -> Self {
        Counters {
            count1: vec![0; FSST_CODE_MAX as usize],
            count2: vec![vec![0; FSST_CODE_MAX as usize]; FSST_CODE_MAX as usize],
        }
    }

    fn count1_set(&mut self, pos1: usize, val: u16) {
        self.count1[pos1] = val;
    }

    fn count1_inc(&mut self, pos1: usize) {
        self.count1[pos1] += 1;
    }

    fn count2_inc(&mut self, pos1: usize, pos2: usize) {
        self.count2[pos1][pos2] += 1;
    }

    fn count1_get_next(&self, pos1: &mut usize) -> u32 {
        self.count1[*pos1] as u32
    }

    fn count2_get_next(&self, pos1: usize, pos2: &mut usize) -> u32 {
        self.count2[pos1][*pos2] as u32
    }

    fn backup1(&self, buf: &mut [u8]) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.count1.as_ptr() as *const u8,
                buf.as_mut_ptr(),
                FSST_CODE_MAX as usize * std::mem::size_of::<u16>(),
            );
        }
    }

    fn restore1(&mut self, buf: &[u8]) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf.as_ptr(),
                self.count1.as_mut_ptr() as *mut u8,
                FSST_CODE_MAX as usize * std::mem::size_of::<u16>(),
            );
        }
    }
}

struct Encoder {
    symbol_table: SymbolTable,
    counters: Counters,
}

impl Encoder {
    fn new() -> Self {
        Encoder {
            symbol_table: SymbolTable::new(),
            counters: Counters::new(),
        }
    }

    pub fn compress(&self, input: &[u8], output: &mut [u8]) -> (usize, usize) {
        let mut in_pos = 0;
        let mut out_pos = 0;

        while in_pos < input.len() && out_pos < output.len() {
            let symbol = self.symbol_table.find_longest_symbol_slice(&input[in_pos..], &input[input.len()..]);
            let code = symbol & FSST_CODE_MASK;
            let len = (symbol >> FSST_LEN_BITS) as usize;

            if code < FSST_CODE_BASE {
                // Escape byte
                if out_pos + 2 > output.len() {
                    break;
                }
                output[out_pos] = FSST_ESC;
                output[out_pos + 1] = input[in_pos];
                out_pos += 2;
                in_pos += 1;
            } else {
                if out_pos + 1 > output.len() {
                    break;
                }
                output[out_pos] = code as u8;
                out_pos += 1;
                in_pos += len;
            }
        }

        (in_pos, out_pos)
    }
}

impl SymbolTable {
    pub fn decompress(&self, input: &[u8], output: &mut [u8]) -> usize {
        let mut in_pos = 0;
        let mut out_pos = 0;

        while in_pos < input.len() && out_pos < output.len() {
            let code = input[in_pos] as u16;
            in_pos += 1;

            if code == FSST_ESC as u16 {
                if in_pos >= input.len() {
                    break;
                }
                output[out_pos] = input[in_pos];
                in_pos += 1;
                out_pos += 1;
            } else {
                let symbol = &self.symbols[code as usize];
                let len = symbol.length() as usize;
                if out_pos + len > output.len() {
                    break;
                }
                output[out_pos..out_pos + len].copy_from_slice(&symbol.val[..len]);
                out_pos += len;
            }
        }

        out_pos
    }
}