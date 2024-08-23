#![doc = include_str!("../README.md")]
#![cfg(target_endian = "little")]

/// Throw a compiler error if a type isn't guaranteed to have a specific size in bytes.
macro_rules! assert_sizeof {
    ($typ:ty => $size_in_bytes:expr) => {
        const _: [u8; $size_in_bytes] = [0; std::mem::size_of::<$typ>()];
    };
}

use std::fmt::{Debug, Formatter};

use lossy_pht::LossyPHT;

mod builder;
mod lossy_pht;

/// `Symbol`s are small (up to 8-byte) segments of strings, stored in a [`Compressor`][`crate::Compressor`] and
/// identified by an 8-bit code.
#[derive(Copy, Clone)]
pub struct Symbol(u64);

assert_sizeof!(Symbol => 8);

impl Symbol {
    /// Zero value for `Symbol`.
    pub const ZERO: Self = Self::zero();

    /// Constructor for a `Symbol` from an 8-element byte slice.
    pub fn from_slice(slice: &[u8; 8]) -> Self {
        let num: u64 = slice[0] as u64
            | (slice[1] as u64) << 8
            | (slice[2] as u64) << 16
            | (slice[3] as u64) << 24
            | (slice[4] as u64) << 32
            | (slice[5] as u64) << 40
            | (slice[6] as u64) << 48
            | (slice[7] as u64) << 56;

        Self(num)
    }

    /// Return a zero symbol
    const fn zero() -> Self {
        Self(0)
    }

    /// Create a new single-byte symbol
    pub fn from_u8(value: u8) -> Self {
        Self(value as u64)
    }
}

impl Symbol {
    /// Calculate the length of the symbol in bytes. Always a value between 1 and 8.
    ///
    /// Each symbol has the capacity to hold up to 8 bytes of data, but the symbols
    /// can contain fewer bytes, padded with 0x00. There is a special case of a symbol
    /// that holds the byte 0x00. In that case, the symbol contains `0x0000000000000000`
    /// but we want to interpret that as a one-byte symbol containing `0x00`.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(self) -> usize {
        let numeric = self.0;
        // For little-endian platforms, this counts the number of *trailing* zeros
        let null_bytes = (numeric.leading_zeros() >> 3) as usize;

        // Special case handling of a symbol with all-zeros. This is actually
        // a 1-byte symbol containing 0x00.
        let len = size_of::<Self>() - null_bytes;
        if len == 0 {
            1
        } else {
            len
        }
    }

    #[inline]
    fn as_u64(self) -> u64 {
        self.0
    }

    /// Get the first byte of the symbol as a `u8`.
    ///
    /// If the symbol is empty, this will return the zero byte.
    #[inline]
    pub fn first_byte(self) -> u8 {
        self.0 as u8
    }

    /// Get the first two bytes of the symbol as a `u16`.
    ///
    /// If the Symbol is one or zero bytes, this will return `0u16`.
    #[inline]
    pub fn first_two_bytes(self) -> u16 {
        self.0 as u16
    }

    /// Return a new `Symbol` by logically concatenating ourselves with another `Symbol`.
    pub fn concat(self, other: Self) -> Self {
        let self_len = self.len();
        let new_len = self_len + other.len();
        debug_assert!(new_len <= 8, "cannot build symbol with length > 8");

        Self(other.0 << (8 * self_len) | self.0)
    }
}

impl Debug for Symbol {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let slice = &self.0.to_le_bytes()[0..self.len()];
        let debug = slice
            .iter()
            .map(|c| *c as char)
            .map(|c| {
                if c.is_ascii() {
                    format!("{c}")
                } else {
                    format!("{c:X?}")
                }
            })
            .collect::<Vec<String>>();
        write!(f, "{:?}", debug)
    }
}

/// Code and associated metadata fro a symbol.
///
/// Logically, codes can range from 0-255 inclusive. This type holds both the 8-bit code as well as
/// other metadata bit-packed into a `u16`.
///
/// The bottom 8 bits contain EITHER a code for a symbol stored in the table, OR a raw byte.
///
/// The interpretation depends on the 9th bit: when toggled off, the value stores a raw byte, and when
/// toggled on, it stores a code. Thus if you examine the bottom 9 bits of the `u16`, you have an extended
/// code range, where the values 0-255 are raw bytes, and the values 256-510 represent codes 0-254. 511 is
/// a placeholder for the invalid code here.
///
/// Bits 12-15 store the length of the symbol (values ranging from 0-8).
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CodeMeta(u16);

/// Code used to indicate bytes that are not in the symbol table.
///
/// When compressing a string that cannot fully be expressed with the symbol table, the compressed
/// output will contain an `ESCAPE` byte followed by a raw byte. At decompression time, the presence
/// of `ESCAPE` indicates that the next byte should be appended directly to the result instead of
/// being looked up in the symbol table.
pub const ESCAPE_CODE: u8 = 255;

/// Maximum value for the extended code range.
///
/// When truncated to u8 this is code 255, which is equivalent to [`ESCAPE_CODE`].
pub const MAX_CODE: u16 = 511;

#[allow(clippy::len_without_is_empty)]
impl CodeMeta {
    const EMPTY: Self = CodeMeta(MAX_CODE);

    fn new(code: u8, escape: bool, len: u16) -> Self {
        let value = (len << 12) | ((escape as u16) << 8) | (code as u16);
        Self(value)
    }

    /// Create a new code from a [`Symbol`].
    fn new_symbol(code: u8, symbol: Symbol) -> Self {
        debug_assert_ne!(code, ESCAPE_CODE, "ESCAPE_CODE cannot be used for symbol");

        Self::new(code, false, symbol.len() as u16)
    }

    #[inline]
    fn code(self) -> u8 {
        self.0 as u8
    }

    #[inline]
    fn extended_code(self) -> u16 {
        self.0 & 0b111_111_111
    }

    #[inline]
    fn len(self) -> u16 {
        self.0 >> 12
    }
}

impl Debug for CodeMeta {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodeMeta")
            .field("code", &(self.0 as u8))
            .field("is_escape", &(self.0 < 256))
            .field("len", &(self.0 >> 12))
            .finish()
    }
}

/// Decompressor uses a symbol table to take a stream of 8-bit codes into a string.
#[derive(Clone)]
pub struct Decompressor<'a> {
    /// Table mapping codes to symbols.
    ///
    /// The first 256 slots are escapes. The following slots (up to 254)
    /// are for symbols with actual codes.
    ///
    /// This physical layout is important so that we can do straight-line execution in the decompress method.
    pub(crate) symbols: &'a [Symbol],
}

impl<'a> Decompressor<'a> {
    /// Returns a new decompressor that uses the provided symbol table.
    ///
    /// # Panics
    ///
    /// If the provided symbol table has length greater than [`MAX_CODE`].
    pub fn new(symbols: &'a [Symbol]) -> Self {
        assert!(
            symbols.len() <= MAX_CODE as usize,
            "symbol table cannot have size exceeding MAX_CODE"
        );

        Self { symbols }
    }

    /// Decompress a byte slice that was previously returned by a compressor using
    /// the same symbol table.
    pub fn decompress(&self, compressed: &[u8]) -> Vec<u8> {
        let mut decoded: Vec<u8> = Vec::with_capacity(size_of::<Symbol>() * (compressed.len() + 1));
        let ptr = decoded.as_mut_ptr();

        let mut in_pos = 0;
        let mut out_pos = 0;

        while in_pos < compressed.len() && out_pos < (decoded.capacity() - size_of::<Symbol>()) {
            let code = compressed[in_pos];
            if code == ESCAPE_CODE {
                // Advance by one, do raw write.
                in_pos += 1;
                // SAFETY: out_pos is always 8 bytes or more from the end of decoded buffer
                unsafe {
                    let write_addr = ptr.byte_offset(out_pos as isize);
                    write_addr.write(compressed[in_pos]);
                }
                out_pos += 1;
                in_pos += 1;
            } else {
                let symbol = self.symbols[256 + code as usize];
                // SAFETY: out_pos is always 8 bytes or more from the end of decoded buffer
                unsafe {
                    let write_addr = ptr.byte_offset(out_pos as isize) as *mut u64;
                    // Perform 8 byte unaligned write.
                    write_addr.write_unaligned(symbol.as_u64());
                }
                in_pos += 1;
                out_pos += symbol.len();
            }
        }

        debug_assert!(
            in_pos >= compressed.len(),
            "decompression should exhaust input before output"
        );

        // SAFETY: we enforce in the loop condition that out_pos <= decoded.capacity()
        unsafe { decoded.set_len(out_pos) };

        decoded
    }
}

/// A compressor that uses a symbol table to greedily compress strings.
///
/// The `Compressor` is the central component of FSST. You can create a compressor either by
/// default (i.e. an empty compressor), or by [training][`Self::train`] it on an input corpus of text.
///
/// Example usage:
///
/// ```
/// use fsst::{Symbol, Compressor};
/// let mut compressor = Compressor::default();
///
/// // Insert a new symbol
/// assert!(compressor.insert(Symbol::from_slice(&[b'h', b'e', b'l', b'l', b'o', 0, 0, 0])));
///
/// let compressed = compressor.compress("hello".as_bytes());
/// assert_eq!(compressed, vec![0u8]);
/// ```
#[derive(Clone)]
pub struct Compressor {
    /// Table mapping codes to symbols.
    pub(crate) symbols: Vec<Symbol>,

    /// The number of entries in the symbol table that have been populated, not counting
    /// the escape values.
    pub(crate) n_symbols: u8,

    /// Inverted index mapping 2-byte symbols to codes
    codes_twobyte: Vec<CodeMeta>,

    /// Lossy perfect hash table for looking up codes to symbols that are 3 bytes or more
    lossy_pht: LossyPHT,
}

impl Default for Compressor {
    fn default() -> Self {
        // NOTE: `vec!` has a specialization for building a new vector of `0u64`. Because Symbol and u64
        //  have the same bit pattern, we can allocate as u64 and transmute. If we do `vec![Symbol::EMPTY; N]`,
        // that will create a new Vec and call `Symbol::EMPTY.clone()` `N` times which is considerably slower.
        let symbols = vec![0u64; 511];
        // SAFETY: transmute safety assured by the compiler.
        let symbols: Vec<Symbol> = unsafe { std::mem::transmute(symbols) };
        let mut table = Self {
            symbols,
            n_symbols: 0,
            codes_twobyte: vec![CodeMeta::EMPTY; 65_536],
            lossy_pht: LossyPHT::new(),
        };

        // Populate the escape byte entries.
        for byte in 0..=255 {
            table.symbols[byte as usize] = Symbol::from_u8(byte);
        }

        table
    }
}

impl Compressor {
    #[inline]
    fn get_twobyte(&self, code: u16) -> CodeMeta {
        unsafe { *self.codes_twobyte.get_unchecked(code as usize) }
    }

    #[inline]
    fn put_twobyte(&mut self, code: u16, code_meta: CodeMeta) {
        unsafe {
            let ptr = self.codes_twobyte.get_unchecked_mut(code as usize);
            *ptr = code_meta
        }
    }
}

/// The core structure of the FSST codec, holding a mapping between `Symbol`s and `Code`s.
///
/// The symbol table is trained on a corpus of data in the form of a single byte array, building up
/// a mapping of 1-byte "codes" to sequences of up to `N` plaintext bytse, or "symbols".
impl Compressor {
    /// Attempt to insert a new symbol at the end of the table.
    ///
    /// # Panics
    /// Panics if the table is already full.
    pub fn insert(&mut self, symbol: Symbol) -> bool {
        assert!(self.n_symbols < 255, "cannot insert into full symbol table");

        let symbol_len = symbol.len();
        if symbol_len <= 2 {
            // Insert the 2-byte symbol into the twobyte cache
            self.put_twobyte(
                symbol.first_two_bytes(),
                CodeMeta::new_symbol(self.n_symbols, symbol),
            );
        } else {
            // Attempt to insert larger symbols into the 3-byte cache
            if !self.lossy_pht.insert(symbol, self.n_symbols) {
                return false;
            }
        }

        // Insert at the end of the symbols table.
        // Note the rescaling from range [0-254] -> [256, 510].
        unsafe {
            *self.symbols.as_mut_ptr().add(256 + self.n_symbols as usize) = symbol;
        }
        // self.symbols[256 + (self.n_symbols as usize)] = symbol;
        self.n_symbols += 1;
        true
    }

    /// Using the symbol table, runs a single cycle of compression on an input word, writing
    /// the output into `out_ptr`.
    ///
    /// # Returns
    ///
    /// This function returns a tuple of (advance_in, advance_out) with the number of bytes
    /// for the caller to advance the input and output pointers.
    ///
    /// `advance_in` is the number of bytes to advance the input pointer before the next call.
    ///
    /// `advance_out` is the number of bytes to advance `out_ptr` before the next call.
    ///
    /// # Safety
    ///
    /// `out_ptr` must never be NULL or otherwise point to invalid memory.
    #[inline]
    pub unsafe fn compress_word(&self, word: u64, out_ptr: *mut u8) -> (usize, usize) {
        // Speculatively write the first byte of `word` at offset 1. This is necessary if it is an escape, and
        // if it isn't, it will be overwritten anyway.
        //
        // SAFETY: caller ensures out_ptr is not null
        let first_byte = word as u8;
        unsafe { out_ptr.byte_add(1).write_unaligned(first_byte) };

        // Probe the hash table
        let entry = self.lossy_pht.lookup(word);

        // Now, downshift the `word` and the `entry` to see if they align.
        let ignored_bits = entry.ignored_bits;

        if !compare_masked(word, entry.symbol.as_u64(), ignored_bits) || entry.is_unused() {
            // lookup the appropriate code for the twobyte sequence and write it
            // This will hold either 511, OR it will hold the actual code.
            let code = self.get_twobyte(word as u16);
            let out = code.code();
            unsafe {
                out_ptr.write(out);
            }

            // Advance the input by one byte and the output by 1 byte (if real code) or 2 bytes (if escape).
            return (
                if out == ESCAPE_CODE {
                    1
                } else {
                    code.len() as usize
                },
                if out == ESCAPE_CODE { 2 } else { 1 },
            );
        }

        let code = entry.code;
        unsafe {
            out_ptr.write_unaligned(code.code());
        }

        (code.len() as usize, 1)
    }

    /// Compress many lines in bulk.
    pub fn compress_bulk(&self, lines: &Vec<&[u8]>) -> Vec<Vec<u8>> {
        let mut res = Vec::new();

        for line in lines {
            res.push(self.compress(line));
        }

        res
    }

    /// Use the symbol table to compress the plaintext into a sequence of codes and escapes.
    pub fn compress(&self, plaintext: &[u8]) -> Vec<u8> {
        if plaintext.is_empty() {
            return Vec::new();
        }

        let mut values: Vec<u8> = Vec::with_capacity(2 * plaintext.len());

        let mut in_ptr = plaintext.as_ptr();
        let mut out_ptr = values.as_mut_ptr();

        // SAFETY: `end` will point just after the end of the `plaintext` slice.
        let in_end = unsafe { in_ptr.byte_add(plaintext.len()) };
        let in_end_sub8 = in_end as usize - 8;
        // SAFETY: `end` will point just after the end of the `values` allocation.
        let out_end = unsafe { out_ptr.byte_add(values.capacity()) };

        while (in_ptr as usize) < in_end_sub8 && out_ptr < out_end {
            // SAFETY: pointer ranges are checked in the loop condition
            unsafe {
                // Load a full 8-byte word of data from in_ptr.
                // SAFETY: caller asserts in_ptr is not null. we may read past end of pointer though.
                let word: u64 = std::ptr::read_unaligned(in_ptr as *const u64);
                let (advance_in, advance_out) = self.compress_word(word, out_ptr);
                in_ptr = in_ptr.byte_add(advance_in);
                out_ptr = out_ptr.byte_add(advance_out);
            };
        }

        let remaining_bytes = unsafe { in_end.byte_offset_from(in_ptr) };
        assert!(
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

        while in_ptr < in_end && out_ptr < out_end {
            unsafe {
                // Load a full 8-byte word of data from in_ptr.
                // SAFETY: caller asserts in_ptr is not null. we may read past end of pointer though.
                let (advance_in, advance_out) = self.compress_word(last_word, out_ptr);
                in_ptr = in_ptr.byte_add(advance_in);
                out_ptr = out_ptr.byte_add(advance_out);

                last_word = advance_8byte_word(last_word, advance_in);
            }
        }

        // in_ptr should have exceeded in_end
        assert!(in_ptr >= in_end, "exhausted output buffer before exhausting input, there is a bug in SymbolTable::compress()");

        // Count the number of bytes written
        // SAFETY: assertion
        unsafe {
            let bytes_written = out_ptr.offset_from(values.as_ptr());
            assert!(
                bytes_written.is_positive(),
                "out_ptr ended before it started, not possible"
            );

            values.set_len(bytes_written as usize);
        }

        values
    }

    /// Access the decompressor that can be used to decompress strings emitted from this
    /// `Compressor` instance.
    pub fn decompressor(&self) -> Decompressor {
        Decompressor::new(self.symbol_table())
    }

    /// Returns a readonly slice of the current symbol table.
    ///
    /// The returned slice will have length of `256 + n_symbols`.
    pub fn symbol_table(&self) -> &[Symbol] {
        unsafe { std::slice::from_raw_parts(self.symbols.as_ptr(), 256 + self.n_symbols as usize) }
    }
}

#[inline]
pub(crate) fn advance_8byte_word(word: u64, bytes: usize) -> u64 {
    // shift the word off the low-end, because little endian means the first
    // char is stored in the LSB.
    //
    // Note that even though this looks like it branches, Rust compiles this to a
    // conditional move instruction. See `<https://godbolt.org/z/Pbvre65Pq>`
    if bytes == 8 {
        0
    } else {
        word >> (8 * bytes)
    }
}

#[inline]
pub(crate) fn compare_masked(left: u64, right: u64, ignored_bits: u16) -> bool {
    let mask = if ignored_bits == 64 {
        0
    } else {
        u64::MAX >> ignored_bits
    };

    (left & mask) == right
}

/// This is a function that will get monomorphized based on the value of `N` to do
/// a load of `N` values from the pointer in a minimum number of instructions into
/// an output `u64`.
#[inline]
pub(crate) unsafe fn extract_u64<const N: usize>(ptr: *const u8) -> u64 {
    match N {
        1 => std::ptr::read(ptr) as u64,
        2 => std::ptr::read_unaligned(ptr as *const u16) as u64,
        3 => {
            let low = std::ptr::read(ptr) as u64;
            let high = std::ptr::read_unaligned(ptr.byte_add(1) as *const u16) as u64;
            high << 8 | low
        }
        4 => std::ptr::read_unaligned(ptr as *const u32) as u64,
        5 => {
            let low = std::ptr::read_unaligned(ptr as *const u32) as u64;
            let high = ptr.byte_add(4).read() as u64;
            high << 32 | low
        }
        6 => {
            let low = std::ptr::read_unaligned(ptr as *const u32) as u64;
            let high = std::ptr::read_unaligned(ptr.byte_add(4) as *const u16) as u64;
            high << 32 | low
        }
        7 => {
            let low = std::ptr::read_unaligned(ptr as *const u32) as u64;
            let mid = std::ptr::read_unaligned(ptr.byte_add(4) as *const u16) as u64;
            let high = std::ptr::read(ptr.byte_add(6)) as u64;
            (high << 48) | (mid << 32) | low
        }
        8 => std::ptr::read_unaligned(ptr as *const u64),
        _ => unreachable!("N must be <= 8"),
    }
}
