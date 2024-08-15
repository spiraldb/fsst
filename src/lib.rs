#![doc = include_str!("../README.md")]
#![cfg(target_endian = "little")]

/// Throw a compiler error if a type isn't guaranteed to have a specific size in bytes.
macro_rules! assert_sizeof {
    ($typ:ty => $size_in_bytes:expr) => {
        const _: [u8; $size_in_bytes] = [0; std::mem::size_of::<$typ>()];
    };
}

use std::fmt::{Debug, Formatter};

pub use builder::*;
use lossy_pht::LossyPHT;

mod builder;
mod find_longest;
mod lossy_pht;

/// `Symbol`s are small (up to 8-byte) segments of strings, stored in a [`SymbolTable`] and
/// identified by an 8-bit code.
#[derive(Copy, Clone)]
pub union Symbol {
    bytes: [u8; 8],
    num: u64,
}

assert_sizeof!(Symbol => 8);

impl Symbol {
    /// Zero value for `Symbol`.
    pub const ZERO: Self = Self::zero();

    /// Constructor for a `Symbol` from an 8-element byte slice.
    pub fn from_slice(slice: &[u8; 8]) -> Self {
        Self { bytes: *slice }
    }

    /// Return a zero symbol
    const fn zero() -> Self {
        Self { num: 0 }
    }

    /// Create a new single-byte symbol
    pub fn from_u8(value: u8) -> Self {
        Self {
            bytes: [value, 0, 0, 0, 0, 0, 0, 0],
        }
    }
}

impl Symbol {
    /// Calculate the length of the symbol in bytes. Always a value between 1 and 8.
    ///
    /// Each symbol has the capacity to hold up to 8 bytes of data, but the symbols
    /// can contain fewer bytes, padded with 0x00. There is a special case of a symbol
    /// that holds the byte 0x00. In that case, the symbol contains `0x0000000000000000`
    /// but we want to interpret that as a one-byte symbol containing `0x00`.
    pub fn len(&self) -> usize {
        let numeric = unsafe { self.num };
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

    /// Returns true if the symbol does not encode any bytes.
    ///
    /// Note that this should only be true for the zero code.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    fn as_u64(&self) -> u64 {
        // SAFETY: the bytes can always be viewed as a u64
        unsafe { self.num }
    }

    /// Get the first byte of the symbol as a `u8`.
    ///
    /// If the symbol is empty, this will return the zero byte.
    #[inline]
    pub fn first_byte(&self) -> u8 {
        // SAFETY: the bytes can always be viewed as a u64
        unsafe { self.num as u8 }
    }

    /// Get the first two bytes of the symbol as a `u16`.
    ///
    /// If the Symbol is one or zero bytes, this will return `0u16`.
    #[inline]
    pub fn first_two_bytes(&self) -> u16 {
        // SAFETY: the bytes can always be viewed as a u64
        unsafe { self.num as u16 }
    }

    /// Access the Symbol as a slice.
    pub fn as_slice(&self) -> &[u8] {
        let len = self.len();
        // SAFETY: constructors will not allow building a struct where len > 8.
        unsafe { &self.bytes[0..len] }
    }

    /// Returns true if the symbol is a prefix of the provided text.
    pub fn is_prefix(&self, text: &[u8]) -> bool {
        text.starts_with(self.as_slice())
    }

    /// Return a new `Symbol` by logically concatenating ourselves with another `Symbol`.
    pub fn concat(&self, other: &Self) -> Self {
        let self_len = self.len();
        let new_len = self_len + other.len();
        assert!(new_len <= 8, "cannot build symbol with length > 8");

        let mut result = *self;

        // SAFETY: self_len and new_len are checked to be <= 8
        unsafe { result.bytes[self_len..new_len].copy_from_slice(other.as_slice()) };

        result
    }
}

impl Debug for Symbol {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", unsafe { self.bytes })
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
        assert_ne!(code, ESCAPE_CODE, "ESCAPE_CODE cannot be used for symbol");

        Self::new(code, false, symbol.len() as u16)
    }

    #[inline]
    fn code(&self) -> u8 {
        self.0 as u8
    }

    #[inline]
    fn extended_code(&self) -> u16 {
        self.0 & 0b111_111_111
    }

    #[inline]
    fn len(&self) -> u16 {
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

/// The static symbol table used for compression and decompression.
///
/// The `SymbolTable` is the central component of FSST. You can create a SymbolTable either by
/// default, or by [training] it on an input corpus of text.
///
/// Example usage:
///
/// ```
/// use fsst_rs::{Symbol, SymbolTable};
/// let mut table = SymbolTable::default();
///
/// // Insert a new symbol
/// assert!(table.insert(Symbol::from_slice(&[b'h', b'e', b'l', b'l', b'o', 0, 0, 0])));
///
/// let compressed = table.compress("hello".as_bytes());
/// assert_eq!(compressed, vec![0u8]);
/// ```
///
/// [training]: [`train`]
#[derive(Clone)]
pub struct SymbolTable {
    /// Table mapping codes to symbols.
    pub(crate) symbols: [Symbol; 511],

    /// Indicates the number of entries in the symbol table that have been populated, not counting
    /// the escape values.
    pub(crate) n_symbols: u8,

    //
    // Index structures used to speedup building the symbol table and compression
    //
    /// Inverted index mapping 2-byte symbols to codes
    codes_twobyte: Vec<CodeMeta>,

    /// Lossy perfect hash table for looking up codes to symbols that are 3 bytes or more
    lossy_pht: LossyPHT,
}

impl Default for SymbolTable {
    fn default() -> Self {
        let mut table = Self {
            symbols: [Symbol::ZERO; 511],
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

/// The core structure of the FSST codec, holding a mapping between `Symbol`s and `Code`s.
///
/// The symbol table is trained on a corpus of data in the form of a single byte array, building up
/// a mapping of 1-byte "codes" to sequences of up to `N` plaintext bytse, or "symbols".
impl SymbolTable {
    /// Attempt to insert a new symbol at the end of the table.
    ///
    /// # Panics
    /// Panics if the table is already full.
    pub fn insert(&mut self, symbol: Symbol) -> bool {
        assert!(self.n_symbols < 255, "cannot insert into full symbol table");

        let symbol_len = symbol.len();
        if symbol_len <= 2 {
            // Insert the 2-byte symbol into the twobyte cache
            self.codes_twobyte[symbol.first_two_bytes() as usize] =
                CodeMeta::new_symbol(self.n_symbols, symbol);
        } else if symbol_len >= 3 {
            // Attempt to insert larger symbols into the 3-byte cache
            if !self.lossy_pht.insert(symbol, self.n_symbols) {
                return false;
            }
        }

        // Insert at the end of the symbols table.
        // Note the rescaling from range [0-254] -> [256, 510].
        self.symbols[256 + (self.n_symbols as usize)] = symbol;
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
    // NOTE(aduffy): uncomment this line to make the function appear in profiles
    #[inline(never)]
    pub(crate) unsafe fn compress_word(&self, word: u64, out_ptr: *mut u8) -> (usize, usize) {
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
            let code = self.codes_twobyte[(word as u16) as usize];
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
        let in_end_sub8 = unsafe { in_end.byte_sub(8) };
        // SAFETY: `end` will point just after the end of the `values` allocation.
        let out_end = unsafe { out_ptr.byte_add(values.capacity()) };

        while in_ptr < in_end_sub8 && out_ptr < out_end {
            // SAFETY: pointer ranges are checked in the loop condition
            unsafe {
                // Load a full 8-byte word of data from in_ptr.
                // SAFETY: caller asserts in_ptr is not null. we may read past end of pointer though.
                let word: u64 = (in_ptr as *const u64).read_unaligned();
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

        // Shift off the remaining bytes
        let mut last_word = unsafe { (in_ptr as *const u64).read_unaligned() };
        last_word = mask_prefix(last_word, remaining_bytes as usize);

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

    /// Decompress a byte slice that was previously returned by [compression][Self::compress].
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
                    write_addr.write_unaligned(symbol.num);
                }
                in_pos += 1;
                out_pos += symbol.len();
            }
        }

        assert!(
            in_pos >= compressed.len(),
            "decompression should exhaust input before output"
        );

        // SAFETY: we enforce in the loop condition that out_pos <= decoded.capacity()
        unsafe { decoded.set_len(out_pos) };

        decoded
    }
}

/// Mask the word, keeping only the `prefix_bytes` front.
fn mask_prefix(word: u64, prefix_bytes: usize) -> u64 {
    let mask = if prefix_bytes == 0 {
        0
    } else {
        u64::MAX >> (8 * (8 - prefix_bytes))
    };

    word & mask
}

fn advance_8byte_word(word: u64, bytes: usize) -> u64 {
    // shift the word off the right-end, because little endian means the first
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

fn compare_masked(left: u64, right: u64, ignored_bits: u16) -> bool {
    let mask = if ignored_bits == 64 {
        0
    } else {
        u64::MAX >> ignored_bits
    };

    (left & mask) == right
}
