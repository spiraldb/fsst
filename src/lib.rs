#![doc = include_str!("../README.md")]
#![cfg(target_endian = "little")]

/// Throw a compiler error if a type isn't guaranteed to have a specific size in bytes.
macro_rules! assert_sizeof {
    ($typ:ty => $size_in_bytes:expr) => {
        const _: [u8; $size_in_bytes] = [0; std::mem::size_of::<$typ>()];
    };
}

use lossy_pht::LossyPHT;
use std::fmt::{Debug, Formatter};

mod builder;
mod lossy_pht;

pub use builder::*;

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
    pub fn first2(self) -> u16 {
        self.0 as u16
    }

    /// Get the first two bytes of the symbol as a `u16`.
    ///
    /// If the Symbol is one or zero bytes, this will return `0u16`.
    #[inline]
    pub fn first3(self) -> u64 {
        self.0 & 0xFF_FF_FF
    }

    /// Return a new `Symbol` by logically concatenating ourselves with another `Symbol`.
    pub fn concat(self, other: Self) -> Self {
        debug_assert!(
            self.len() + other.len() <= 8,
            "cannot build symbol with length > 8"
        );

        let self_len = self.len();

        Self((other.0 << (8 * self_len)) | self.0)
    }
}

impl Debug for Symbol {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;

        let slice = &self.0.to_le_bytes()[0..self.len()];
        for c in slice.iter().map(|c| *c as char) {
            if ('!'..='~').contains(&c) {
                write!(f, "{c}")?;
            } else if c == '\n' {
                write!(f, " \\n ")?;
            } else if c == '\t' {
                write!(f, " \\t ")?;
            } else if c == ' ' {
                write!(f, " SPACE ")?;
            } else {
                write!(f, " 0x{:X?} ", c as u8)?
            }
        }

        write!(f, "]")
    }
}

/// A packed type containing a code value, as well as metadata about the symbol referred to by
/// the code.
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
struct Code(u16);

/// Code used to indicate bytes that are not in the symbol table.
///
/// When compressing a string that cannot fully be expressed with the symbol table, the compressed
/// output will contain an `ESCAPE` byte followed by a raw byte. At decompression time, the presence
/// of `ESCAPE` indicates that the next byte should be appended directly to the result instead of
/// being looked up in the symbol table.
pub const ESCAPE_CODE: u8 = 255;

/// Number of bits in the `ExtendedCode` that are used to dictate a code value.
pub const FSST_CODE_BITS: usize = 9;

/// First bit of the "length" portion of an extended code.
pub const FSST_LEN_BITS: usize = 12;

/// A code that never appears in practice, indicating an unused slot.
pub const FSST_CODE_UNUSED: u16 = 1u16 << FSST_CODE_BITS;

/// Maximum code value in the extended code range.
pub const FSST_CODE_MAX: u16 = 1 << FSST_CODE_BITS;

/// Maximum value for the extended code range.
///
/// When truncated to u8 this is code 255, which is equivalent to [`ESCAPE_CODE`].
pub const FSST_CODE_MASK: u16 = FSST_CODE_MAX - 1;

/// First code in the symbol table that corresponds to a non-escape symbol.
pub const FSST_CODE_BASE: u16 = 256;

#[allow(clippy::len_without_is_empty)]
impl Code {
    /// Code for an unused slot in a symbol table or index.
    ///
    /// This corresponds to the maximum code with a length of 1.
    pub const UNUSED: Self = Code(FSST_CODE_MASK + (1 << 12));

    /// Create a new code for a symbol of given length.
    fn new_symbol(code: u8, len: usize) -> Self {
        Self(code as u16 + ((len as u16) << FSST_LEN_BITS))
    }

    /// Code for a new symbol during the building phase.
    ///
    /// The code is remapped from 0..254 to 256...510.
    fn new_symbol_building(code: u8, len: usize) -> Self {
        Self(code as u16 + 256 + ((len as u16) << FSST_LEN_BITS))
    }

    /// Create a new code corresponding for an escaped byte.
    fn new_escape(byte: u8) -> Self {
        Self((byte as u16) + (1 << FSST_LEN_BITS))
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
        self.0 >> FSST_LEN_BITS
    }
}

impl Debug for Code {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainingCode")
            .field("code", &(self.0 as u8))
            .field("is_escape", &(self.0 < 256))
            .field("len", &(self.0 >> 12))
            .finish()
    }
}

/// Decompressor uses a symbol table to take a stream of 8-bit codes into a string.
#[derive(Clone)]
pub struct Decompressor<'a> {
    /// Slice mapping codes to symbols.
    pub(crate) symbols: &'a [Symbol],

    /// Slice containing the length of each symbol in the `symbols` slice.
    pub(crate) lengths: &'a [u8],
}

impl<'a> Decompressor<'a> {
    /// Returns a new decompressor that uses the provided symbol table.
    ///
    /// # Panics
    ///
    /// If the provided symbol table has length greater than 256
    pub fn new(symbols: &'a [Symbol], lengths: &'a [u8]) -> Self {
        assert!(
            symbols.len() <= 255,
            "symbol table cannot have size exceeding 255"
        );

        Self { symbols, lengths }
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
                    std::ptr::write(write_addr, compressed[in_pos]);
                }
                out_pos += 1;
                in_pos += 1;
            } else {
                let symbol = self.symbols[code as usize];
                let length = self.lengths[code as usize];
                // SAFETY: out_pos is always 8 bytes or more from the end of decoded buffer
                unsafe {
                    let write_addr = ptr.byte_offset(out_pos as isize) as *mut u64;
                    // Perform 8 byte unaligned write.
                    write_addr.write_unaligned(symbol.as_u64());
                }
                in_pos += 1;
                out_pos += length as usize;
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
/// use fsst::{Symbol, Compressor, CompressorBuilder};
/// let compressor = {
///     let mut builder = CompressorBuilder::new();
///     builder.insert(Symbol::from_slice(&[b'h', b'e', b'l', b'l', b'o', 0, 0, 0]), 5);
///     builder.build()
/// };
///
/// let compressed = compressor.compress("hello".as_bytes());
/// assert_eq!(compressed, vec![0u8]);
/// ```
#[derive(Clone)]
pub struct Compressor {
    /// Table mapping codes to symbols.
    pub(crate) symbols: Vec<Symbol>,

    /// Length of each symbol, values range from 1-8.
    pub(crate) lengths: Vec<u8>,

    /// The number of entries in the symbol table that have been populated, not counting
    /// the escape values.
    pub(crate) n_symbols: u8,

    /// Inverted index mapping 2-byte symbols to codes
    codes_two_byte: Vec<Code>,

    /// Limit of no suffixes.
    has_suffix_code: u8,

    /// Lossy perfect hash table for looking up codes to symbols that are 3 bytes or more
    lossy_pht: LossyPHT,
}

/// The core structure of the FSST codec, holding a mapping between `Symbol`s and `Code`s.
///
/// The symbol table is trained on a corpus of data in the form of a single byte array, building up
/// a mapping of 1-byte "codes" to sequences of up to `N` plaintext bytse, or "symbols".
impl Compressor {
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
    #[inline(never)]
    pub unsafe fn compress_word(&self, word: u64, out_ptr: *mut u8) -> (usize, usize) {
        // Speculatively write the first byte of `word` at offset 1. This is necessary if it is an escape, and
        // if it isn't, it will be overwritten anyway.
        //
        // SAFETY: caller ensures out_ptr is not null
        let first_byte = word as u8;
        out_ptr.byte_add(1).write_unaligned(first_byte);

        // First, check the two_bytes table
        let code_twobyte = self.codes_two_byte[word as u16 as usize];

        if code_twobyte.code() < self.has_suffix_code {
            // 2 byte code without having to worry about longer matches.
            std::ptr::write(out_ptr, code_twobyte.code());

            // Advance input by symbol length (2) and output by a single code byte
            (2, 1)
        } else {
            // Probe the hash table
            let entry = self.lossy_pht.lookup(word);

            // Now, downshift the `word` and the `entry` to see if they align.
            let ignored_bits = entry.ignored_bits;
            if entry.code != Code::UNUSED
                && compare_masked(word, entry.symbol.as_u64(), ignored_bits)
            {
                // Advance the input by the symbol length (variable) and the output by one code byte
                std::ptr::write(out_ptr, entry.code.code());
                (entry.code.len() as usize, 1)
            } else {
                std::ptr::write(out_ptr, code_twobyte.code());

                // Advance the input by the symbol length (variable) and the output by either 1
                // byte (if was one-byte code) or two bytes (escape).
                (
                    code_twobyte.len() as usize,
                    // Predicated version of:
                    //
                    // if entry.code >= 256 {
                    //      2
                    // } else {
                    //      1
                    // }
                    1 + (code_twobyte.extended_code() >> 8) as usize,
                )
            }
        }
    }

    /// Compress many lines in bulk.
    pub fn compress_bulk(&self, lines: &Vec<&[u8]>) -> Vec<Vec<u8>> {
        let mut res = Vec::new();

        for line in lines {
            res.push(self.compress(line));
        }

        res
    }

    /// Compress a string, writing its result into a target buffer.
    ///
    /// The target buffer is a byte vector that must have capacity large enough
    /// to hold the encoded data.
    ///
    /// When this call returns, `values` will hold the compressed bytes and have
    /// its length set to the length of the compresed text.
    ///
    /// ```
    /// use fsst::{Compressor, CompressorBuilder, Symbol};
    ///
    /// let mut compressor = CompressorBuilder::new();
    /// assert!(compressor.insert(Symbol::from_slice(b"aaaaaaaa"), 8));
    ///
    /// let compressor = compressor.build();
    ///
    /// let mut compressed_values = Vec::with_capacity(1_024);
    ///
    /// // SAFETY: we have over-sized compressed_values.
    /// unsafe {
    ///     compressor.compress_into(b"aaaaaaaa", &mut compressed_values);
    /// }
    ///
    /// assert_eq!(compressed_values, vec![0u8]);
    /// ```
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure the provided buffer is large enough to hold
    /// all encoded data.
    pub unsafe fn compress_into(&self, plaintext: &[u8], values: &mut Vec<u8>) {
        let mut in_ptr = plaintext.as_ptr();
        let mut out_ptr = values.as_mut_ptr();

        // SAFETY: `end` will point just after the end of the `plaintext` slice.
        let in_end = unsafe { in_ptr.byte_add(plaintext.len()) };
        let in_end_sub8 = in_end as usize - 8;
        // SAFETY: `end` will point just after the end of the `values` allocation.
        let out_end = unsafe { out_ptr.byte_add(values.capacity()) };

        while (in_ptr as usize) <= in_end_sub8 && out_ptr < out_end {
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
            out_ptr < out_end || remaining_bytes == 0,
            "output buffer sized too small"
        );

        let remaining_bytes = remaining_bytes as usize;

        // Load the last `remaining_byte`s of data into a final world. We then replicate the loop above,
        // but shift data out of this word rather than advancing an input pointer and potentially reading
        // unowned memory.
        let mut bytes = [0u8; 8];
        std::ptr::copy_nonoverlapping(in_ptr, bytes.as_mut_ptr(), remaining_bytes);
        let mut last_word = u64::from_le_bytes(bytes);

        while in_ptr < in_end && out_ptr < out_end {
            // Load a full 8-byte word of data from in_ptr.
            // SAFETY: caller asserts in_ptr is not null. we may read past end of pointer though.
            let (advance_in, advance_out) = self.compress_word(last_word, out_ptr);
            in_ptr = in_ptr.byte_add(advance_in);
            out_ptr = out_ptr.byte_add(advance_out);

            last_word = advance_8byte_word(last_word, advance_in);
        }

        // in_ptr should have exceeded in_end
        assert!(in_ptr >= in_end, "exhausted output buffer before exhausting input, there is a bug in SymbolTable::compress()");

        // Count the number of bytes written
        // SAFETY: assertion
        let bytes_written = out_ptr.offset_from(values.as_ptr());
        assert!(
            bytes_written.is_positive(),
            "out_ptr ended before it started, not possible"
        );

        values.set_len(bytes_written as usize);
    }

    /// Use the symbol table to compress the plaintext into a sequence of codes and escapes.
    pub fn compress(&self, plaintext: &[u8]) -> Vec<u8> {
        if plaintext.is_empty() {
            return Vec::new();
        }

        let mut buffer = Vec::with_capacity(plaintext.len() * 2);

        // SAFETY: the largest compressed size would be all escapes == 2*plaintext_len
        unsafe { self.compress_into(plaintext, &mut buffer) };

        buffer
    }

    /// Access the decompressor that can be used to decompress strings emitted from this
    /// `Compressor` instance.
    pub fn decompressor(&self) -> Decompressor {
        Decompressor::new(self.symbol_table(), self.symbol_lengths())
    }

    /// Returns a readonly slice of the current symbol table.
    ///
    /// The returned slice will have length of `n_symbols`.
    pub fn symbol_table(&self) -> &[Symbol] {
        &self.symbols[0..self.n_symbols as usize]
    }

    /// Returns a readonly slice where index `i` contains the
    /// length of the symbol represented by code `i`.
    ///
    /// Values range from 1-8.
    pub fn symbol_lengths(&self) -> &[u8] {
        &self.lengths[0..self.n_symbols as usize]
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
    let mask = u64::MAX >> ignored_bits;
    (left & mask) == right
}
