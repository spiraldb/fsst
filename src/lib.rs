//! A pure-Rust, zero-dependency implementation of the [FSST string compression algorithm][whitepaper].
//!
//! FSST is a string compression algorithm meant for use in database systems. It was designed by
//! [Peter Boncz, Thomas Neumann, and Viktor Leis][whitepaper]. It provides 1-3GB/sec compression
//! and decompression of strings at compression rates competitive with or better than LZ4.
//!
//! NOTE: This  current implementation is still in-progress, please use at your own risk.
//!
//! [whitepaper]: https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf

use std::fmt::{Debug, Formatter};

pub use builder::*;

mod builder;
mod longest;

/// A Symbol wraps a set of values of
#[derive(Copy, Clone)]
pub union Symbol {
    bytes: [u8; 8],
    num: u64,
}

impl Debug for Symbol {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", unsafe { self.num })
    }
}

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
    /// Calculate the length of the symbol in bytes.
    ///
    /// Each symbol has the capacity to hold up to 8 bytes of data, but the symbols
    /// can contain fewer bytes, padded with 0x00.
    pub fn len(&self) -> usize {
        let numeric = unsafe { self.num };
        // For little-endian platforms, this counts the number of *trailing* zeros
        let null_bytes = (numeric.leading_zeros() >> 3) as usize;

        size_of::<Self>() - null_bytes
    }

    /// Returns true if the symbol does not encode any bytes.
    ///
    /// Note that this should only be true for the zero code.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Create a ew
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
        let new_len = self.len() + other.len();
        assert!(new_len <= 8, "cannot build symbol with length > 8");

        let self_len = self.len();
        let mut result = *self;
        unsafe { result.bytes[self_len..new_len].copy_from_slice(other.as_slice()) };

        result
    }
}

/// Codes used to map symbols to bytes.
///
/// Logically, codes can range from 0-255 inclusive. Physically, we represent them as a 9-bit
/// value packed into a `u16`.
///
/// Physically in-memory, `Code(0)` through `Code(255)` corresponds to escape sequences of raw bytes
/// 0 through 255. `Code(256)` through `Code(511)` represent the actual codes -255.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Code(u16);

impl Code {
    /// Maximum code value for the in-memory `Code` representation.
    pub const CODE_MAX: u16 = 512;

    /// Maximum code value. Code 255 is reserved as the [escape code][`Self::ESCAPE_CODE`].
    pub const MAX_CODE: u8 = 254;

    /// Code used to indicate bytes that are not in the symbol table.
    ///
    /// When compressing a string that cannot fully be expressed with the symbol table, the compressed
    /// output will contain an `ESCAPE` byte followed by a raw byte. At decompression time, the presence
    /// of `ESCAPE` indicates that the next byte should be appended directly to the result instead of
    /// being looked up in the symbol table.
    pub const ESCAPE_CODE: u8 = 255;

    /// Create a new code representing an escape byte.
    pub fn new_escaped(byte: u8) -> Self {
        Self(byte as u16)
    }

    /// Create a new code representing a symbol.
    pub fn new_symbol(code: u8) -> Self {
        assert_ne!(
            code,
            Code::ESCAPE_CODE,
            "code {code} cannot be used for symbol, reserved for ESCAPE"
        );

        Self((code as u16) + 256)
    }

    /// Create a `Code` directly from a `u16` value.
    ///
    /// # Panics
    /// Panic if the value is ≥ the defined `CODE_MAX`.
    pub fn from_u16(code: u16) -> Self {
        assert!(code < Self::CODE_MAX, "code value higher than CODE_MAX");

        Self(code)
    }

    /// Returns true if the code is for an escape byte.
    #[inline]
    pub fn is_escape(&self) -> bool {
        self.0 <= 255
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
/// table.insert(Symbol::from_slice(&[b'h', b'e', b'l', b'l', b'o', 0, 0, 0]));
///
/// let compressed = table.compress("hello".as_bytes());
/// assert_eq!(compressed, vec![0u8]);
/// ```
///
/// training: [`train`]
#[derive(Clone, Debug)]
pub struct SymbolTable {
    /// Table mapping codes to symbols.
    pub(crate) symbols: [Symbol; 511],

    /// Indicates the number of entries in the symbol table that have been populated.
    ///
    /// This value is always at least 256, as the first 256 entries in the `table` are the escape
    /// bytes.
    pub(crate) n_symbols: usize,
}

impl Default for SymbolTable {
    fn default() -> Self {
        let mut table = Self {
            symbols: [Symbol::ZERO; 511],
            n_symbols: 0,
        };

        // Populate the escape byte entries.
        for byte in 0..=255 {
            table.symbols[byte as usize] = Symbol::from_u8(byte);
        }
        table.n_symbols = 256;

        table
    }
}

/// The core structure of the FSST codec, holding a mapping between `Symbol`s and `Code`s.
///
/// The symbol table is trained on a corpus of data in the form of a single byte array, building up
/// a mapping of 1-byte "codes" to sequences of up to `N` plaintext bytse, or "symbols".
impl SymbolTable {
    /// Insert a new symbol at the end of the table.
    ///
    /// # Panics
    /// Panics if the table is already full.
    pub fn insert(&mut self, symbol: Symbol) {
        assert!(
            self.n_symbols < self.symbols.len(),
            "cannot insert into full symbol table"
        );
        self.symbols[self.n_symbols] = symbol;
        self.n_symbols += 1;
    }

    /// Return a new encoded sequence of data bytes instead.
    pub fn compress(&self, plaintext: &[u8]) -> Vec<u8> {
        let mut values = Vec::with_capacity(2 * plaintext.len());
        let len = plaintext.len();
        let mut pos = 0;
        while pos < len {
            // println!("COMPRESS pos={pos} len={len} in_progress_size={}", values.len());
            let next_code = self.find_longest_symbol(&plaintext[pos..len]);
            if next_code.is_escape() {
                // Case 1 -escape: push an ESCAPE followed by the next byte.
                // println!("ESCAPE");
                values.push(Code::ESCAPE_CODE);
                values.push(next_code.0 as u8);
                pos += 1;
            } else {
                // Case 2 - code: push the code, increment position by symbol length
                let symbol = self.symbols[next_code.0 as usize];
                // println!("APPEND symbol={:?} len={}", symbol.as_slice(), symbol.len());
                values.push(next_code.0 as u8);
                pos += symbol.len();
            }
        }

        values
    }

    /// Decompress the provided byte slice into a [`String`] using the symbol table.
    pub fn decompress(&self, compressed: &[u8]) -> Vec<u8> {
        let mut decoded: Vec<u8> = Vec::with_capacity(size_of::<Symbol>() * compressed.len());
        let ptr = decoded.as_mut_ptr();

        let mut in_pos = 0;
        let mut out_pos = 0;

        while in_pos < compressed.len() && out_pos < (decoded.capacity() + size_of::<Symbol>()) {
            let code = compressed[in_pos];
            if code == Code::ESCAPE_CODE {
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
