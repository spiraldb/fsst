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

#![doc = include_str!("../README.md")]

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
/// identified by an 8-bit [`Code`].
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
    /// Calculate the length of the symbol in bytes.
    ///
    /// Each symbol has the capacity to hold up to 8 bytes of data, but the symbols
    /// can contain fewer bytes, padded with 0x00.
    #[inline(never)]
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

    #[inline]
    pub fn as_u64(&self) -> u64 {
        // SAFETY: the bytes can always be viewed as a u64
        unsafe { self.num }
    }

    /// Get the first byte of the symbol as a `u8`.
    ///
    /// # Safety
    /// The function will never panic, but if the symbol's len is < 1, the
    /// result may be meaningless. It is up to the caller to ensure that
    /// the first byte of the symbol contains valid data.
    #[inline]
    pub fn first_byte(&self) -> u8 {
        // SAFETY: the bytes can always be viewed as a u64
        unsafe { self.num as u8 }
    }

    /// Get the first two bytes of the symbol as a `u16`.
    ///
    /// # Safety
    /// The function will never panic, but if the symbol's len is < 2, the
    /// result may be meaningless. It is up to the caller to ensure that
    /// the first two bytes of the symbol contain valid data.
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
        let new_len = self.len() + other.len();
        assert!(new_len <= 8, "cannot build symbol with length > 8");

        let self_len = self.len();
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

/// Codes used to map symbols to bytes.
///
/// Logically, codes can range from 0-255 inclusive. Physically, we represent them as a 9-bit
/// value packed into a `u16`.
///
/// Physically in-memory, `Code(0)` through `Code(255)` corresponds to escape sequences of raw bytes
/// 0 through 255. `Code(256)` through `Code(511)` represent the actual codes -255.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Code(u16);

impl Code {
    /// Maximum value for the in-memory `Code` representation.
    ///
    /// When truncated to u8 this is code 255, which is equivalent to [`Self::ESCAPE_CODE`].
    pub const CODE_MAX: u16 = 511;

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

impl Debug for Code {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Code")
            .field("code_byte", &(self.0 as u8))
            .field("escape", &(self.0 < 256))
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
/// table.insert(Symbol::from_slice(&[b'h', b'e', b'l', b'l', b'o', 0, 0, 0]));
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
    codes_twobyte: [u16; 65_536],

    /// Lossy perfect hash table for looking up codes to symbols that are 3 bytes or more
    lossy_pht: LossyPHT,
}

impl Default for SymbolTable {
    fn default() -> Self {
        let mut table = Self {
            symbols: [Symbol::ZERO; 511],
            n_symbols: 0,
            codes_twobyte: [0; 65_536],
            lossy_pht: LossyPHT::new(),
        };

        // Populate the escape byte entries.
        for byte in 0..=255 {
            table.symbols[byte as usize] = Symbol::from_u8(byte);
        }

        // Populate the "codes" for twobytes to default to the escape sequence.
        for first in 0..256 {
            for second in 0..256 {
                let index = (first << 8) | second;
                table.codes_twobyte[index as usize] =
                    ((first << 8) | (Code::ESCAPE_CODE as usize)) as u16;
            }
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
        if symbol_len == 2 {
            // Speculatively insert the symbol into the twobyte cache
            self.codes_twobyte[symbol.first_two_bytes() as usize] = self.n_symbols as u16;
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

    /// Using the symbol table, runs a single cycle of compression from the front of `in_ptr`, writing
    /// the output into `out_ptr`.
    ///
    /// # Returns
    ///
    /// This function returns a tuple of (code, advance_in, advance_out).
    ///
    /// `code` is the code that was emitted into the output buffer.
    ///
    /// `advance_in` is the number of bytes to advance `in_ptr` before the next call.
    ///
    /// `advance_out` is the number of bytes to advance `out_ptr` before the next call.
    ///
    /// # Safety
    ///
    /// `in_ptr` and `out_ptr` must never be NULL or otherwise point to invalid memory.
    pub(crate) unsafe fn compress_single(
        &self,
        in_ptr: *const u8,
        out_ptr: *mut u8,
    ) -> (u8, usize, usize) {
        // Load a full 8-byte word of data from in_ptr.
        // SAFETY: caller asserts in_ptr is not null. we may read past end of pointer though.
        let word: u64 = unsafe { (in_ptr as *const u64).read_unaligned() };

        // Speculatively write the first byte of `word` at offset 1. This is necessary if it is an escape, and
        // if it isn't, it will be overwritten anyway.
        //
        // SAFETY: caller ensures out_ptr is not null
        let first_byte = word as u8;
        unsafe { out_ptr.byte_add(1).write_unaligned(first_byte) };

        // Access the hash table, and see if we have a match.
        let entry = self.lossy_pht.lookup(word);

        // Now, downshift the `word` and the `entry` to see if they align.
        let word_prefix =
            word >> (0xFF_FF_FF_FF_FF_FF_FF_FFu64 >> entry.packed_meta.ignored_bits());

        // This ternary-like branch corresponds to the "conditional move" line from the paper's Algorithm 4:
        // if the shifted word and symbol match, we use it. Else, we use the precomputed twobyte code for this
        // byte sequence.
        let code = if entry.symbol.as_u64() == word_prefix && !entry.packed_meta.is_unused() {
            entry.packed_meta.code() as u16
        } else {
            self.codes_twobyte[(word as u16) as usize]
        };
        // Write the first byte of `code` to the output position.
        // The code will either by a real code with a single byte of padding, OR a two-byte code sequence.
        unsafe {
            out_ptr.write_unaligned(code as u8);
        };

        // Seek the output pointer forward.
        let advance_in = (64 - entry.packed_meta.ignored_bits()) >> 3;
        let advance_out = 2 - ((code >> 8) & 1) as usize;

        (code as u8, advance_in, advance_out)
    }

    /// Use the symbol table to compress the plaintext into a sequence of codes and escapes.
    pub fn compress(&self, plaintext: &[u8]) -> Vec<u8> {
        let mut values: Vec<u8> = Vec::with_capacity(2 * plaintext.len());

        let mut in_ptr = plaintext.as_ptr();
        let mut out_ptr = values.as_mut_ptr();

        // SAFETY: `end` will point just after the end of the `plaintext` slice.
        let in_end = unsafe { in_ptr.byte_add(plaintext.len()) };
        // SAFETY: `end` will point just after the end of the `values` allocation.
        let out_end = unsafe { out_ptr.byte_add(values.capacity()) };

        while in_ptr < in_end && out_ptr < out_end {
            // SAFETY: pointer ranges are checked in the loop condition
            unsafe {
                let (_, advance_in, advance_out) = self.compress_single(in_ptr, out_ptr);
                in_ptr = in_ptr.byte_add(advance_in);
                out_ptr = out_ptr.byte_add(advance_out);
            };
        }

        // in_ptr should have exceeded in_end
        assert!(in_ptr >= in_end, "exhausted output buffer before exhausting input, there is a bug in SymbolTable::compress()");

        values
    }

    /// Decompress a byte slice that was previously returned by [compression][Self::compress].
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
