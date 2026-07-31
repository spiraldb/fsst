//! Test to demonstrate successful decompression on exact capacity bounds without panicking

#![cfg(test)]

use fsst::{Compressor, CompressorBuilder, Decompressor, ESCAPE_CODE, Symbol};
use hegel::generators::{self, Generator};

const SYMBOL_BYTES: [[u8; 8]; 8] = [
    [b'a', 0, 0, 0, 0, 0, 0, 0],
    [b'b', b'c', 0, 0, 0, 0, 0, 0],
    [b'd', b'e', b'f', 0, 0, 0, 0, 0],
    [b'g', b'h', b'i', b'j', 0, 0, 0, 0],
    [b'k', b'l', b'm', b'n', b'o', 0, 0, 0],
    [b'p', b'q', b'r', b's', b't', b'u', 0, 0],
    [b'v', b'w', b'x', b'y', b'z', b'0', b'1', 0],
    *b"23456789",
];
const SYMBOL_LENGTHS: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

fn padded_symbol_table() -> ([Symbol; 255], [u8; 255]) {
    let mut symbols = [Symbol::ZERO; 255];
    let mut lengths = [0; 255];

    for (idx, bytes) in SYMBOL_BYTES.iter().enumerate() {
        symbols[idx] = Symbol::from_slice(bytes);
        lengths[idx] = SYMBOL_LENGTHS[idx];
    }

    (symbols, lengths)
}

#[derive(Clone, Copy, Debug)]
enum Token {
    Symbol(u8),
    Escape(u8),
}

fn compressor_with_all_symbol_lengths() -> Compressor {
    let mut builder = CompressorBuilder::new();
    for (bytes, len) in SYMBOL_BYTES.iter().zip(SYMBOL_LENGTHS) {
        assert!(builder.insert(Symbol::from_slice(bytes), len as usize));
    }
    builder.build()
}

fn draw_tokens(tc: &hegel::TestCase) -> Vec<Token> {
    let token_count = tc.draw(generators::integers::<usize>().max_value(256));
    tc.draw(
        generators::vecs(hegel::one_of!(
            generators::integers::<u8>()
                .max_value((SYMBOL_BYTES.len() - 1) as u8)
                .map(Token::Symbol),
            generators::integers::<u8>().map(Token::Escape),
        ))
        .min_size(token_count)
        .max_size(token_count),
    )
}

fn model_stream(tokens: &[Token]) -> (Vec<u8>, Vec<u8>) {
    let mut compressed = Vec::new();
    let mut plaintext = Vec::new();
    for &token in tokens {
        match token {
            Token::Symbol(code) => {
                compressed.push(code);
                let len = SYMBOL_LENGTHS[code as usize] as usize;
                plaintext.extend_from_slice(&SYMBOL_BYTES[code as usize][..len]);
            }
            Token::Escape(byte) => {
                compressed.extend_from_slice(&[ESCAPE_CODE, byte]);
                plaintext.push(byte);
            }
        }
    }
    (compressed, plaintext)
}

fn decompress_with_capacity(
    decompressor: &Decompressor<'_>,
    compressed: &[u8],
    capacity: usize,
) -> Vec<u8> {
    let mut decompressed = Vec::with_capacity(capacity);
    let len = decompressor.decompress_into(compressed, decompressed.spare_capacity_mut());
    // SAFETY: decompress_into initialized exactly len bytes.
    unsafe { decompressed.set_len(len) };
    decompressed
}

#[test]
fn test_decompress_exact_capacity() {
    // Train a compressor with a symbol that expands significantly
    let compressor = {
        let mut builder = CompressorBuilder::new();
        // Insert a highly compressible 8-byte symbol
        builder.insert(Symbol::from_slice(b"aaaaaaaa"), 8);
        builder.build()
    };

    // Create a large, highly compressible string. Under miri a much smaller buffer still
    // exercises both the 8-byte block loop and the byte-by-byte tail fallback this test targets,
    // and miri's cost here is proportional to the buffer size.
    let plaintext = vec![b'a'; if cfg!(miri) { 2_000 } else { 100_000 }];

    // Compress it into an over-allocated buffer to avoid any compression bugs
    let mut compressed = Vec::with_capacity(plaintext.len() * 2);
    unsafe {
        compressor.compress_into(&plaintext, &mut compressed);
    }

    let decompressor = compressor.decompressor();

    // If the caller allocates EXACTLY the uncompressed length (which is theoretically
    // sufficient and correct), `decompress_into` should successfully decode the entire
    // stream using the safe byte-by-byte fallback loop without early termination.
    let mut decompressed = Vec::with_capacity(plaintext.len());
    let spare = decompressed.spare_capacity_mut();

    // This previously panicked with `exhaust input before output`. It should now succeed.
    let len = decompressor.decompress_into(&compressed, spare);
    unsafe { decompressed.set_len(len) };

    assert_eq!(decompressed, plaintext);
}

#[cfg_attr(miri, ignore)]
#[hegel::test]
fn decompress_mixed_stream_matches_model(tc: hegel::TestCase) {
    let (symbols, lengths) = padded_symbol_table();
    let decompressor = Decompressor::new(&symbols, &lengths);

    let tokens = draw_tokens(&tc);
    let (compressed, expected) = model_stream(&tokens);

    tc.target(compressed.len() as f64);
    tc.target(expected.len() as f64);

    let exact = decompress_with_capacity(&decompressor, &compressed, expected.len());
    let overallocated = decompress_with_capacity(
        &decompressor,
        &compressed,
        decompressor.max_decompression_capacity(&compressed) + 7,
    );

    assert_eq!((exact, overallocated), (expected.clone(), expected));
}

#[cfg_attr(miri, ignore)]
#[hegel::test]
fn compress_into_empty_clears_output(tc: hegel::TestCase) {
    let compressor = CompressorBuilder::new().build();
    let mut output = tc.draw(generators::binary());

    // SAFETY: empty input requires no output capacity.
    unsafe { compressor.compress_into(&[], &mut output) };

    assert!(output.is_empty());
}

#[cfg_attr(miri, ignore)]
#[hegel::test]
fn decompress_empty_stream_is_empty(tc: hegel::TestCase) {
    let (symbols, lengths) = padded_symbol_table();
    let decompressor = Decompressor::new(&symbols, &lengths);
    let capacity = tc.draw(generators::integers::<usize>().max_value(256));

    assert!(decompress_with_capacity(&decompressor, &[], capacity).is_empty());
}

#[cfg_attr(miri, ignore)]
#[hegel::test]
fn compress_into_exact_capacity_matches_compress(tc: hegel::TestCase) {
    let compressor = compressor_with_all_symbol_lengths();
    let tokens = tc.draw(
        generators::vecs(hegel::one_of!(
            generators::integers::<u8>()
                .max_value((SYMBOL_BYTES.len() - 1) as u8)
                .map(Token::Symbol),
            generators::integers::<u8>().map(Token::Escape),
        ))
        .max_size(256),
    );
    let (_, plaintext) = model_stream(&tokens);

    let expected = compressor.compress(&plaintext);
    let mut exact = Vec::with_capacity(expected.len());
    // SAFETY: exact has the capacity required by the canonical compression result.
    unsafe { compressor.compress_into(&plaintext, &mut exact) };

    assert_eq!(exact, expected);
}
